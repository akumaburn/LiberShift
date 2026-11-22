"""Exact load attribution of the refusal direction onto write atoms.

A transformer block writes into the residual stream through a sum of atoms:

    attn_write  = sum_h  W_O[:, h] @ o_h          (one atom per query head)
    ffn_write   = sum_f  c_f * d_f                (one atom per FFN feature)

Both decompositions are exact, not approximate: ``o_h`` is a column block of
``o_proj``'s input and ``c_f`` is a row of ``down_proj``'s input.  Because the
projection onto a direction is linear, the change in the mean projection caused
by an atom is exactly

    lambda_u = (mean_bad[c_u] - mean_good[c_u]) * (w_u . r)

where ``w_u`` is the atom's write vector and ``r`` the refusal direction.  The
lambda values therefore sum to the layer's measured change in refusal
projection.  ``check_identity`` verifies that agreement numerically; it is the
main correctness gate for the whole method.

``cost`` is the squared Frobenius norm of the matrix change that ablating the
atom would make (``||r a^T||_F^2 = ||a||^2``), so costs are additive across
layers and comparable between heads and features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .capture import Stats
from .model import LoadedModel


@dataclass
class Attribution:
    # [L, n_features]
    feature_load: Tensor
    feature_alpha: Tensor
    feature_cost: Tensor
    feature_t: Tensor
    # [L, n_heads]
    head_load: Tensor
    head_cost: Tensor
    head_t: Tensor
    # [L+1, hidden]
    residual_direction: Tensor
    # diagnostics: measured change in projection, per layer
    measured_attn_proj: Tensor
    measured_ffn_proj: Tensor
    measured_residual_delta: Tensor
    n_bad: int
    n_good: int


def _welch_t(mean_bad: Tensor, mean_good: Tensor, var_bad: Tensor, var_good: Tensor,
             n_bad: int, n_good: int) -> Tensor:
    se = torch.sqrt(var_bad / n_bad + var_good / n_good).clamp_min(1e-12)
    return (mean_bad - mean_good) / se


def attribute(lm: LoadedModel, good: Stats, bad: Stats, residual_direction: Tensor) -> Attribution:
    """Assign refusal load to every head and every FFN feature of every layer."""
    n_layers = lm.n_layers
    n_feat = lm.ffn_down(0).weight.shape[1]
    n_heads = lm.n_heads
    head_dim = lm.head_dim
    hidden = lm.hidden_size

    f_load = torch.zeros(n_layers, n_feat, dtype=torch.float32)
    f_alpha = torch.zeros_like(f_load)
    f_cost = torch.zeros_like(f_load)
    f_t = torch.zeros_like(f_load)
    h_load = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    h_cost = torch.zeros_like(h_load)
    h_t = torch.zeros_like(h_load)

    n_good, n_bad = good.count, bad.count
    attn_proj = torch.zeros(n_layers)
    ffn_proj = torch.zeros(n_layers)
    resid_delta = torch.zeros(n_layers)

    for l in range(n_layers):
        r = residual_direction[l + 1].float()

        # FFN: write vector of feature f is column f of down_proj.
        down = lm.ffn_down(l).weight.detach().float().cpu()  # [hidden, n_feat]
        alpha = down.T @ r  # [n_feat] == (d_f . r)
        d_act = (bad.act[l] - good.act[l]).float()
        f_load[l] = d_act * alpha
        f_alpha[l] = alpha
        f_cost[l] = alpha**2
        f_t[l] = _welch_t(
            bad.act[l].float(), good.act[l].float(),
            bad.act_var()[l].float(), good.act_var()[l].float(), n_bad, n_good,
        )

        # Attention: write vector block of head h is W_O[:, h*Dh:(h+1)*Dh].
        o = lm.attn_out(l).weight.detach().float().cpu()  # [hidden, n_heads*head_dim]
        beta = o.T @ r  # [n_heads*head_dim] == W_h^T r, flattened
        d_heads = (bad.heads[l] - good.heads[l]).float()
        h_load[l] = (d_heads * beta).view(n_heads, head_dim).sum(dim=1)
        h_cost[l] = (beta**2).view(n_heads, head_dim).sum(dim=1)
        h_t[l] = _welch_t(
            bad.heads[l].float(), good.heads[l].float(),
            bad.heads_var()[l].float(), good.heads_var()[l].float(), n_bad, n_good,
        ).view(n_heads, head_dim).mean(dim=1)

        attn_proj[l] = float((bad.attn[l] - good.attn[l]).float() @ r)
        ffn_proj[l] = float((bad.ffn[l] - good.ffn[l]).float() @ r)
        resid_delta[l] = float((bad.h[l + 1] - good.h[l + 1]).float() @ r)

    return Attribution(
        feature_load=f_load,
        feature_alpha=f_alpha,
        feature_cost=f_cost,
        feature_t=f_t,
        head_load=h_load,
        head_cost=h_cost,
        head_t=h_t,
        residual_direction=residual_direction,
        measured_attn_proj=attn_proj,
        measured_ffn_proj=ffn_proj,
        measured_residual_delta=resid_delta,
        n_bad=n_bad,
        n_good=n_good,
    )


def check_identity(attr: Attribution, residual_direction: Tensor, good: Stats, bad: Stats) -> dict:
    """Verify that attributed loads reconstruct the measured projections.

    Returns the largest relative error over layers, plus the per-layer tables.
    A large error means the atom decomposition or the direction is wrong, so
    every result downstream would be meaningless.
    """
    n_layers = attr.feature_load.shape[0]
    down_sum = attr.feature_load.sum(dim=1) + attr.head_load.sum(dim=1)
    worst_attn = worst_ffn = 0.0
    for l in range(n_layers):
        r = residual_direction[l + 1].float()
        measured = float(attr.measured_attn_proj[l] + attr.measured_ffn_proj[l])
        scale = max(abs(float(attr.measured_attn_proj[l])) + abs(float(attr.measured_ffn_proj[l])), 1e-6)
        worst_ffn = max(worst_ffn, abs(float(down_sum[l]) - measured) / scale)
        # carry-over check: mean residual delta must equal carry + writes
        carry = float((bad.h[l] - good.h[l]).float() @ r)
        resid = carry + float(attr.measured_attn_proj[l]) + float(attr.measured_ffn_proj[l])
        delta = float(attr.measured_residual_delta[l])
        worst_attn = max(worst_attn, abs(resid - delta) / max(abs(delta), 1e-6))
    return {
        "max_relative_error_atom_sum_vs_writes": worst_ffn,
        "max_relative_error_residual_vs_sum": worst_attn,
        "per_layer_attributed": down_sum.tolist(),
        "per_layer_measured_attn": attr.measured_attn_proj.tolist(),
        "per_layer_measured_ffn": attr.measured_ffn_proj.tolist(),
    }


def cost_curve(loads: Tensor, costs: Tensor, targets: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)) -> dict:
    """Cost of capturing a fraction of the positive refusal load.

    Atoms are taken in order of load per unit cost.  The returned cost is
    expressed as a fraction of the cost of ablating *every* positive-load atom,
    which is the dense alternative.  A small number means a sparse edit can do
    most of the work a dense edit would do.
    """
    load = loads.flatten()
    cost = costs.flatten()
    keep = load > 0
    load, cost = load[keep], cost[keep]
    if load.numel() == 0:
        return {}
    order = torch.argsort(load / cost.clamp_min(1e-12), descending=True)
    load_cum = load[order].cumsum(0) / load.sum()
    cost_cum = cost[order].cumsum(0) / cost.sum()
    out = {"n_atoms": int(load.numel())}
    for target in targets:
        idx = int(torch.searchsorted(load_cum, torch.tensor(target)).clamp(max=load.numel() - 1))
        out[f"cost_for_{target:.2f}_load"] = float(cost_cum[idx])
        out[f"atoms_for_{target:.2f}_load"] = idx + 1
        out[f"atom_fraction_for_{target:.2f}_load"] = (idx + 1) / load.numel()
    return out



def concentration(loads: Tensor) -> dict[str, float]:
    """How concentrated positive load is: share held by the top 1/5/10/25%."""
    x = loads.flatten()
    x = x[x > 0]
    if x.numel() == 0:
        return {"n_positive": 0}
    x, _ = torch.sort(x, descending=True)
    total = float(x.sum())
    out = {"n_positive": int(x.numel()), "total_positive_load": total}
    for frac in (0.001, 0.005, 0.01, 0.05, 0.10, 0.25):
        k = max(1, int(round(frac * x.numel())))
        out[f"top_{frac:.3f}_share"] = float(x[:k].sum() / total)
    # Normalised entropy: 1.0 means load spread evenly over every atom.
    p = x / total
    out["entropy_over_log_n"] = float(-(p * p.log()).sum() / torch.log(torch.tensor(float(x.numel()))))
    return out
