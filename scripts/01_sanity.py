"""Gate 1: the trace contract and the attribution identity must hold.

Runs on a small prompt sample so failures surface in seconds.

  1. h[l+1] == h[l] + attn_write[l] + ffn_write[l] at the final position.
  2. attn_write == W_O @ heads and ffn_write == D @ act (the atom decompositions).
  3. sum of atom loads == measured change in projection onto the refusal
     direction, per layer, for both the attention and FFN sides.
  4. how concentrated the refusal load is across atoms, and how much of the
     weight change a sparse edit needs to capture most of it.
"""

from __future__ import annotations

import json
import sys

import torch

from libershift import attribute, check_identity, collect, concentration, directions, load_model, separation
from libershift.attribute import cost_curve
from libershift.capture import Tracer
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_fit_prompts, good_fit_prompts

N_PROMPTS = 32


def main() -> None:
    lm = load_model(QWEN_1_5B.path)
    print(f"loaded {QWEN_1_5B.name}: {lm.n_layers} layers, d={lm.hidden_size}, "
          f"heads={lm.n_heads}x{lm.head_dim}")

    # --- 1 and 2: additive and atom-decomposition identities on a few prompts.
    tracer = Tracer(lm)
    cap = tracer.capture(good_fit_prompts()[:4])
    tracer.close()
    rel_add, rel_attn, rel_ffn = 0.0, 0.0, 0.0
    for l in range(lm.n_layers):
        recon = cap.residual[l] + cap.attn[l] + cap.ffn[l]
        rel_add = max(rel_add, float((recon - cap.residual[l + 1]).norm() / cap.residual[l + 1].norm()))
        attn_recon = cap.heads[l] @ lm.attn_out(l).weight.detach().float().T
        rel_attn = max(rel_attn, float((attn_recon - cap.attn[l]).norm() / cap.attn[l].norm()))
        ffn_recon = cap.act[l] @ lm.ffn_down(l).weight.detach().float().T
        rel_ffn = max(rel_ffn, float((ffn_recon - cap.ffn[l]).norm() / cap.ffn[l].norm()))
    print(f"relative error  h[l+1] vs h[l]+attn+ffn : {rel_add:.2e}")
    print(f"relative error  attn_write vs W_O@heads  : {rel_attn:.2e}")
    print(f"relative error  ffn_write  vs D@act     : {rel_ffn:.2e}")

    # --- 3 and 4: attribution on a small sample.
    good = collect(lm, good_fit_prompts()[:N_PROMPTS])
    bad = collect(lm, bad_fit_prompts()[:N_PROMPTS])
    dirs = directions(bad.h, good.h)
    attr = attribute(lm, good, bad, dirs)
    ident = check_identity(attr, dirs, good, bad)
    print(f"identity: atom sum vs measured writes   : "
          f"{ident['max_relative_error_atom_sum_vs_writes']:.2e}")
    print(f"identity: residual vs carried+writes    : "
          f"{ident['max_relative_error_residual_vs_sum']:.2e}")

    sep = separation(dirs, bad.h_prompts, good.h_prompts)
    d = [s["cohens_d"] for s in sep]
    best = int(torch.tensor(d).abs().argmax())
    print(f"best separation at plane {best}: cohen_d={d[best]:.2f}")
    print("cohen_d by plane:", " ".join(f"{x:.1f}" for x in d))

    per_layer = (attr.feature_load.sum(1) + attr.head_load.sum(1)).tolist()
    print("refusal load written per layer:")
    print(" ".join(f"{x:.1f}" for x in per_layer))

    feat = concentration(attr.feature_load.clamp_min(0))
    heads = concentration(attr.head_load.clamp_min(0))
    print("\nFFN feature load concentration:", json.dumps(feat, indent=1))
    print("attention head load concentration:", json.dumps(heads, indent=1))
    print("\nsparse cost curve (all atoms pooled):")
    print(json.dumps(cost_curve(attr.feature_load, attr.feature_cost), indent=1))
    print("sparse cost curve (attention only):")
    print(json.dumps(cost_curve(attr.head_load, attr.head_cost), indent=1))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sanity.json").write_text(json.dumps({
        "additive_relative_error": rel_add,
        "attn_atom_relative_error": rel_attn,
        "ffn_atom_relative_error": rel_ffn,
        "identity": ident,
        "separation": sep,
        "best_plane": best,
        "per_layer_load": per_layer,
        "feature_concentration": feat,
        "head_concentration": heads,
        "feature_cost_curve": cost_curve(attr.feature_load, attr.feature_cost),
        "head_cost_curve": cost_curve(attr.head_load, attr.head_cost),
        "n_prompts": N_PROMPTS,
    }, indent=2))
    print(f"\nwrote {OUT / 'sanity.json'}")


if __name__ == "__main__":
    sys.exit(main())
