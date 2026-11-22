"""Residual capture under the additive residual contract.

For every decoder block ``l`` the residual stream obeys

    h[l+1] = h[l] + attn_write[l] + ffn_write[l]

This module records exactly those three quantities at the final prompt
position, plus the two vectors whose sparse combination produces the writes:

* ``heads`` -- the input to ``o_proj``, i.e. the concatenation of per-head
  context vectors ``o_h``.  ``attn_write = W_O @ heads``.
* ``act`` -- the input to ``down_proj``, i.e. the per-feature activation
  coefficients ``c_f``.  ``ffn_write = D @ act``, and column ``f`` of ``D`` is
  the vector feature ``f`` writes into the residual stream.

The captures are hooks, so they work for any HF decoder with this block shape
and do not depend on the attention kernel (sdpa/eager/flash) in use.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor

from .model import LoadedModel


class LayerCapture:
    """One prompt batch's worth of captures, last position only.

    ``residual`` holds the pre-norm residual plane entering each block plus the
    reconstructed plane after the final block (``L + 1`` planes).  The
    reconstruction is used because transformers 5.x returns the final plane
    *after* the model's closing norm, so it is not in the same space as the
    planes below it; ``residual[l+1] = residual[l] + attn[l] + ffn[l]`` would not
    hold for the returned value.  Reconstructing keeps every plane additive.
    """

    def __init__(self, residual: list[Tensor], attn: list[Tensor], ffn: list[Tensor],
                 heads: list[Tensor], act: list[Tensor]) -> None:
        self.residual = residual
        self.attn = attn
        self.ffn = ffn
        self.heads = heads
        self.act = act


class Tracer:
    def __init__(self, lm: LoadedModel) -> None:
        self.lm = lm
        self._buf: dict[tuple[str, int], Tensor] = {}
        self._handles = []
        for i, layer in enumerate(lm.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                attn = layer.linear_attn
            out_proj = getattr(attn, "o_proj", None)
            if out_proj is None:
                out_proj = attn.out_proj
            mlp = layer.mlp
            self._handles.append(attn.register_forward_hook(partial(self._out, "attn", i)))
            self._handles.append(mlp.register_forward_hook(partial(self._out, "ffn", i)))
            self._handles.append(
                out_proj.register_forward_pre_hook(partial(self._in, "heads", i))
            )
            self._handles.append(
                mlp.down_proj.register_forward_pre_hook(partial(self._in, "act", i))
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _out(self, name: str, index: int, _module, _args, output) -> None:
        self._buf[(name, index)] = output[0] if isinstance(output, tuple) else output

    def _in(self, name: str, index: int, _module, args) -> None:
        self._buf[(name, index)] = args[0]

    def capture(self, prompts: list[str]) -> LayerCapture:
        enc = self.lm.encode(prompts)
        self._buf.clear()
        with torch.no_grad():
            out = self.lm.model(**enc, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states
        n = self.lm.n_layers
        last = lambda t: t[:, -1, :].float()
        residual = [last(hidden[l]) for l in range(n)]
        attn = [last(self._buf[("attn", l)]) for l in range(n)]
        ffn = [last(self._buf[("ffn", l)]) for l in range(n)]
        residual.append(residual[-1] + attn[-1] + ffn[-1])
        cap = LayerCapture(
            residual=residual,
            attn=attn,
            ffn=ffn,
            heads=[last(self._buf[("heads", l)]) for l in range(n)],
            act=[last(self._buf[("act", l)]) for l in range(n)],
        )
        self._buf.clear()
        return cap


class Accumulator:
    """Sums per-layer captures over prompts; means and variances come out."""

    def __init__(self, n_layers: int, hidden: int, features: int, o_dim: int) -> None:
        self.n_layers = n_layers
        self.count = 0
        z = lambda *shape: torch.zeros(*shape, dtype=torch.float64)
        self.sum_h = z(n_layers + 1, hidden)
        self.sum_h_sq = z(n_layers + 1, hidden)
        self.sum_attn = z(n_layers, hidden)
        self.sum_ffn = z(n_layers, hidden)
        self.sum_act = z(n_layers, features)
        self.sum_act_sq = z(n_layers, features)
        self.sum_heads = z(n_layers, o_dim)
        self.sum_heads_sq = z(n_layers, o_dim)
        self.h_prompts: list[torch.Tensor] = []

    def add(self, cap: LayerCapture) -> None:
        b = cap.residual[0].shape[0]
        self.count += b
        self.h_prompts.append(torch.stack([p.cpu() for p in cap.residual], dim=1))
        for plane in range(self.n_layers + 1):
            self.sum_h[plane] += cap.residual[plane].double().sum(0).cpu()
            self.sum_h_sq[plane] += (cap.residual[plane].double() ** 2).sum(0).cpu()
        for l in range(self.n_layers):
            self.sum_attn[l] += cap.attn[l].double().sum(0).cpu()
            self.sum_ffn[l] += cap.ffn[l].double().sum(0).cpu()
            self.sum_act[l] += cap.act[l].double().sum(0).cpu()
            self.sum_act_sq[l] += (cap.act[l].double() ** 2).sum(0).cpu()
            self.sum_heads[l] += cap.heads[l].double().sum(0).cpu()
            self.sum_heads_sq[l] += (cap.heads[l].double() ** 2).sum(0).cpu()

    def mean(self) -> "Stats":
        n = float(self.count)
        return Stats(
            h=self.sum_h / n,
            attn=self.sum_attn / n,
            ffn=self.sum_ffn / n,
            act=self.sum_act / n,
            act_sq=self.sum_act_sq / n,
            heads=self.sum_heads / n,
            heads_sq=self.sum_heads_sq / n,
            h_sq=self.sum_h_sq / n,
            count=self.count,
            h_prompts=torch.cat(self.h_prompts, dim=0),
        )


class Stats:
    """Per-layer first and second moments, float64, on the CPU."""

    def __init__(self, h: Tensor, attn: Tensor, ffn: Tensor, act: Tensor,
                 act_sq: Tensor, heads: Tensor, heads_sq: Tensor, h_sq: Tensor,
                 count: int, h_prompts: Tensor) -> None:
        self.h = h
        self.attn = attn
        self.ffn = ffn
        self.act = act
        self.act_sq = act_sq
        self.heads = heads
        self.heads_sq = heads_sq
        self.h_sq = h_sq
        self.count = count
        # [prompts, planes, hidden] -- only used for diagnostics (separation).
        self.h_prompts = h_prompts

    def act_var(self) -> Tensor:
        return (self.act_sq - self.act**2).clamp_min(0.0)

    def heads_var(self) -> Tensor:
        return (self.heads_sq - self.heads**2).clamp_min(0.0)

    def save(self, path) -> None:
        torch.save(
            {
                "h": self.h,
                "attn": self.attn,
                "ffn": self.ffn,
                "act": self.act,
                "act_sq": self.act_sq,
                "heads": self.heads,
                "heads_sq": self.heads_sq,
                "h_sq": self.h_sq,
                "count": self.count,
                "h_prompts": self.h_prompts,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "Stats":
        d = torch.load(path, weights_only=True)
        return cls(**d)


def collect(lm: LoadedModel, prompts: list[str], batch_size: int | None = None) -> Stats:
    if batch_size is None:
        batch_size = lm.capture_batch_size
    tracer = Tracer(lm)
    down = lm.ffn_down(0).weight
    feature_dim = down.shape[1]
    o_dim = lm.attn_out(0).weight.shape[1]
    acc = Accumulator(lm.n_layers, lm.hidden_size, feature_dim, o_dim)
    try:
        for i in range(0, len(prompts), batch_size):
            acc.add(tracer.capture(prompts[i : i + batch_size]))
    finally:
        tracer.close()
    assert acc.count > 0
    return acc.mean()
