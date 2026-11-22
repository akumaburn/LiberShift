"""Per-layer refusal directions from captured residual means.

``r_l`` is the difference of mean residuals, ``mean_bad - mean_good``, at plane
``l`` (plane 0 is the embedding).  With ``projected=True`` the component along
the mean *good* residual is removed first, which is the "projected
abliteration" correction: it keeps the edit from subtracting directions the
model needs for ordinary prompts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def directions(bad: Tensor, good: Tensor, projected: bool = True) -> Tensor:
    """Return ``[n_planes, hidden]`` unit refusal directions."""
    diff = bad - good
    if projected:
        g = F.normalize(good, p=2, dim=1)
        proj = (diff * g).sum(dim=1, keepdim=True)
        diff = diff - proj * g
    return F.normalize(diff, p=2, dim=1)


def separation(direction: Tensor, bad: Tensor, good: Tensor) -> list[dict[str, float]]:
    """Per-plane standardised mean gap along each plane's direction."""
    out: list[dict[str, float]] = []
    for plane in range(direction.shape[0]):
        pb = bad[:, plane, :].float() @ direction[plane].float()
        pg = good[:, plane, :].float() @ direction[plane].float()
        pooled = torch.sqrt(
            (pb.var(unbiased=True) + pg.var(unbiased=True)) / 2
        ).clamp_min(1e-12)
        out.append(
            {
                "mean_bad": float(pb.mean()),
                "mean_good": float(pg.mean()),
                "mean_gap": float(pb.mean() - pg.mean()),
                "cohens_d": float((pb.mean() - pg.mean()) / pooled),
            }
        )
    return out
