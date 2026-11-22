"""Selection and application of sparse ablation edits.

Selection maximises refusal load removed per unit of matrix perturbation.  The
perturbation caused by ablating one atom is ``-t * a * r`` with ``||a||`` its
own norm, so the squared Frobenius norm removed is exactly ``cost``.  Ranking
by ``load / cost`` is the greedy solution of "remove at least L load while
changing the weights as little as possible".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .attribute import Attribution
from .model import LoadedModel

RENORM_GUARD = 0.25


@dataclass
class Atom:
    layer: int
    kind: str  # "feature" | "head"
    index: int
    load: float
    cost: float
    t_stat: float


@dataclass
class Selection:
    atoms: list[Atom]
    total_positive_load: float
    captured_load: float
    total_cost: float
    candidates: int

    @property
    def captured_fraction(self) -> float:
        return self.captured_load / self.total_positive_load if self.total_positive_load else 0.0

    def summary(self) -> dict:
        feats = [a for a in self.atoms if a.kind == "feature"]
        heads = [a for a in self.atoms if a.kind == "head"]
        return {
            "atoms": len(self.atoms),
            "feature_atoms": len(feats),
            "head_atoms": len(heads),
            "candidates": self.candidates,
            "captured_load": self.captured_load,
            "total_positive_load": self.total_positive_load,
            "captured_fraction": self.captured_fraction,
            "total_cost": self.total_cost,
            "layers_touched": len({a.layer for a in self.atoms}),
        }


def _candidates(attr: Attribution, kinds: tuple[str, ...], min_t: float) -> list[Atom]:
    out: list[Atom] = []
    n_layers = attr.feature_load.shape[0]
    for l in range(n_layers):
        if "feature" in kinds:
            load = attr.feature_load[l]
            cost = attr.feature_cost[l]
            t = attr.feature_t[l]
            idx = torch.nonzero((load > 0) & (t >= min_t)).flatten()
            for i in idx.tolist():
                out.append(Atom(l, "feature", i, float(load[i]), float(cost[i]), float(t[i])))
        if "head" in kinds:
            load = attr.head_load[l]
            cost = attr.head_cost[l]
            t = attr.head_t[l]
            idx = torch.nonzero((load > 0) & (t >= min_t)).flatten()
            for i in idx.tolist():
                out.append(Atom(l, "head", i, float(load[i]), float(cost[i]), float(t[i])))
    return out


def select(
    attr: Attribution,
    kinds: tuple[str, ...] = ("feature", "head"),
    load_target: float = 0.90,
    cost_budget: float | None = None,
    max_fraction: float = 1.0,
    min_t: float = 2.0,
) -> Selection:
    """Greedy selection by load removed per unit of squared weight change.

    ``load_target`` is the fraction of total positive refusal load to remove.
    ``cost_budget`` optionally caps total squared Frobenius change.
    ``max_fraction`` caps the number of atoms as a fraction of all candidates.
    """
    cand = _candidates(attr, kinds, min_t)
    if not cand:
        return Selection([], 0.0, 0.0, 0.0, 0)
    total = sum(a.load for a in cand)
    cap = max(1, int(max_fraction * len(cand)))
    cand.sort(key=lambda a: a.load / max(a.cost, 1e-12), reverse=True)

    chosen: list[Atom] = []
    captured = cost = 0.0
    for atom in cand:
        if captured / total >= load_target:
            break
        if cost_budget is not None and cost + atom.cost > cost_budget:
            continue
        if len(chosen) >= cap:
            break
        chosen.append(atom)
        captured += atom.load
        cost += atom.cost
    return Selection(chosen, total, captured, cost, len(cand))


def select_all(attr: Attribution, kinds: tuple[str, ...] = ("feature", "head"),
               only_positive: bool = True) -> Selection:
    """Dense controls.

    ``only_positive`` keeps atoms whose load pushes toward refusal, which is the
    strongest dense edit.  ``only_positive=False`` reproduces classic
    abliteration, which orthogonalises every row regardless of sign.
    """
    cand = _candidates(attr, kinds, min_t=0.0)
    if not only_positive:
        # Rebuild without the load>0 filter: every atom, both signs.
        cand = []
        for l in range(attr.feature_load.shape[0]):
            if "feature" in kinds:
                for i in range(attr.feature_load.shape[1]):
                    cand.append(Atom(l, "feature", i, float(attr.feature_load[l, i]),
                                     float(attr.feature_cost[l, i]), float(attr.feature_t[l, i])))
            if "head" in kinds:
                for i in range(attr.head_load.shape[1]):
                    cand.append(Atom(l, "head", i, float(attr.head_load[l, i]),
                                     float(attr.head_cost[l, i]), float(attr.head_t[l, i])))
    total = sum(max(a.load, 0.0) for a in cand)
    return Selection(cand, total, total, sum(a.cost for a in cand), len(cand))


class Editor:
    """Applies ablation to a loaded model and can restore it exactly."""

    def __init__(self, lm: LoadedModel) -> None:
        self.lm = lm
        self._snapshot: dict[tuple[str, int], Tensor] = {}
        # Recorded edits, one entry per (layer, kind).  The patch exporter turns
        # this into a replayable artifact; it is also what makes the edit
        # auditable without reloading the model.
        self.ops: dict[int, dict[str, dict]] = {}
        self.log: list[dict] = []

    def _orig(self, kind: str, layer: int, module: torch.nn.Linear) -> Tensor:
        key = (kind, layer)
        if key not in self._snapshot:
            self._snapshot[key] = module.weight.detach().to("cpu", torch.float32).clone()
        return self._snapshot[key]

    def reset(self) -> None:
        for (kind, layer), weight in self._snapshot.items():
            module = self.lm.ffn_down(layer) if kind == "feature" else self.lm.attn_out(layer)
            module.weight.data.copy_(weight.to(module.weight.device, module.weight.dtype))
        self.ops = {}
        self.log = []

    def apply(self, sel, renorm: bool = True, strength: float = 1.0,
              strength_by_layer: dict[int, float] | None = None,
              cost_denominator_layers: list[int] | None = None) -> dict:
        """Ablate the direction from every selected atom.

        ``strength`` scales the removed component (1.0 = full orthogonalisation).
        """
        atoms = sel.atoms if hasattr(sel, "atoms") else sel
        by_layer: dict[int, list[Atom]] = {}
        for atom in atoms:
            by_layer.setdefault(atom.layer, []).append(atom)
        dirs = getattr(self, "_directions", None)
        if dirs is None:
            raise RuntimeError("Editor.set_directions() must be called before apply()")

        changed = 0.0
        # The denominator is the squared norm of every matrix this edit family
        # could touch, so "relative change" is comparable whether 4 layers or 28
        # were edited.
        kinds_used = {a.kind for a in atoms}
        total = 0.0
        for layer in (cost_denominator_layers if cost_denominator_layers is not None
                      else sorted({a.layer for a in atoms})):
            if "feature" in kinds_used:
                module = self.lm.ffn_down(layer)
                total += float(self._orig("feature", layer, module).pow(2).sum())
            if "head" in kinds_used:
                module = self.lm.attn_out(layer)
                total += float(self._orig("head", layer, module).pow(2).sum())
        for layer, atoms in sorted(by_layer.items()):
            s = strength_by_layer.get(layer, strength) if strength_by_layer else strength
            if s == 0.0:
                continue
            r = dirs[layer + 1].float().to(self.lm.device)
            feats = [a for a in atoms if a.kind == "feature"]
            if feats:
                module = self.lm.ffn_down(layer)
                self._orig("feature", layer, module)
                W = module.weight.data.float()
                idx = torch.tensor([a.index for a in feats], device=W.device)
                cols = W[:, idx]
                alpha = cols.T @ r
                new = cols - torch.outer(r, s * alpha)
                if renorm:
                    scale = cols.norm(dim=0) / new.norm(dim=0).clamp_min(1e-12)
                    scale = torch.where(new.norm(dim=0) < RENORM_GUARD * cols.norm(dim=0),
                                        torch.ones_like(scale), scale)
                    new = new * scale
                else:
                    scale = torch.ones(len(feats), device=W.device)
                W[:, idx] = new
                module.weight.data.copy_(W.to(module.weight.dtype))
                changed += float((new - cols).pow(2).sum())
                self.ops.setdefault(layer, {})["feature"] = {
                    "direction": r.detach().float().cpu(),
                    "indices": idx.detach().cpu(),
                    "scale": scale.detach().float().cpu(),
                    "strength": s,
                    "full": len(feats) == self.lm.ffn_down(layer).weight.shape[1],
                }
            heads = [a for a in atoms if a.kind == "head"]
            if heads:
                module = self.lm.attn_out(layer)
                self._orig("head", layer, module)
                W = module.weight.data.float()
                hd = self.lm.head_dim
                head_idx, head_scale = [], []
                for atom in heads:
                    block = W[:, atom.index * hd : (atom.index + 1) * hd]
                    beta = block.T @ r
                    new = block - torch.outer(r, s * beta)
                    scale = 1.0
                    if renorm:
                        n0 = block.norm()
                        n1 = new.norm().clamp_min(1e-12)
                        scale = float(n0 / n1) if n1 > RENORM_GUARD * n0 else 1.0
                        new = new * scale
                    W[:, atom.index * hd : (atom.index + 1) * hd] = new
                    changed += float((new - block).pow(2).sum())
                    head_idx.append(atom.index)
                    head_scale.append(scale)
                module.weight.data.copy_(W.to(module.weight.dtype))
                self.ops.setdefault(layer, {})["head"] = {
                    "direction": r.detach().float().cpu(),
                    "indices": torch.tensor(head_idx),
                    "scale": torch.tensor(head_scale, dtype=torch.float32),
                    "strength": s,
                    "full": len(heads) == self.lm.n_heads,
                }
        torch.cuda.empty_cache()
        return {
            "changed_sq_norm": changed,
            "edited_family_sq_norm": total,
            "relative_change": (changed / total) ** 0.5 if total else 0.0,
        }

    def set_directions(self, directions: Tensor) -> None:
        self._directions = directions
