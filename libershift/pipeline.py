"""The method as a procedure: attribute, select, edit, and correct.

A run is a sequence of rounds.  Each round captures the current model's
residual statistics on the harmful and harmless prompt sets, attributes the
refusal direction onto every write atom, selects atoms by load removed per unit
of weight change, and ablates the selected atoms.  Round two therefore works on
whatever refusal load *survives* round one, which is the closed-loop part: the
edit is measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .attribute import Attribution, attribute
from .capture import Stats, collect
from .direction import directions
from .edit import Editor, Selection, select
from .model import LoadedModel


@dataclass
class Config:
    # "local": ablate each layer against the direction of its own output plane.
    # "global": ablate every layer against the final plane's direction.
    direction_scope: str = "local"
    load_target: float = 0.9
    rounds: int = 1
    renorm: bool = True
    min_t: float = 2.0
    kinds: tuple[str, ...] = ("feature", "head")
    max_fraction: float = 1.0
    # For the dense control: keep atoms with negative load (classic
    # abliteration orthogonalises every row, whatever its sign).
    include_negative: bool = False

    def key(self) -> str:
        return (f"{self.direction_scope}_t{self.load_target:.2f}_r{self.rounds}"
                f"_renorm{int(self.renorm)}_tstat{self.min_t:g}")


def build_directions(good: Stats, bad: Stats, scope: str) -> Tensor:
    local = directions(bad.h, good.h)
    if scope == "local":
        return local
    if scope == "global":
        return local[-1].unsqueeze(0).repeat(local.shape[0], 1)
    raise ValueError(scope)


@dataclass
class RoundReport:
    round: int
    selection: dict
    edit: dict
    load_remaining: float
    load_before: float


@dataclass
class RunReport:
    config: Config
    rounds: list[RoundReport] = field(default_factory=list)
    attribution_identity: dict = field(default_factory=dict)
    atoms: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "config": {
                "direction_scope": self.config.direction_scope,
                "load_target": self.config.load_target,
                "rounds": self.config.rounds,
                "renorm": self.config.renorm,
                "min_t": self.config.min_t,
                "kinds": list(self.config.kinds),
                "max_fraction": self.config.max_fraction,
            },
            "rounds": [
                {
                    "round": r.round,
                    "selection": r.selection,
                    "edit": r.edit,
                    "load_before": r.load_before,
                    "load_remaining": r.load_remaining,
                }
                for r in self.rounds
            ],
            "attribution_identity": self.attribution_identity,
            "atoms_total": len(self.atoms),
        }


def positive_load(attr: Attribution, kinds: tuple[str, ...]) -> float:
    load = 0.0
    if "feature" in kinds:
        load += float(attr.feature_load.clamp_min(0).sum())
    if "head" in kinds:
        load += float(attr.head_load.clamp_min(0).sum())
    return load


def run(
    lm: LoadedModel,
    good_prompts: list[str],
    bad_prompts: list[str],
    cfg: Config,
    direction: Tensor | None = None,
    stats: tuple[Stats, Stats] | None = None,
) -> tuple[Editor, RunReport]:
    """Apply ``cfg`` to ``lm`` in place and return the editor holding the edit."""
    report = RunReport(config=cfg)
    good, bad = stats if stats is not None else (
        collect(lm, good_prompts), collect(lm, bad_prompts)
    )
    dirs = direction if direction is not None else build_directions(good, bad, cfg.direction_scope)

    editor = Editor(lm)
    editor.set_directions(dirs)

    for rnd in range(1, cfg.rounds + 1):
        if rnd > 1:
            good = collect(lm, good_prompts)
            bad = collect(lm, bad_prompts)
        attr = attribute(lm, good, bad, dirs)
        if rnd == 1:
            from .attribute import check_identity

            report.attribution_identity = {
                k: v for k, v in check_identity(attr, dirs, good, bad).items() if "max" in k
            }
        before = positive_load(attr, cfg.kinds)
        if cfg.load_target >= 1.0:
            from .edit import select_all

            sel = select_all(attr, cfg.kinds, only_positive=not cfg.include_negative)
        else:
            sel = select(
                attr,
                kinds=cfg.kinds,
                load_target=cfg.load_target,
                min_t=cfg.min_t,
                max_fraction=cfg.max_fraction,
            )
        info = editor.apply(sel, renorm=cfg.renorm)
        report.atoms.extend(sel.atoms)
        report.rounds.append(
            RoundReport(
                round=rnd,
                selection=sel.summary(),
                edit=info,
                load_before=before,
                load_remaining=before - sel.captured_load,
            )
        )
    return editor, report


def fused_selection(report: RunReport) -> Selection:
    """All atoms the run touched, for patch export."""
    total = sum(r.selection["total_positive_load"] for r in report.rounds)
    captured = sum(r.selection["captured_load"] for r in report.rounds)
    cost = sum(r.selection["total_cost"] for r in report.rounds)
    return Selection(report.atoms, total, captured, cost, len(report.atoms))


def atoms_for_layers(attr: Attribution, kinds: tuple[str, ...], layers: list[int]):
    """Every atom of the selected paths in the selected layers."""
    from .edit import Atom

    out = []
    for l in layers:
        if "feature" in kinds:
            for i in range(attr.feature_load.shape[1]):
                out.append(Atom(l, "feature", i, float(attr.feature_load[l, i]),
                                float(attr.feature_cost[l, i]), float(attr.feature_t[l, i])))
        if "head" in kinds:
            for i in range(attr.head_load.shape[1]):
                out.append(Atom(l, "head", i, float(attr.head_load[l, i]),
                                float(attr.head_cost[l, i]), float(attr.head_t[l, i])))
    return out
