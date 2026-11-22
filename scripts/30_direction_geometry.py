"""Direction geometry and the teacher shift, measured on Qwen2.5-1.5B.

Two claims in ``docs/method.md`` were stated without a frozen file:

* the refusal direction rotates with depth (cosine between planes, and the
  energy held by the leading singular directions);
* the compliant teacher's residual shift is close to ``-r`` where ``-r``
  works and different where it does not.

This script recomputes both from the model and writes them to
``results/analysis/direction_geometry.json``.

The energy shares are the uncentered singular-value decomposition of the 28
unit directions (planes 1-28): the leading direction maximises the summed
squared projection. The centered variant is recorded beside it so the
distinction cannot be lost.

    PYTHONPATH=. .venv/bin/python scripts/30_direction_geometry.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from libershift import attribute, collect, directions, load_model
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_fit_prompts, good_fit_prompts
from libershift.edit import Editor
from libershift.pipeline import atoms_for_layers


def energy_shares(matrix: torch.Tensor) -> tuple[list[float], list[float]]:
    s = torch.linalg.svd(matrix, full_matrices=False).S
    energy = s.pow(2)
    share = (energy / energy.sum()).tolist()
    cumulative = (torch.cumsum(energy, 0) / energy.sum()).tolist()
    return share, cumulative


def main() -> None:
    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[:400])
    bad = collect(lm, bad_fit_prompts()[:400])
    dirs = directions(bad.h, good.h)  # [planes, hidden], unit
    print(f"[{time.time()-t0:6.1f}s] {dirs.shape[0]} directions")

    unit_all = F.normalize(dirs.float(), p=2, dim=1)
    unit = unit_all[1:]  # planes 1-28; plane 0 has no write to attribute
    def cos(a: int, b: int) -> float:
        return float(torch.dot(unit_all[a], unit_all[b]))

    share, cumulative = energy_shares(unit)
    share_c, cumulative_c = energy_shares(unit - unit.mean(dim=0, keepdim=True))

    # The teacher is the exported dense-removal checkpoint. Replay is exact,
    # so applying the same edit in memory reproduces its residuals.
    plane = 18
    dirs_plane = dirs[plane].unsqueeze(0).repeat(dirs.shape[0], 1)
    attr = attribute(lm, good, bad, dirs_plane)
    all_atoms = atoms_for_layers(attr, ("feature", "head"), list(range(lm.n_layers)))
    editor = Editor(lm)
    editor.set_directions(dirs_plane)
    info = editor.apply(all_atoms, renorm=True)
    print(f"[{time.time()-t0:6.1f}s] dense removal applied ({len(all_atoms)} atoms)")

    teacher_bad = collect(lm, bad_fit_prompts()[:400])
    shift = teacher_bad.h - bad.h
    a = F.normalize(shift.float(), p=2, dim=1)
    teacher_cos = F.cosine_similarity(a, -dirs_plane, dim=1)
    editor.reset()

    payload = {
        "model": "Qwen2.5-1.5B-Instruct",
        "prompts_per_class": 400,
        "direction_cosines": {
            "r2_r28": cos(2, 28),
            "r27_r28": cos(27, 28),
            "r18_r28": cos(18, 28),
            "adjacent_min": min(cos(i, i + 1) for i in range(1, 28)),
            "adjacent_max": max(cos(i, i + 1) for i in range(1, 28)),
        },
        "energy_uncentered": {
            "directions": 28,
            "share": share,
            "cumulative": cumulative,
            "first_component": cumulative[0],
            "ten_components": cumulative[9],
        },
        "energy_centered": {
            "directions": 28,
            "share": share_c,
            "cumulative": cumulative_c,
            "first_component": cumulative_c[0],
            "ten_components": cumulative_c[9],
        },
        "teacher": {
            "plane": plane,
            "atoms": len(all_atoms),
            "relative_change": info["relative_change"],
            "cos_shift_neg_r_by_plane": {
                str(i): float(x) for i, x in enumerate(teacher_cos) if i >= 1
            },
            "cos_min": float(teacher_cos[1:].min()),
            "cos_max": float(teacher_cos[1:].max()),
            "cos_at_plane_18": float(teacher_cos[18]),
            "cos_at_plane_27": float(teacher_cos[27]),
        },
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = OUT / "direction_geometry.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "direction_cosines": payload["direction_cosines"],
        "uncentered_first": cumulative[0],
        "uncentered_ten": cumulative[9],
        "centered_first": cumulative_c[0],
        "centered_ten": cumulative_c[9],
    }, indent=2))
    print("teacher cos at plane 18", payload["teacher"]["cos_at_plane_18"],
          "at plane 27", payload["teacher"]["cos_at_plane_27"])
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
