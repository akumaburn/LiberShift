"""Time the edit step: dense removal, sparse removal, redirect and hybrid.

Both edits produce a normal dense checkpoint, so inference speed is identical.
The only possible speed difference is in producing the edit.  This measures the
weight-write step (and the selection) on otherwise identical setup.

    PYTHONPATH=. .venv/bin/python scripts/40_edit_timing.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from libershift import collect, directions, load_model
from libershift.attribute import attribute
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_fit_prompts, good_fit_prompts
from libershift.edit import Editor, select, select_all
from libershift.redirect import HybridEdit, WeightSnapshot, apply_redirect


def timed(fn, reps: int):
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=str(OUT / "edit_timing.json"))
    args = ap.parse_args()

    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[:400])
    bad = collect(lm, bad_fit_prompts()[:400])
    dirs = directions(bad.h, good.h)
    dirs = dirs[18].unsqueeze(0).repeat(dirs.shape[0], 1)
    attr = attribute(lm, good, bad, dirs)

    t0 = time.perf_counter()
    sparse = select(attr, ("feature", "head"), load_target=0.90, min_t=2.0)
    select_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    dense = select_all(attr, only_positive=False)
    dense_select_time = time.perf_counter() - t0

    editor = Editor(lm)
    editor.set_directions(dirs)
    snapshot = WeightSnapshot(lm)
    hybrid_editor = HybridEdit(lm)
    ranked = sorted(sparse.atoms, key=lambda atom: atom.load, reverse=True)
    k = int(round(0.25 * len(ranked)))
    removal, redirect = ranked[:k], ranked[k:]

    methods = {
        # The actual removal exporter path (Editor snapshots to CPU for reset).
        "editor_dense_removal": lambda: editor.apply(dense, renorm=True, strength=1.0),
        "editor_sparse_removal": lambda: editor.apply(sparse, renorm=True, strength=1.0),
        # The same GPU write path used by the redirect/hybrid, so the atom
        # count is the only difference.  gamma=0 is removal.
        "write_dense_removal": lambda: apply_redirect(
            lm, dense, dirs, dirs, gamma=0.0, snapshot=snapshot
        ),
        "write_sparse_removal": lambda: apply_redirect(
            lm, sparse, dirs, dirs, gamma=0.0, snapshot=snapshot
        ),
        "write_sparse_redirect": lambda: apply_redirect(
            lm, sparse, dirs, dirs, gamma=0.3, snapshot=snapshot
        ),
        "write_hybrid": lambda: hybrid_editor.apply(removal, redirect, dirs, dirs, 0.3),
    }
    results = {
        "model": QWEN_1_5B.name,
        "atoms": {
            "dense_removal": len(dense.atoms),
            "sparse": len(sparse.atoms),
            "hybrid_removal": len(removal),
            "hybrid_redirect": len(redirect),
        },
        "selection_seconds": {
            "sparse_90pct": select_time,
            "dense_all": dense_select_time,
        },
        "methods": {},
    }
    for name, fn in methods.items():
        times = timed(fn, args.reps)
        results["methods"][name] = {
            "seconds": times,
            "median": statistics.median(times),
        }
        if name.startswith("editor_"):
            editor.reset()
        elif name == "write_hybrid":
            hybrid_editor.restore()
        else:
            snapshot.restore()
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    sys.exit(main())
