"""Clean timing of each stage of the method, with a synchronised clock.

Run with the GPU otherwise idle.  Reports per-stage seconds so the cost can be
compared with a search-based tool stage by stage, not just end to end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from libershift import attribute, collect, directions, load_model
from libershift.constants import OUT, spec as get_spec
from libershift.data import bad_eval_prompts, bad_extended_prompts, bad_fit_prompts, good_fit_prompts
from libershift.edit import Editor
from libershift.evaluate import refusal_rate
from libershift.patch import write_patch
from libershift.pipeline import atoms_for_layers


class Clock:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.stages: dict[str, float] = {}
        self.last = self.t0

    def mark(self, name: str) -> float:
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.stages[name] = now - self.last
        self.last = now
        print(f"{name:32s} {self.stages[name]:7.2f} s")
        return self.stages[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--planes", default="14,16,18,20,22,24")
    ap.add_argument("--calibration", type=int, default=30)
    ap.add_argument("--export", action="store_true", default=True)
    args = ap.parse_args()

    ms = get_spec(args.model)
    tag = "_main" if "1.5B" in ms.name else ""
    out = OUT / (ms.name + tag)
    c = Clock()

    lm = load_model(ms.path)
    c.mark("load model")

    good_prompts = good_fit_prompts()[:400]
    bad_prompts = bad_fit_prompts()[:400]
    good = collect(lm, good_prompts)
    bad = collect(lm, bad_prompts)
    c.mark("capture 800 prompts")

    dirs = directions(bad.h, good.h)
    attr = attribute(lm, good, bad, dirs)
    c.mark("attribute 28 layers")

    editor = Editor(lm)
    atoms = atoms_for_layers(attr, ("feature", "head"), list(range(lm.n_layers)))
    evals = bad_eval_prompts()
    cal = evals[: args.calibration]
    chosen, scores = None, {}
    for plane in [int(x) for x in args.planes.split(",")]:
        editor.set_directions(dirs[plane].unsqueeze(0).repeat(dirs.shape[0], 1))
        editor.apply(atoms, renorm=True)
        scores[plane] = refusal_rate(lm, cal)["refusal_rate"]
        editor.reset()
    chosen = min(scores, key=lambda p: scores[p])
    c.mark(f"calibrate {len(scores)} planes x {len(cal)} prompts")

    editor.set_directions(dirs[chosen].unsqueeze(0).repeat(dirs.shape[0], 1))
    info = editor.apply(atoms, renorm=True)
    c.mark("apply edit")

    if args.export:
        target = OUT / "timing_export"
        target.mkdir(parents=True, exist_ok=True)
        lm.model.save_pretrained(target, safe_serialization=True)
        lm.tokenizer.save_pretrained(target)
        write_patch(OUT / "timing_export.json", editor.ops,
                    {"model": ms.name, "method": "timing",
                     "selection": {"plane": chosen, "atoms": len(atoms), "edit": info}})
        c.mark("export checkpoint + patch")

    # A user needs one evaluation to know whether it worked.
    refusal_rate(lm, bad_eval_prompts()[args.calibration :])
    refusal_rate(lm, bad_extended_prompts())
    c.mark("final evaluation (370 prompts)")

    total = time.perf_counter() - c.t0
    print(f"\n{'total':32s} {total:7.2f} s")
    print(f"chosen plane {chosen}, scores {scores}")
    (OUT / "timing.json").write_text(json.dumps(
        {"model": ms.name, "stages": c.stages, "total": total,
         "chosen_plane": chosen, "calibration_scores": scores,
         "atoms": len(atoms), "edit": info}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
