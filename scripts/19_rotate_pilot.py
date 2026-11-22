"""Pilot: rotate refusal writers into acceptance writers instead of ablating them.

Removal applies ``d_f <- d_f - alpha r`` and leaves the feature writing nothing
along the refusal direction.  Rotation applies the same update with strength
two, so the refusal component flips sign: ``d_f <- d_f - 2 alpha r``.  The
feature keeps its norm and its gate, and writes the opposite direction.

The pilot captures fresh statistics with the current package, then compares
removal and rotation at matched atom selections.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from libershift import collect, directions, load_model
from libershift.attribute import attribute
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_fit_prompts, good_fit_prompts
from libershift.edit import Editor, select
from libershift.evaluate import measure, save_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--direction", default="local",
                    help="'local' for per-layer directions, or 'plane:N' for one plane at every layer")
    ap.add_argument("--kinds", default="feature,head")
    ap.add_argument("--targets", default="0.5,0.9")
    ap.add_argument("--strengths", default="1.0,2.0")
    ap.add_argument("--out", default=str(OUT / "rotation_pilot.json"))
    args = ap.parse_args()

    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    if args.direction.startswith("plane:"):
        plane = int(args.direction.split(":", 1)[1])
        dirs = dirs[plane].unsqueeze(0).repeat(dirs.shape[0], 1)
        scope = f"plane{plane}"
    elif args.direction == "local":
        scope = "local"
    else:
        raise SystemExit(f"unknown direction scope {args.direction!r}")
    attr = attribute(lm, good, bad, dirs)
    baseline = torch.load(args.baseline, weights_only=True)
    readout = dirs[-1]
    print(f"[{time.time()-t0:6.1f}s] capture and attribution ready "
          f"({good.count} good, {bad.count} bad, scope {scope})")

    results = []
    for target in [float(x) for x in args.targets.split(",")]:
        sel = select(attr, tuple(args.kinds.split(",")), load_target=target, min_t=2.0)
        print(f"target {target:.2f}: {len(sel.atoms)} atoms, "
              f"captured {sel.captured_fraction:.3f}, cost {sel.total_cost:.1f}")
        for strength in [float(x) for x in args.strengths.split(",")]:
            editor = Editor(lm)
            editor.set_directions(dirs)
            info = editor.apply(sel, renorm=True, strength=strength)
            label = f"rot_{scope}_{args.kinds.replace(chr(44), chr(43))}_t{target:.2f}_s{strength:g}"
            res = measure(lm, baseline, readout, label, keep_texts=True)
            res["edit"] = info
            res["selection"] = sel.summary()
            res["strength"] = strength
            res["target"] = target
            path = Path(args.out).with_name(label + ".json")
            save_result(path, res)
            results.append({
                "label": label,
                "target": target,
                "strength": strength,
                "atoms": len(sel.atoms),
                "captured_fraction": sel.captured_fraction,
                "relative_change": info["relative_change"],
                "refusal": res["refusal"],
                "kl_divergence": res["kl_divergence"],
                "perplexity_wikitext2": res["perplexity_wikitext2"],
                "bad_projection": res["bad_projection"],
                "good_projection": res["good_projection"],
            })
            print(f"[{time.time()-t0:6.1f}s] {label:18s} "
                  f"refusal={res['refusal']['refusal_rate']:.3f} "
                  f"opening={res['refusal']['opening_refusal_rate']:.3f} "
                  f"kl={res['kl_divergence']:.4f} "
                  f"ppl={res['perplexity_wikitext2']:.2f} "
                  f"rel_dw={info['relative_change']:.4f}")
            editor.reset()
            torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
