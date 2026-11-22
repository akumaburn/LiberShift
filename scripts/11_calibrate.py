"""Calibration of the one free parameter, and two failing alternatives.

The method has one free parameter: which plane's direction becomes the edit
direction.  Everything else follows from the trace.

Calibration uses the first 30 prompts of the evaluation slice; every reported
number uses the remaining 70 plus 300 held-out prompts, so the calibration and
the report do not share a prompt.  The calibration metric is the same keyword
refusal rate the comparison uses, because that is what the method is asked to
remove.

The two schedules below test a tempting idea: scale each layer's ablation
strength by how much refusal write that layer carries.  Both fail, and the
failures are reported rather than dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from libershift import attribute, collect, directions, load_model
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_eval_prompts, bad_fit_prompts, good_fit_prompts
from libershift.edit import Editor
from libershift.evaluate import logprobs, measure, refusal_rate, save_result
from libershift.pipeline import atoms_for_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--calibration", type=int, default=30)
    ap.add_argument("--planes", default="14,16,18,20,22,24")
    ap.add_argument("--tag", default="_main")
    args = ap.parse_args()

    out = OUT / (QWEN_1_5B.name + args.tag)
    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    attr = attribute(lm, good, bad, dirs)
    baseline = torch.load(out / "baseline_logprobs.pt", weights_only=True)
    readout = dirs[-1]
    all_atoms = atoms_for_layers(attr, ("feature", "head"), list(range(lm.n_layers)))

    evals = bad_eval_prompts()
    cal = evals[: args.calibration]
    report = evals[args.calibration:]
    print(f"calibration prompts {len(cal)}, report prompts {len(report)}")

    editor = Editor(lm)
    results = {"calibration": {}, "schedules": {}, "chosen_plane": None,
               "calibration_n": len(cal), "report_n": len(report)}

    for plane in [int(x) for x in args.planes.split(",")]:
        editor.set_directions(dirs[plane].unsqueeze(0).repeat(dirs.shape[0], 1))
        editor.apply(all_atoms, renorm=True)
        rate = refusal_rate(lm, cal)
        results["calibration"][plane] = rate
        print(f"[{time.time()-t0:6.1f}s] plane {plane:2d}: calibration refusal="
              f"{rate['refusal_rate']:.2f} ({rate['refusals']}/{rate['n']})")
        editor.reset()
        torch.cuda.empty_cache()

    chosen = min(results["calibration"], key=lambda p: results["calibration"][p]["refusal_rate"])
    results["chosen_plane"] = chosen
    print(f"chosen plane: {chosen}")

    # Report-split numbers for every candidate, so the calibration can be checked.
    for plane in [int(x) for x in args.planes.split(",")]:
        editor.set_directions(dirs[plane].unsqueeze(0).repeat(dirs.shape[0], 1))
        info = editor.apply(all_atoms, renorm=True)
        res = measure(lm, baseline, readout, f"plane_{plane}", keep_texts=True)
        res["edit"] = info
        res["plane"] = plane
        results.setdefault("report", {})[plane] = {
            "refusal": {k: v for k, v in res["refusal"].items() if k != "responses"},
            "kl_divergence": res["kl_divergence"],
            "perplexity_wikitext2": res["perplexity_wikitext2"],
            "edit": info,
        }
        save_result(out / f"plane_{plane}_report.json", res)
        print(f"[{time.time()-t0:6.1f}s] plane {plane:2d} report: refusal="
              f"{res['refusal']['refusal_rate']:.3f} opening="
              f"{res['refusal']['opening_refusal_rate']:.3f} kl={res['kl_divergence']:.4f} "
              f"ppl={res['perplexity_wikitext2']:.2f}")
        editor.reset()
        torch.cuda.empty_cache()

    # Trace-derived strength schedules on the chosen plane.
    proj = (((bad.attn + bad.ffn) - (good.attn + good.ffn)) @ dirs[chosen]).clamp_min(0.0)
    norm = proj / proj.max() if float(proj.max()) > 0 else proj
    schedules = {
        "flat": {l: 1.0 for l in range(lm.n_layers)},
        "proj": {l: float(norm[l]) for l in range(lm.n_layers)},
        "proj_floor": {l: max(0.25, float(norm[l])) for l in range(lm.n_layers)},
    }
    editor.set_directions(dirs[chosen].unsqueeze(0).repeat(dirs.shape[0], 1))
    for name, strengths in schedules.items():
        atoms = [a for a in all_atoms if strengths[a.layer] > 0]
        info = editor.apply(atoms, renorm=True, strength_by_layer=strengths)
        res = measure(lm, baseline, readout, f"schedule_{name}", keep_texts=True)
        res["edit"] = info
        res["strengths"] = {str(k): v for k, v in strengths.items()}
        results["schedules"][name] = {
            "refusal": {k: v for k, v in res["refusal"].items() if k != "responses"},
            "kl_divergence": res["kl_divergence"],
            "perplexity_wikitext2": res["perplexity_wikitext2"],
            "edit": info,
        }
        save_result(out / f"chosen_schedule_{name}.json", res)
        print(f"[{time.time()-t0:6.1f}s] schedule {name:10s} refusal="
              f"{res['refusal']['refusal_rate']:.3f} kl={res['kl_divergence']:.4f} "
              f"ppl={res['perplexity_wikitext2']:.2f} rel_dw={info['relative_change']:.4f}")
        editor.reset()
        torch.cuda.empty_cache()

    (out / "calibration.json").write_text(json.dumps(results, indent=2))
    print(f"[{time.time()-t0:6.1f}s] wrote {out / 'calibration.json'}")


if __name__ == "__main__":
    sys.exit(main())
