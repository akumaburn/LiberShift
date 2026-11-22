"""Whole procedure on a second model, to check the method is not tuned to one.

Runs the same steps as the 1.5B study: capture, attribute, calibrate the plane
on 30 prompts, export the chosen edit, then measure on the report split and on
300 held-out prompts.  Writes one JSON per model plus the exported checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from libershift import attribute, collect, directions, load_model
from libershift.constants import OUT, spec as get_spec
from libershift.data import bad_eval_prompts, bad_extended_prompts, bad_fit_prompts, good_eval_prompts, good_fit_prompts
from libershift.edit import Editor
from libershift.evaluate import logprobs, measure, refusal_rate, save_result
from libershift.patch import write_patch
from libershift.pipeline import atoms_for_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-0.5B-Instruct")
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--calibration", type=int, default=30)
    ap.add_argument("--planes", default="8,10,12,14,16,18")
    ap.add_argument("--export", action="store_true", default=True)
    args = ap.parse_args()

    ms = get_spec(args.model)
    out = OUT / ms.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lm = load_model(ms.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    attr = attribute(lm, good, bad, dirs)
    readout = dirs[-1]
    print(f"[{time.time()-t0:6.1f}s] captured and attributed ({lm.n_layers} layers)")

    baseline = logprobs(lm, good_eval_prompts())
    torch.save(baseline, out / "baseline_logprobs.pt")

    evals = bad_eval_prompts()
    cal, report = evals[: args.calibration], evals[args.calibration:]
    editor = Editor(lm)
    all_atoms = atoms_for_layers(attr, ("feature", "head"), list(range(lm.n_layers)))

    base = measure(lm, baseline, readout, "baseline", keep_texts=True)
    base_ext = refusal_rate(lm, bad_extended_prompts())
    save_result(out / "baseline.json", base)
    print(f"[{time.time()-t0:6.1f}s] baseline report refusal={base['refusal']['refusal_rate']:.2f} "
          f"extended={base_ext['refusal_rate']:.2f} kl={base['kl_divergence']:.4f} "
          f"ppl={base['perplexity_wikitext2']:.2f}")

    results = {"model": ms.name, "baseline": {k: v for k, v in base.items() if k != "responses"},
               "baseline_extended": {k: v for k, v in base_ext.items() if k != "responses"},
               "calibration": {}, "planes": {}, "calibration_n": len(cal),
               "report_n": len(report)}

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

    editor.set_directions(dirs[chosen].unsqueeze(0).repeat(dirs.shape[0], 1))
    info = editor.apply(all_atoms, renorm=True)
    res = measure(lm, baseline, readout, "taa", keep_texts=True)
    res["edit"] = info
    res["plane"] = chosen
    ext = refusal_rate(lm, bad_extended_prompts())
    res["extended_refusal"] = {k: v for k, v in ext.items() if k != "responses"}
    save_result(out / "taa.json", res)
    print(f"[{time.time()-t0:6.1f}s] method: report refusal={res['refusal']['refusal_rate']:.2f} "
          f"extended={ext['refusal_rate']:.2f} kl={res['kl_divergence']:.4f} "
          f"ppl={res['perplexity_wikitext2']:.2f} rel_dw={info['relative_change']:.4f}")

    if args.export:
        target = OUT / "export" / f"{ms.name}_taa"
        target.mkdir(parents=True, exist_ok=True)
        lm.model.save_pretrained(target, safe_serialization=True)
        lm.tokenizer.save_pretrained(target)
        write_patch(OUT / "export" / f"{ms.name}_taa.json", editor.ops,
                    {"model": ms.name, "method": "taa", "selection": {"plane": chosen,
                     "atoms": len(all_atoms), "edit": info}})
        print(f"saved {target}")

    results["taa"] = {k: v for k, v in res.items() if k != "responses"}
    (out / "method.json").write_text(json.dumps(results, indent=2))
    print(f"[{time.time()-t0:6.1f}s] wrote {out / 'method.json'}")


if __name__ == "__main__":
    sys.exit(main())
