"""Does over-ablation help?  Strength sweep on the chosen plane.

Strength 1.0 removes the direction exactly.  A strength above 1.0 removes more
than the projection and flips the sign of the component.  Heretic's optimiser
chose 1.45 for attention on this model, so this tests the same idea inside the
single-plane edit.
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
from libershift.pipeline import atoms_for_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--plane", type=int, default=18)
    ap.add_argument("--strengths", default="1.0,1.15,1.3,1.45")
    ap.add_argument("--fit-prompts", type=int, default=400)
    args = ap.parse_args()

    ms = get_spec(args.model)
    out = OUT / (ms.name + ("_main" if "1.5B" in ms.name else ""))
    t0 = time.time()
    lm = load_model(ms.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    attr = attribute(lm, good, bad, dirs)
    baseline = torch.load(out / "baseline_logprobs.pt", weights_only=True)
    readout = dirs[-1]
    atoms = atoms_for_layers(attr, ("feature", "head"), list(range(lm.n_layers)))
    evals = bad_eval_prompts()
    report = evals[30:]

    editor = Editor(lm)
    editor.set_directions(dirs[args.plane].unsqueeze(0).repeat(dirs.shape[0], 1))
    results = {"plane": args.plane, "model": ms.name, "strengths": {}}
    for strength in [float(x) for x in args.strengths.split(",")]:
        info = editor.apply(atoms, renorm=True, strength=strength)
        res = measure(lm, baseline, readout, f"strength_{strength:.2f}", keep_texts=True)
        ext = refusal_rate(lm, bad_extended_prompts())
        res["extended_refusal"] = {k: v for k, v in ext.items() if k != "responses"}
        res["edit"] = info
        res["strength"] = strength
        results["strengths"][strength] = {
            "report": res["refusal"], "extended": res["extended_refusal"],
            "kl_divergence": res["kl_divergence"],
            "perplexity_wikitext2": res["perplexity_wikitext2"], "edit": info,
        }
        save_result(out / f"strength_{strength:.2f}.json", res)
        print(f"[{time.time()-t0:6.1f}s] strength {strength:.2f}: report="
              f"{res['refusal']['refusal_rate']:.3f} extended="
              f"{ext['refusal_rate']:.3f} kl={res['kl_divergence']:.4f} "
              f"ppl={res['perplexity_wikitext2']:.2f} rel_dw={info['relative_change']:.4f}")
        editor.reset()
        torch.cuda.empty_cache()

    (out / "strength_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out / 'strength_sweep.json'}")


if __name__ == "__main__":
    sys.exit(main())
