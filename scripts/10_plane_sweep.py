"""Which plane's direction is best, measured on the behaviour?

The propagation score from the trace ranks candidate planes.  This sweep
measures the refusal rate for each candidate, so the score can be checked
against the behaviour it claims to predict.  Every candidate is applied to the
same rows: all layers, all atoms, row-norm preserved.  The only variable is the
direction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from libershift import attribute, collect, directions, load_model
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_fit_prompts, good_fit_prompts
from libershift.edit import Editor
from libershift.evaluate import logprobs, measure, save_result
from libershift.pipeline import atoms_for_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--planes", default="14,16,18,20,22,24,26,28")
    ap.add_argument("--tag", default="_main")
    args = ap.parse_args()

    out = OUT / (QWEN_1_5B.name + args.tag)
    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    attr = attribute(lm, good, bad, dirs)
    load = (attr.feature_load.sum(1) + attr.head_load.sum(1))
    baseline = torch.load(out / "baseline_logprobs.pt", weights_only=True)
    readout = dirs[-1]
    all_atoms = atoms_for_layers(attr, ("feature", "head"), list(range(lm.n_layers)))

    # Propagation score, for the record next to the measured rates.
    planes = dirs[1:]
    scores = {}
    for p in range(planes.shape[0]):
        cos = (planes @ planes[p]).clamp_min(0.0)
        later = torch.zeros_like(load)
        later[p:] = load[p:]
        scores[p + 1] = float((later * cos).sum())

    editor = Editor(lm)
    results = {"propagation": scores, "planes": {}}
    for plane in [int(x) for x in args.planes.split(",")]:
        editor.set_directions(dirs[plane].unsqueeze(0).repeat(dirs.shape[0], 1))
        info = editor.apply(all_atoms, renorm=True)
        res = measure(lm, baseline, readout, f"plane_{plane}", keep_texts=True)
        res["edit"] = info
        res["plane"] = plane
        res["propagation_score"] = scores.get(plane)
        results["planes"][plane] = res
        save_result(out / f"plane_{plane}.json", res)
        print(f"[{time.time()-t0:6.1f}s] plane {plane:2d} (S={scores.get(plane, 0):5.0f}) "
              f"refusal={res['refusal']['refusal_rate']:.2f} kl={res['kl_divergence']:.4f} "
              f"ppl={res['perplexity_wikitext2']:.2f} rel_dw={info['relative_change']:.4f} "
              f"badproj={res['bad_projection']:.2f}")
        editor.reset()
        torch.cuda.empty_cache()

    # Per-layer directions, all layers, as the multi-direction reference.
    editor.set_directions(dirs)
    info = editor.apply(all_atoms, renorm=True)
    res = measure(lm, baseline, readout, "per_layer_all", keep_texts=True)
    res["edit"] = info
    results["per_layer_all"] = res
    save_result(out / "per_layer_all.json", res)
    print(f"[{time.time()-t0:6.1f}s] per-layer     refusal={res['refusal']['refusal_rate']:.2f} "
          f"kl={res['kl_divergence']:.4f} ppl={res['perplexity_wikitext2']:.2f} "
          f"rel_dw={info['relative_change']:.4f} badproj={res['bad_projection']:.2f}")

    (out / "plane_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"[{time.time()-t0:6.1f}s] wrote {out / 'plane_sweep.json'}")


if __name__ == "__main__":
    sys.exit(main())
