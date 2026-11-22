"""Ablation schedules derived from the measured per-layer load.

Every schedule here edits the whole FFN down matrix and the whole attention
output matrix of the layers it touches, at a per-layer strength in [0, 1].  The
question is how to allocate that strength.  The trace gives the per-layer load,
so the allocation can be computed instead of searched:

  uniform      every layer at full strength (classic abliteration)
  top_k        full strength on the k highest-load layers, nothing elsewhere
  proportional strength = load / max(load)
  floor_half   full strength on the higher-load half, proportional below
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
from libershift.evaluate import measure, save_result
from libershift.pipeline import atoms_for_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--tag", default="_main")
    arans = ap.parse_args()

    out = OUT / (QWEN_1_5B.name + arans.tag)
    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[: arans.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: arans.fit_prompts])
    dirs = directions(bad.h, good.h)
    attr = attribute(lm, good, bad, dirs)
    baseline = torch.load(out / "baseline_logprobs.pt", weights_only=True)
    readout = dirs[-1]

    load = (attr.feature_load.sum(1) + attr.head_load.sum(1))
    load_norm = (load / load.max()).tolist()
    order = sorted(range(lm.n_layers), key=lambda l: float(load[l]), reverse=True)

    schedules: dict[str, dict[int, float]] = {
        "uniform": {l: 1.0 for l in range(lm.n_layers)},
        "top8": {l: (1.0 if l in order[:8] else 0.0) for l in range(lm.n_layers)},
        "top14": {l: (1.0 if l in order[:14] else 0.0) for l in range(lm.n_layers)},
        "top14_uniform": {l: 1.0 for l in order[:14]},
        "proportional": {l: float(load_norm[l]) for l in range(lm.n_layers)},
        "floor_half": {
            l: (1.0 if l in order[: lm.n_layers // 2] else float(load_norm[l]))
            for l in range(lm.n_layers)
        },
    }

    editor = Editor(lm)
    editor.set_directions(dirs)
    all_atoms = atoms_for_layers(attr, ("feature", "head"), list(range(lm.n_layers)))
    results = {}
    for name, strengths in schedules.items():
        touched = [l for l, s in strengths.items() if s > 0]
        atoms = [a for a in all_atoms if strengths.get(a.layer, 0.0) > 0]
        info = editor.apply(atoms, renorm=True, strength_by_layer=strengths)
        res = measure(lm, baseline, readout, name, keep_texts=True)
        res["edit"] = info
        res["strengths"] = {str(k): v for k, v in sorted(strengths.items())}
        res["layers_touched"] = sorted(touched)
        results[name] = res
        save_result(out / f"schedule_{name}.json", res)
        print(f"[{time.time()-t0:6.1f}s] {name:14s} refusal={res['refusal']['refusal_rate']:.2f} "
              f"kl={res['kl_divergence']:.4f} ppl={res['perplexity_wikitext2']:.1f} "
              f"rel_dw={info['relative_change']:.4f} layers={len(touched)} "
              f"badproj={res['bad_projection']:.1f}")
        editor.reset()
        torch.cuda.empty_cache()

    (out / "schedules.json").write_text(json.dumps(results, indent=2))
    print(f"[{time.time()-t0:6.1f}s] done")


if __name__ == "__main__":
    sys.exit(main())
