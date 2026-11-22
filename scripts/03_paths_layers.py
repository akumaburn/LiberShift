"""Path and layer ablation: what the exact decomposition buys for free.

Two questions the per-atom decomposition cannot answer but the per-path and
per-layer one can:

  1. Which sublayer carries the refusal write -- attention, FFN, or both?
     Ablating one path completely is a counterfactual the trace predicts.
  2. Can the edit skip layers?  Ablating the highest-load layers only is
     sparse in weights (untouched layers stay bit-identical) and is the
     structurally honest form of sparsity, unlike row subsets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from libershift import attribute, collect, directions, load_model
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_fit_prompts, good_eval_prompts, good_fit_prompts
from libershift.edit import Atom, Editor
from libershift.evaluate import logprobs, measure, save_result


def atoms_for(attr, kinds: tuple[str, ...], layers: list[int], strength_by_layer=None) -> list[Atom]:
    """Every atom of the requested paths in the requested layers."""
    out: list[Atom] = []
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--tag", default="_main")
    args = ap.parse_args()

    out = OUT / (QWEN_1_5B.name + args.tag)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lm = load_model(QWEN_1_5B.path)
    good_prompts = good_fit_prompts()[: args.fit_prompts]
    bad_prompts = bad_fit_prompts()[: args.fit_prompts]
    good = collect(lm, good_prompts)
    bad = collect(lm, bad_prompts)
    dirs = directions(bad.h, good.h)
    attr = attribute(lm, good, bad, dirs)
    baseline = torch.load(out / "baseline_logprobs.pt", weights_only=True)
    readout = dirs[-1]
    all_layers = list(range(lm.n_layers))

    per_layer = (attr.feature_load.sum(1) + attr.head_load.sum(1)).tolist()
    attn_by_layer = attr.head_load.sum(1).tolist()
    ffn_by_layer = attr.feature_load.sum(1).tolist()
    order = sorted(all_layers, key=lambda l: per_layer[l], reverse=True)
    print("load per layer:", " ".join(f"{x:.1f}" for x in per_layer))
    print("layer order by load:", order)

    editor = Editor(lm)
    editor.set_directions(dirs)
    results = {}
    plan = [
        ("attn_only", ("head",), all_layers),
        ("ffn_only", ("feature",), all_layers),
        ("both", ("feature", "head"), all_layers),
        ("top4_layers", ("feature", "head"), order[:4]),
        ("top8_layers", ("feature", "head"), order[:8]),
        ("top14_layers", ("feature", "head"), order[:14]),
        ("top21_layers", ("feature", "head"), order[:21]),
    ]
    for name, kinds, layers in plan:
        atoms = atoms_for(attr, kinds, layers)
        info = editor.apply(atoms, renorm=True)
        res = measure(lm, baseline, readout, name, keep_texts=True)
        res["edit"] = info
        res["layers"] = sorted(layers)
        res["kinds"] = list(kinds)
        res["predicted_load_removed"] = float(
            sum(attr.feature_load[l].clamp_min(0).sum().item() for l in layers) if "feature" in kinds else 0
        ) + float(sum(attr.head_load[l].clamp_min(0).sum().item() for l in layers) if "head" in kinds else 0)
        results[name] = res
        save_result(out / f"{name}.json", res)
        print(f"[{time.time()-t0:6.1f}s] {name:12s} refusal={res['refusal']['refusal_rate']:.2f} "
              f"kl={res['kl_divergence']:.4f} ppl={res['perplexity_wikitext2']:.1f} "
              f"rel_dw={info['relative_change']:.4f} badproj={res['bad_projection']:.1f} "
              f"goodproj={res['good_projection']:.1f}")
        editor.reset()
        torch.cuda.empty_cache()

    (out / "paths_layers.json").write_text(json.dumps(results, indent=2))
    (out / "load_by_layer.json").write_text(json.dumps({
        "per_layer": per_layer, "attn": attn_by_layer, "ffn": ffn_by_layer, "order": order,
    }, indent=2))
    print(f"[{time.time()-t0:6.1f}s] done")


if __name__ == "__main__":
    sys.exit(main())
