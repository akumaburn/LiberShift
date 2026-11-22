"""Probe: does a rotation keep the refusal detector firing while it changes the
feature's down edge from a refusal target to another target?

The rotation is the same edit as ``19_rotate_pilot.py``: selected refusal
writers get ``d_f <- d_f - 2 alpha r``.  This script records, before and after:

* the mean activation of the selected features on harmful prompts (the gate and
  up weights are untouched, so a ratio near one means the detector still fires);
* the top tokens of each selected feature's down vector (the graph edge target);
* refusal on the report split and on 300 held-out prompts, KL and perplexity.

It also runs the dense removal control for comparison.
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
from libershift.data import bad_extended_prompts, bad_fit_prompts, good_fit_prompts
from libershift.edit import Editor, select, select_all
from libershift.evaluate import measure, refusal_rate, save_result


def top_tokens(lm, layer: int, feature: int, k: int = 5) -> list[str]:
    W = lm.model.lm_head.weight.detach().float().cpu()
    d = lm.ffn_down(layer).weight[:, feature].detach().float().cpu()
    idx = (W @ d).topk(k).indices.tolist()
    return lm.tokenizer.convert_ids_to_tokens(idx)


def probe_features(lm, attr, sel, n: int = 200) -> list[tuple[int, int, float]]:
    feats = [a for a in sel.atoms if a.kind == "feature"]
    feats.sort(key=lambda a: a.load, reverse=True)
    return [(a.layer, a.index, a.load) for a in feats[:n]]


def activation_ratio(before, after, pairs) -> dict:
    ratios = []
    for layer, feature, _ in pairs:
        b = float(before.act[layer, feature])
        a = float(after.act[layer, feature])
        if abs(b) > 1e-6:
            ratios.append(a / b)
    ratios_t = torch.tensor(ratios) if ratios else torch.tensor([0.0])
    return {
        "n": len(ratios),
        "mean": float(ratios_t.mean()),
        "median": float(ratios_t.median()),
        "min": float(ratios_t.min()),
        "max": float(ratios_t.max()),
        "within_20pct": float(((ratios_t - 1).abs() < 0.2).float().mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--probe-n", type=int, default=200)
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--out", default=str(OUT / "redirect_probe.json"))
    args = ap.parse_args()

    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    good0 = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad0 = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad0.h, good0.h)
    plane = 18
    dirs = dirs[plane].unsqueeze(0).repeat(dirs.shape[0], 1)
    attr = attribute(lm, good0, bad0, dirs)
    baseline = torch.load(args.baseline, weights_only=True)
    readout = dirs[-1]
    heldout = bad_extended_prompts()

    sel = select(attr, ("feature", "head"), load_target=args.target, min_t=2.0)
    pairs = probe_features(lm, attr, sel, args.probe_n)
    print(f"[{time.time()-t0:6.1f}s] {len(sel.atoms)} atoms, probe {len(pairs)} features")
    before_tokens = {(l, f): top_tokens(lm, l, f) for l, f, _ in pairs}

    results = {}
    controls = {
        "rotation_s2": (sel, 2.0),
        "removal_s1": (sel, 1.0),
        "dense_removal": (select_all(attr, only_positive=False), 1.0),
    }
    for name, (atoms, strength) in controls.items():
        editor = Editor(lm)
        editor.set_directions(dirs)
        info = editor.apply(atoms, renorm=True, strength=strength)
        bad1 = collect(lm, bad_fit_prompts()[: args.fit_prompts])
        ratio = activation_ratio(bad0, bad1, pairs)
        after_tokens = {(l, f): top_tokens(lm, l, f) for l, f, _ in pairs}
        changed = sum(before_tokens[k] != after_tokens[k] for k in before_tokens)
        res = measure(lm, baseline, readout, name, keep_texts=False)
        hold = refusal_rate(lm, heldout)
        results[name] = {
            "strength": strength,
            "atoms": len(atoms.atoms) if hasattr(atoms, "atoms") else len(atoms),
            "edit": info,
            "refusal_100": res["refusal"],
            "heldout_300": hold,
            "kl_divergence": res["kl_divergence"],
            "perplexity_wikitext2": res["perplexity_wikitext2"],
            "activation_ratio": ratio,
            "down_top_token_changed": changed,
            "down_top_token_changed_fraction": changed / max(len(pairs), 1),
            "examples": [
                {
                    "layer": l,
                    "feature": f,
                    "load": load,
                    "before": before_tokens[(l, f)],
                    "after": after_tokens[(l, f)],
                }
                for l, f, load in pairs[:12]
            ],
        }
        save_result(Path(args.out).with_name(f"redirect_{name}.json"), results[name])
        print(f"[{time.time()-t0:6.1f}s] {name:14s} "
              f"refusal={res['refusal']['refusal_rate']:.3f} "
              f"holdout={hold['refusal_rate']:.3f} kl={res['kl_divergence']:.4f} "
              f"ppl={res['perplexity_wikitext2']:.2f} "
              f"act_ratio={ratio['median']:.2f} "
              f"edge_changed={changed}/{len(pairs)}")
        editor.reset()
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
