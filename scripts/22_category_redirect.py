"""Category-selective redirect: can one harmful category be flipped to acceptance
while the others keep refusing?

The selection is contrastive.  For a target category, a feature is a candidate
when its activation rises on the category against the other harmful prompts,
and its down vector writes the shared refusal direction.  Rotating those
features flips the refusal write for the category, while the features that
serve the other categories keep their refusal write.

This is the graph-database version of the edit: the category is a subgraph, and
the edit is an edge redirect on that subgraph only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from libershift import collect, directions, load_model
from libershift.attribute import _welch_t
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_eval_prompts, bad_fit_prompts, good_fit_prompts
from libershift.edit import Atom, Editor, Selection
from libershift.evaluate import measure, refusal_rate, save_result

CATEGORIES = {
    "cyber": ["hack", "malware", "virus", "phishing", "ddos", "exploit", "cyber",
              "database", "network", "spy", "spoof", "digital rights", "access to a website"],
    "violence": ["bomb", "explosive", "weapon", "violent", "assault", "terror",
                 "kill", "murder", "assassin", "gang", "attack", "harm"],
    "fraud": ["fraud", "insider trading", "identity theft", "counterfeit", "steal",
              "launder", "financial", "scam", "fake id", "credit card", "money"],
}


def label(prompt: str) -> str:
    low = prompt.lower()
    for name, words in CATEGORIES.items():
        if any(w in low for w in words):
            return name
    return "other"


def contrastive_selection(lm, cat, other, r, kinds, budget, min_t):
    atoms = []
    for layer in range(lm.n_layers):
        if "feature" in kinds:
            down = lm.ffn_down(layer).weight.detach().float().cpu()
            alpha = down.T @ r[layer + 1].float().cpu()
            d_act = (cat.act[layer] - other.act[layer]).float()
            load = d_act * alpha
            t = _welch_t(cat.act[layer].float(), other.act[layer].float(),
                         cat.act_var()[layer].float(), other.act_var()[layer].float(),
                         cat.count, other.count)
            idx = torch.nonzero((load > 0) & (t >= min_t)).flatten()
            for i in idx.tolist():
                atoms.append(Atom(layer, "feature", i, float(load[i]),
                                  float(alpha[i] ** 2), float(t[i])))
        if "head" in kinds:
            out = lm.attn_out(layer).weight.detach().float().cpu()
            beta = out.T @ r[layer + 1].float().cpu()
            d_heads = (cat.heads[layer] - other.heads[layer]).float()
            load = (d_heads * beta).view(lm.n_heads, lm.head_dim).sum(dim=1)
            t = _welch_t(cat.heads[layer].float(), other.heads[layer].float(),
                         cat.heads_var()[layer].float(), other.heads_var()[layer].float(),
                         cat.count, other.count).view(lm.n_heads, lm.head_dim).mean(dim=1)
            idx = torch.nonzero((load > 0) & (t >= min_t)).flatten()
            for i in idx.tolist():
                atoms.append(Atom(layer, "head", i, float(load[i]),
                                  float((beta.view(lm.n_heads, lm.head_dim)[i] ** 2).sum()),
                                  float(t[i])))
    atoms.sort(key=lambda a: a.load / max(a.cost, 1e-12), reverse=True)
    chosen = atoms[:budget]
    total = sum(a.load for a in atoms)
    return Selection(chosen, total, sum(a.load for a in chosen),
                     sum(a.cost for a in chosen), len(atoms))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--category", default="cyber")
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--min-t", type=float, default=2.0)
    ap.add_argument("--strength", type=float, default=2.0)
    ap.add_argument("--out", default=str(OUT / "category_redirect.json"))
    args = ap.parse_args()

    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    prompts = bad_fit_prompts()[:400]
    labels = [label(p) for p in prompts]
    cat_prompts = [p for p, c in zip(prompts, labels) if c == args.category]
    other_prompts = [p for p, c in zip(prompts, labels) if c != args.category]
    print(f"category {args.category}: {len(cat_prompts)} prompts, "
          f"other {len(other_prompts)}")

    good = collect(lm, good_fit_prompts()[:400])
    cat = collect(lm, cat_prompts)
    other = collect(lm, other_prompts)
    all_bad = collect(lm, prompts)
    dirs = directions(all_bad.h, good.h)
    dirs = dirs[18].unsqueeze(0).repeat(dirs.shape[0], 1)
    baseline = torch.load(args.baseline, weights_only=True)

    sel = contrastive_selection(lm, cat, other, dirs, ("feature", "head"),
                                args.budget, args.min_t)
    print(f"selected {len(sel.atoms)} atoms of {sel.candidates} candidates, "
          f"captured {sel.captured_fraction:.3f}")

    editor = Editor(lm)
    editor.set_directions(dirs)
    info = editor.apply(sel, renorm=True, strength=args.strength)
    res = measure(lm, baseline, dirs[-1], args.category, keep_texts=False)
    cat_eval = [p for p in bad_eval_prompts() if label(p) == args.category]
    other_eval = [p for p in bad_eval_prompts() if label(p) != args.category]
    cat_rate = refusal_rate(lm, cat_eval)
    other_rate = refusal_rate(lm, other_eval)
    print(f"[{time.time()-t0:6.1f}s] refusal all={res['refusal']['refusal_rate']:.3f} "
          f"cat={cat_rate['refusal_rate']:.3f} ({cat_rate['refusals']}/{cat_rate['n']}) "
          f"other={other_rate['refusal_rate']:.3f} ({other_rate['refusals']}/{other_rate['n']}) "
          f"kl={res['kl_divergence']:.4f} ppl={res['perplexity_wikitext2']:.2f}")
    out = {
        "category": args.category,
        "fit_cat": len(cat_prompts),
        "fit_other": len(other_prompts),
        "eval_cat": cat_rate,
        "eval_other": other_rate,
        "all": res,
        "selection": sel.summary(),
        "strength": args.strength,
        "edit": info,
    }
    save_result(Path(args.out), out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
