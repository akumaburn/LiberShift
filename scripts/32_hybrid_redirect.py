"""Hybrid edit: remove the atoms whose redirect disturbs harmless prompts most.

The redirect adds a target component to every selected column.  That component
can move harmless-prompt logits as well as harmful ones.  The hypothesis is that
a subset of atoms carries most of that unintended movement, and that removing
the refusal component from those atoms (instead of redirecting it) buys KL at a
small cost in conversion.

The per-atom signal is the first-order size of the *extra* write on harmless
prompts.  For feature ``f`` the extra write is ``c_f * gamma * alpha_f * a``, so
its magnitude on the harmless set is ``gamma * |alpha_f| * rms(c_f)``.  For head
``h`` it is ``gamma * rms(o_h . beta_h)``.  Atoms are ranked by that quantity
(``damage``), or by refusal load per unit of it (``efficiency``); the top ``q``
fraction is removed and the rest redirected.

Run:

    PYTHONPATH=. .venv/bin/python scripts/32_hybrid_redirect.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from libershift import (
    ACCEPT_WORDS,
    REFUSE_WORDS,
    acceptance_gradient,
    collect,
    directions,
    load_model,
    target,
    token_ids,
)
from libershift.attribute import attribute
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import (
    bad_extended_prompts,
    bad_fit_prompts,
    good_eval_prompts,
    good_fit_prompts,
)
from libershift.edit import select
from libershift.evaluate import logprobs, measure, refusal_rate
from libershift.redirect import HybridEdit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--over", type=float, default=0.90)
    ap.add_argument("--plane", type=int, default=18)
    ap.add_argument("--gammas", default="0.3,0.75")
    ap.add_argument("--quantiles", default="0.0,0.1,0.25,0.5,1.0")
    ap.add_argument("--ranks", default="damage,efficiency")
    ap.add_argument("--out", default=str(OUT / "redirect" / "hybrid_sweep.json"))
    args = ap.parse_args()

    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    dirs = dirs[args.plane].unsqueeze(0).repeat(dirs.shape[0], 1)
    attr = attribute(lm, good, bad, dirs)
    sel = select(attr, ("feature", "head"), load_target=args.over, min_t=2.0)
    baseline = logprobs(lm, good_eval_prompts())
    print(f"[{time.time()-t0:6.1f}s] {len(sel.atoms)} atoms selected")

    accept = token_ids(lm.tokenizer, ACCEPT_WORDS)
    refuse = token_ids(lm.tokenizer, REFUSE_WORDS)
    grads, _ = acceptance_gradient(
        lm, bad_fit_prompts()[: args.fit_prompts], accept, refuse
    )
    a = target(grads, dirs, "grad")

    # Extra harmless write, first order: gamma * |alpha| * rms(c), and the head
    # analogue.  The ranking itself does not depend on gamma.
    n_heads, head_dim = lm.n_heads, lm.head_dim
    proxy: dict[tuple[int, str, int], float] = {}
    for atom in sel.atoms:
        if atom.kind == "feature":
            alpha = float(attr.feature_alpha[atom.layer, atom.index])
            rms = math.sqrt(float(good.act_sq[atom.layer, atom.index]))
            proxy[(atom.layer, atom.kind, atom.index)] = abs(alpha) * rms
        else:
            beta = (lm.attn_out(atom.layer).weight.detach().float().cpu().T
                    @ dirs[atom.layer + 1].float().cpu())
            beta = beta.reshape(n_heads, head_dim)[atom.index]
            sq = good.heads_sq[atom.layer].reshape(n_heads, head_dim)[atom.index]
            proxy[(atom.layer, atom.kind, atom.index)] = math.sqrt(
                float((sq * beta.pow(2)).sum())
            )

    def key(atom):
        return (atom.layer, atom.kind, atom.index)

    ranked = {
        "damage": sorted(sel.atoms, key=lambda atom: proxy[key(atom)], reverse=True),
        "efficiency": sorted(
            sel.atoms,
            key=lambda atom: atom.load / max(proxy[key(atom)], 1e-12),
        ),
    }

    editor = HybridEdit(lm)
    results = {
        "model": QWEN_1_5B.name,
        "plane": args.plane,
        "gamma_targets": args.gammas,
        "selection": sel.summary(),
        "proxy": "gamma * |alpha| * rms(harmless activation)",
        "results": {},
    }
    quantiles = [float(x) for x in args.quantiles.split(",")]
    gammas = [float(x) for x in args.gammas.split(",")]

    for rank in args.ranks.split(","):
        atoms = ranked[rank]
        for gamma in gammas:
            for q in quantiles:
                k = int(round(q * len(atoms)))
                removal = atoms[:k]
                redirect = atoms[k:]
                label = f"{rank}_q{q:g}_g{gamma:g}"
                info = editor.apply(removal, redirect, dirs, a, gamma)
                res = measure(lm, baseline, dirs[-1], label, keep_texts=False)
                hold = refusal_rate(lm, bad_extended_prompts())
                results["results"][label] = {
                    "rank": rank,
                    "quantile": q,
                    "gamma": gamma,
                    "removal_atoms": len(removal),
                    "redirect_atoms": len(redirect),
                    "removal_load": sum(x.load for x in removal),
                    "redirect_load": sum(x.load for x in redirect),
                    "edit": info,
                    "refusal_100": res["refusal"],
                    "heldout_300": hold,
                    "kl_divergence": res["kl_divergence"],
                    "perplexity_wikitext2": res["perplexity_wikitext2"],
                }
                print(
                    f"[{time.time()-t0:6.1f}s] {label:24s} "
                    f"ref={res['refusal']['refusal_rate']:.3f} "
                    f"held={hold['refusal_rate']:.3f} "
                    f"kl={res['kl_divergence']:.4f} "
                    f"ppl={res['perplexity_wikitext2']:.2f}"
                )
                editor.restore()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
