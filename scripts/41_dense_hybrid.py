"""The hybrid rule over the full atom set, against dense removal.

The 35k hybrid removes the atoms with the worst refusal load per unit of
harmless write and redirects the rest.  This applies the same rule to all
251,216 atoms: q=1 is dense removal, q=0 is a dense redirect, and the values
between remove the lowest-efficiency atoms and redirect the rest.

    PYTHONPATH=. .venv/bin/python scripts/41_dense_hybrid.py
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
from libershift.constants import OUT, QWEN_0_5B, QWEN_1_5B, QWEN_3_5_4B
from libershift.data import (
    bad_eval_prompts,
    bad_extended_prompts,
    bad_fit_prompts,
    good_eval_prompts,
    good_fit_prompts,
)
from libershift.edit import select_all
from libershift.evaluate import logprobs, measure, refusal_rate
from libershift.redirect import HybridEdit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", default="0.3,0.75")
    ap.add_argument("--quantiles", default="0,0.1,0.25,0.5,1.0")
    ap.add_argument("--model", default="1.5b", choices=("1.5b", "0.5b", "3.5b"))
    ap.add_argument("--plane", type=int, default=0,
                    help="refusal-direction plane; 0 uses the calibrated plane")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--responses", action="store_true",
                    help="store all 100 evaluation responses for the report split")
    ap.add_argument("--ranks", default="efficiency",
                    help="efficiency | damage | load; how the redirect set is chosen")
    # Explicit (quantile, gamma) pairs, e.g. "0.6:0.3,0.45:0.15"; overrides the
    # cartesian product of --quantiles and --gammas.
    ap.add_argument("--points", default="")
    ap.add_argument("--out", default=str(OUT / "redirect" / "dense_hybrid.json"))
    args = ap.parse_args()

    spec = {"1.5b": QWEN_1_5B, "0.5b": QWEN_0_5B, "3.5b": QWEN_3_5_4B}[args.model]
    calibrated = {"1.5b": 18, "0.5b": 14}
    if args.plane:
        plane = args.plane
    elif args.model in calibrated:
        plane = calibrated[args.model]
    else:
        raise SystemExit("this model has no calibrated plane; pass --plane")
    t0 = time.time()
    lm = load_model(spec.path)
    good = collect(lm, good_fit_prompts()[:400])
    bad = collect(lm, bad_fit_prompts()[:400])
    dirs = directions(bad.h, good.h)
    dirs = dirs[plane].unsqueeze(0).repeat(dirs.shape[0], 1)
    attr = attribute(lm, good, bad, dirs)
    atoms = select_all(attr, only_positive=False).atoms
    baseline = logprobs(lm, good_eval_prompts())
    accept = token_ids(lm.tokenizer, ACCEPT_WORDS)
    refuse = token_ids(lm.tokenizer, REFUSE_WORDS)
    grads, _ = acceptance_gradient(lm, bad_fit_prompts()[:400], accept, refuse)
    a = target(grads, dirs, "grad")
    print(f"[{time.time()-t0:6.1f}s] {len(atoms)} atoms (all signs)")

    n_heads, head_dim = lm.n_heads, lm.head_dim
    proxy: dict[tuple[int, str, int], float] = {}
    for atom in atoms:
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

    def key(name):
        if name == "efficiency":
            return lambda atom: atom.load / max(
                proxy[(atom.layer, atom.kind, atom.index)], 1e-12)
        if name == "damage":
            return lambda atom: proxy[(atom.layer, atom.kind, atom.index)]
        if name == "load":
            return lambda atom: atom.load
        raise ValueError(name)

    orderings = {name: sorted(atoms, key=key(name)) for name in args.ranks.split(",")}
    editor = HybridEdit(lm)
    results = {
        "model": spec.name,
        "plane": plane,
        "atoms": len(atoms),
        "proxy": "gamma * |alpha| * rms(harmless activation)",
        "results": {},
    }

    if args.points:
        pairs = [tuple(float(v) for v in point.split(":"))
                 for point in args.points.split(",")]
    else:
        pairs = [(q, g) for g in [float(x) for x in args.gammas.split(",")]
                 for q in [float(x) for x in args.quantiles.split(",")]]

    for rank_name, ranked in orderings.items():
        prefix = "dense" if rank_name == "efficiency" else rank_name
        for q, gamma in pairs:
            k = int(round(q * len(ranked)))
            removal, redirect = ranked[:k], ranked[k:]
            label = f"{prefix}_q{q:g}_g{gamma:g}"
            info = editor.apply(removal, redirect, dirs, a, gamma)
            res = measure(lm, baseline, dirs[-1], label, keep_texts=True)
            responses = res.pop("responses")
            texts = [r["response"] for r in responses]
            hold = refusal_rate(lm, bad_extended_prompts())
            results["results"][label] = {
                "rank": rank_name,
                "quantile": q,
                "gamma": gamma,
                "removal_atoms": len(removal),
                "redirect_atoms": len(redirect),
                "edit": info,
                "refusal_100": res["refusal"],
                "heldout_300": hold,
                "kl_divergence": res["kl_divergence"],
                "perplexity_wikitext2": res["perplexity_wikitext2"],
            }
            if args.responses:
                results["results"][label]["responses"] = responses
            if args.samples:
                results["results"][label]["samples"] = [
                    {"prompt": p, "response": t}
                    for p, t in zip(bad_eval_prompts()[: args.samples],
                                    texts[: args.samples])
                ]
            print(f"[{time.time()-t0:6.1f}s] {label:20s} "
                  f"ref={res['refusal']['refusal_rate']:.3f} "
                  f"held={hold['refusal_rate']:.3f} "
                  f"kl={res['kl_divergence']:.4f} "
                  f"ppl={res['perplexity_wikitext2']:.2f}")
            editor.restore()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
