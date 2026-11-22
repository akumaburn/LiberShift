"""Two ways to make the redirect smarter, measured against the uniform target.

The redirect moves each atom's refusal component into a target direction ``a_l``.
The shipped target is the unit acceptance gradient, one direction per plane, with
the same ``gamma`` on every layer.  Both choices are arbitrary, and this script
tests the two obvious alternatives at the operating point.

Target variants
---------------
``grad``        the shipped unit acceptance gradient.
``grad_orth``   orthogonalised against the refusal direction ``r``.
``grad_good``   orthogonalised against the mean harmless top-1 gradient.
``grad_pca``    orthogonalised against the top principal directions of the
                per-prompt harmless top-1 gradient, which are the rows of the
                harmless logit Jacobian that carry the most variance.
``neg_r``       the pilot's mirrored control.

The acceptance gradient raises the acceptance score.  It says nothing about the
harmless next-token distribution, and KL is the price of moving that.  The
``grad_good`` and ``grad_pca`` targets remove the component that pushes the
harmless next token, so they should keep more of the conversion per unit KL.

Per-layer allocation
--------------------
Every layer's redirect contributes some acceptance and some KL.  This measures
both per layer, then sets ``gamma_l`` in proportion to the measured ratio
``d acceptance / d KL`` instead of applying one gamma everywhere.

    PYTHONPATH=. .venv/bin/python scripts/44_smart_redirect.py
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
    acceptance_score,
    collect,
    directions,
    load_model,
    token_ids,
    unit,
)
from libershift.attribute import attribute
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import (
    bad_eval_prompts,
    bad_extended_prompts,
    bad_fit_prompts,
    good_eval_prompts,
    good_fit_prompts,
)
from libershift.edit import select_all
from libershift.evaluate import kl_vs_baseline, logprobs, measure, refusal_rate
from libershift.redirect import (
    HybridEdit,
    harmlessness_basis,
    project_out,
    top_token_gradient,
)


def score_accept(lm, prompts, accept, refuse) -> float:
    values = []
    for i in range(0, len(prompts), lm.batch_size):
        enc = lm.encode(prompts[i : i + lm.batch_size])
        with torch.no_grad():
            logits = lm.model(**enc, use_cache=False).logits[:, -1, :].float()
        values.append(acceptance_score(logits, accept, refuse))
    return float(torch.cat(values).mean())


def score_kl(lm, baseline, prompts) -> float:
    return kl_vs_baseline(baseline, logprobs(lm, prompts))


def build_targets(grads, r, grads_good, basis) -> dict:
    def clean(x):
        a = unit(x)
        a = a.clone()
        a[0] = 0.0
        return a

    r_flat = r
    return {
        "grad": clean(grads),
        "grad_orth": clean(grads - (grads * r_flat).sum(-1, keepdim=True) * r_flat),
        "grad_good": clean(
            grads - (grads * grads_good).sum(-1, keepdim=True) * grads_good
        ),
        "grad_pca": clean(project_out(grads, basis)),
        "neg_r": clean(-r_flat),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=32,
                    help="principal directions removed from the target")
    ap.add_argument("--targets", default="grad,grad_orth,grad_good,grad_pca")
    ap.add_argument("--points", default="0.45:0.3")
    ap.add_argument("--layers", default="none,efficiency")
    ap.add_argument("--plane", type=int, default=18)
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--out", default=str(OUT / "redirect" / "smart.json"))
    args = ap.parse_args()

    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    dirs = dirs[args.plane].unsqueeze(0).repeat(dirs.shape[0], 1)
    attr = attribute(lm, good, bad, dirs)
    atoms = select_all(attr, only_positive=False).atoms
    good_eval = good_eval_prompts()
    baseline = logprobs(lm, good_eval)
    accept = token_ids(lm.tokenizer, ACCEPT_WORDS)
    refuse = token_ids(lm.tokenizer, REFUSE_WORDS)
    grads, _ = acceptance_gradient(lm, bad_fit_prompts()[: args.fit_prompts],
                                   accept, refuse)
    grads_good, per_prompt = top_token_gradient(
        lm, good_fit_prompts()[: args.fit_prompts]
    )
    torch.cuda.empty_cache()
    basis = harmlessness_basis(per_prompt, args.rank)
    targets = build_targets(grads, dirs, grads_good, basis)
    del per_prompt
    print(f"[{time.time()-t0:6.1f}s] {len(atoms)} atoms, "
          f"{len(targets)} targets, harmless basis rank {args.rank}")

    # The efficiency ranking, unchanged from the shipped hybrid.
    n_heads, head_dim = lm.n_heads, lm.head_dim
    proxy = {}
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
    ranked = sorted(
        atoms,
        key=lambda atom: atom.load / max(proxy[(atom.layer, atom.kind, atom.index)], 1e-12),
    )

    points = [tuple(float(v) for v in p.split(":")) for p in args.points.split(",")]
    splits = {}
    for q, gamma in points:
        k = int(round(q * len(ranked)))
        splits[(q, gamma)] = (ranked[:k], ranked[k:])

    results = {
        "model": QWEN_1_5B.name,
        "plane": args.plane,
        "atoms": len(atoms),
        "ranking": "refusal load per unit of first-order harmless write",
        "harmless_basis_rank": args.rank,
        "results": {},
    }

    def record(label, res, hold, info, **extra):
        results["results"][label] = {
            "refusal_100": res["refusal"],
            "heldout_300": hold,
            "kl_divergence": res["kl_divergence"],
            "perplexity_wikitext2": res["perplexity_wikitext2"],
            "edit": info,
            **extra,
        }
        print(
            f"[{time.time()-t0:6.1f}s] {label:28s} "
            f"ref={res['refusal']['refusal_rate']:.3f} "
            f"held={hold['refusal_rate']:.3f} "
            f"kl={res['kl_divergence']:.4f} "
            f"ppl={res['perplexity_wikitext2']:.2f}"
        )

    # 1. Target variants, uniform gamma.
    editor = HybridEdit(lm)
    for q, gamma in points:
        removal, redirect = splits[(q, gamma)]
        for name in args.targets.split(","):
            label = f"{name}_q{q:g}_g{gamma:g}"
            info = editor.apply(removal, redirect, dirs, targets[name], gamma)
            res = measure(lm, baseline, dirs[-1], label, keep_texts=False)
            hold = refusal_rate(lm, bad_extended_prompts())
            record(label, res, hold, info, target=name, quantile=q, gamma=gamma)
            editor.restore()

    # 2. Per-layer allocation: measure each layer's acceptance and KL at the
    # operating point, then spend gamma where conversion per unit KL is highest.
    if "efficiency" in args.layers.split(","):
        q, gamma = points[0]
        removal, redirect = splits[(q, gamma)]
        a = targets["grad"]
        bad_eval = bad_eval_prompts()
        editor.apply(removal, redirect, dirs, a, 0.0)
        base_accept = score_accept(lm, bad_eval, accept, refuse)
        base_kl = score_kl(lm, baseline, good_eval)
        rows = []
        for layer in range(lm.n_layers):
            editor.restore()
            editor.apply(removal, redirect, dirs, a, 0.0, gamma_by_layer={layer: gamma})
            d_accept = score_accept(lm, bad_eval, accept, refuse) - base_accept
            d_kl = score_kl(lm, baseline, good_eval) - base_kl
            rows.append({"layer": layer, "d_accept": d_accept, "d_kl": d_kl,
                         "ratio": d_accept / max(d_kl, 1e-6)})
        editor.restore()

        ratios = [row["ratio"] for row in rows]
        positive = sorted(r for r in ratios if r > 0)
        scale = positive[len(positive) // 2] if positive else 1.0
        weights = {row["layer"]: min(3.0, max(0.0, row["ratio"] / scale))
                   for row in rows}
        allocation = {layer: gamma * weight for layer, weight in weights.items()}
        results["layer_efficiency"] = {
            "rows": rows,
            "median_ratio": scale,
            "weights": weights,
            "gamma_by_layer": allocation,
        }
        for row in rows:
            print(f"[{time.time()-t0:6.1f}s] layer {row['layer']:2d} "
                  f"daccept={row['d_accept']:+.4f} dkl={row['d_kl']:+.4f} "
                  f"ratio={row['ratio']:+.2f} weight={weights[row['layer']]:.2f}")

        for q2, gamma2 in points:
            removal, redirect = splits[(q2, gamma2)]
            label = f"efficiency_q{q2:g}_g{gamma2:g}"
            info = editor.apply(removal, redirect, dirs, a, gamma2,
                                gamma_by_layer={l: gamma2 * w for l, w in weights.items()})
            res = measure(lm, baseline, dirs[-1], label, keep_texts=False)
            hold = refusal_rate(lm, bad_extended_prompts())
            record(label, res, hold, info, target="grad", quantile=q2, gamma=gamma2,
                   allocated=True)
            editor.restore()

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2))
    print("wrote", target)


if __name__ == "__main__":
    sys.exit(main())
