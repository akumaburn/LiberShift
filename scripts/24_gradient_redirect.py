"""Redirect refusal edges toward the acceptance gradient.

The pilot redirected each selected atom with ``a = -r``, which is the direction
that points away from refusal.  It is not a direction that points toward
acceptance.  This script redirects toward the gradient of the acceptance score
instead.

For a harmful prompt the acceptance score is

    s = logsumexp(logit[accept set]) - logsumexp(logit[refuse set])

at the last prompt position.  ``ds/dh[p]`` is the direction in which plane ``p``
must move to raise acceptance, including the effect of every later block.  The
redirect for layer ``l`` is

    d_f' = d_f - alpha_f r + gamma * alpha_f a_l

with ``a_l`` the unit gradient at plane ``l + 1``.  ``gamma`` scales the
acceptance component against the refusal component it replaces; ``gamma = 0`` is
plain removal and ``gamma = 1`` gives the replacement the same magnitude.

Every configuration uses the same 90% selection as the pilot, so the numbers are
directly comparable with ``rotation_plane18.json`` and ``redirect_probe.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from libershift import (
    ACCEPT_WORDS,
    REFUSE_WORDS,
    WeightSnapshot,
    acceptance_gradient,
    acceptance_score,
    apply_redirect,
    collect,
    directions,
    load_model,
    select,
    target,
    token_ids,
)
from libershift.attribute import attribute
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_extended_prompts, bad_fit_prompts, good_fit_prompts
from libershift.evaluate import measure, refusal_rate, save_result


def mean_score(lm, prompts: list[str], accept_ids: list[int],
               refuse_ids: list[int]) -> float:
    logits = lm.first_token_logits(prompts)
    return float(acceptance_score(logits, accept_ids, refuse_ids).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",
                    default=str(OUT / f"{QWEN_1_5B.name}_main" / "baseline_logprobs.pt"))
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--plane", type=int, default=18)
    ap.add_argument("--modes", default="neg_r,grad,grad_orth")
    ap.add_argument("--gammas", default="0.5,1.0,1.5")
    ap.add_argument("--grad-batch", type=int, default=8)
    ap.add_argument("--out", default=str(OUT / "gradient_redirect.json"))
    ap.add_argument("--grad-out", default=str(OUT / "acceptance_gradient.pt"))
    args = ap.parse_args()

    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    accept_ids = token_ids(lm.tokenizer, ACCEPT_WORDS)
    refuse_ids = token_ids(lm.tokenizer, REFUSE_WORDS)

    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    dirs = dirs[args.plane].unsqueeze(0).repeat(dirs.shape[0], 1).float()
    attr = attribute(lm, good, bad, dirs)
    baseline = torch.load(args.baseline, weights_only=True)
    heldout = bad_extended_prompts()
    sel = select(attr, ("feature", "head"), load_target=args.target, min_t=2.0)
    print(f"[{time.time()-t0:6.1f}s] {len(sel.atoms)} atoms, "
          f"captured {sel.captured_fraction:.3f}, cost {sel.total_cost:.1f}")

    fit = bad_fit_prompts()[: args.fit_prompts]
    score_before = mean_score(lm, fit, accept_ids, refuse_ids)
    grads, grad_score = acceptance_gradient(lm, fit, accept_ids, refuse_ids, args.grad_batch)
    print(f"[{time.time()-t0:6.1f}s] acceptance gradient ready; "
          f"mean score on fit prompts {grad_score:.3f} "
          f"(cross-check {score_before:.3f})")
    torch.save(
        {
            "gradient": grads,
            "plane": args.plane,
            "direction": dirs,
            "accept_ids": accept_ids,
            "refuse_ids": refuse_ids,
            "accept_words": ACCEPT_WORDS,
            "refuse_words": REFUSE_WORDS,
            "score": grad_score,
        },
        args.grad_out,
    )

    cosines = {}
    for plane in range(1, lm.n_layers + 1):
        g = grads[plane]
        gn = float(g.norm())
        cosines[f"plane_{plane}"] = {
            "cos_grad_r": float(F.cosine_similarity(g, dirs[plane], dim=0)),
            "cos_grad_neg_r": float(F.cosine_similarity(g, -dirs[plane], dim=0)),
            "grad_norm": gn,
            "orth_fraction": float((g - (g @ dirs[plane]) * dirs[plane]).norm() / gn)
            if gn > 0 else 0.0,
        }
    print("cos(grad, -r) by plane 1..%d: %s" % (
        lm.n_layers,
        [round(cosines[f"plane_{p}"]["cos_grad_neg_r"], 3) for p in range(1, lm.n_layers + 1)],
    ))

    results = {}
    snapshot = WeightSnapshot(lm)
    for mode in args.modes.split(","):
        a = target(grads, dirs, mode)
        for gamma in [float(x) for x in args.gammas.split(",")]:
            label = f"{mode}_g{gamma:g}"
            info = apply_redirect(lm, sel, dirs, a, gamma, snapshot)
            res = measure(lm, baseline, dirs[-1], label, keep_texts=True)
            hold = refusal_rate(lm, heldout)
            after = mean_score(lm, fit, accept_ids, refuse_ids)
            entry = {
                "label": label,
                "mode": mode,
                "gamma": gamma,
                "atoms": len(sel.atoms),
                "captured_fraction": sel.captured_fraction,
                "edit": info,
                "refusal_100": res["refusal"],
                "heldout_300": hold,
                "kl_divergence": res["kl_divergence"],
                "perplexity_wikitext2": res["perplexity_wikitext2"],
                "bad_projection": res["bad_projection"],
                "good_projection": res["good_projection"],
                "acceptance_score_fit_before": score_before,
                "acceptance_score_fit_after": after,
                "acceptance_score_delta": after - score_before,
                "responses": res["responses"],
            }
            results[label] = entry
            save_result(Path(args.out).with_name(f"gradient_{label}.json"), entry)
            print(f"[{time.time()-t0:6.1f}s] {label:16s} "
                  f"refusal={res['refusal']['refusal_rate']:.3f} "
                  f"opening={res['refusal']['opening_refusal_rate']:.3f} "
                  f"holdout={hold['refusal_rate']:.3f} "
                  f"kl={res['kl_divergence']:.4f} "
                  f"ppl={res['perplexity_wikitext2']:.2f} "
                  f"dscore={after - score_before:+.2f} "
                  f"rel_dw={info['relative_change']:.4f}")
            snapshot.restore()
            torch.cuda.empty_cache()

    payload = {"selection": sel.summary(), "cosines": cosines,
               "score_before": score_before, "results": results}
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
