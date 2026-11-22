"""Where does the harmless-prompt KL come from?

The KL in every table is Heretic's scorer: first-token logits on 100 harmless
prompts, against the base model.  It is a proxy for "did the edit damage the
model", but the acceptance redirect deliberately moves first-token mass from
refusal openers to acceptance openers.  Some of that KL is the intended
redistribution and some is damage, and the scalar cannot tell them apart.

This script decomposes the KL per token and reports the two parts:

* mass carried by the acceptance and refusal token sets, which the method is
  trying to move;
* mass carried by every other token, which has no reason to change.

The same decomposition is computed for plain removal and for the dense edit, so
the ratio is comparable across methods.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from libershift import (
    WeightSnapshot,
    apply_redirect,
    collect,
    directions,
    load_model,
    select,
    select_all,
    target,
)
from libershift.attribute import attribute
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_fit_prompts, capability_text, good_eval_prompts, good_fit_prompts
from libershift.edit import Editor
from libershift.evaluate import logprobs, perplexity


def decompose(baseline: torch.Tensor, current: torch.Tensor, mask: torch.Tensor) -> dict:
    """Split the per-token KL into the accept/refuse part and everything else."""
    p = baseline.double().exp()
    contrib = p * (baseline.double() - current.double())
    row = contrib.sum(dim=1)
    in_set = contrib[:, mask].sum(dim=1)
    off_set = contrib[:, ~mask].sum(dim=1)
    top = contrib.mean(dim=0)
    idx = top.topk(10).indices.tolist()
    return {
        "kl_total": float(row.mean()),
        "kl_in_accept_refuse_set": float(in_set.mean()),
        "kl_outside_set": float(off_set.mean()),
        "in_set_share": float(in_set.mean() / row.mean()) if float(row.mean()) else 0.0,
        "mean_accept_mass_base": float(p[:, mask].sum(dim=1).mean()),
        "mean_accept_mass_now": float(current.double().exp()[:, mask].sum(dim=1).mean()),
        "top_contributors_rowsum": float(top[idx].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",
                    default=str(OUT / f"{QWEN_1_5B.name}_main" / "baseline_logprobs.pt"))
    ap.add_argument("--grad-pt", default=str(OUT / "acceptance_gradient.pt"))
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--out", default=str(OUT / "kl_split.json"))
    args = ap.parse_args()

    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    payload = torch.load(args.grad_pt, weights_only=False)
    grads, dirs = payload["gradient"], payload["direction"].float()
    accept_ids, refuse_ids = payload["accept_ids"], payload["refuse_ids"]
    mask = torch.zeros(lm.model.config.vocab_size, dtype=torch.bool)
    mask[accept_ids + refuse_ids] = True

    baseline = torch.load(args.baseline, weights_only=True)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    attr = attribute(lm, good, bad, dirs)
    sel = select(attr, ("feature", "head"), load_target=args.target, min_t=2.0)
    text = capability_text()
    print(f"[{time.time()-t0:6.1f}s] ready; {len(sel.atoms)} atoms")

    # (name, target direction, gamma, atoms, removal strength).  A target of
    # None means a plain removal edit at the given strength.
    configs = [
        ("removal_90", None, None, sel, 1.0),
        ("neg_r_g0.5", target(grads, dirs, "neg_r"), 0.5, sel, None),
        ("neg_r_g1", target(grads, dirs, "neg_r"), 1.0, sel, None),
        ("grad_g0.3", target(grads, dirs, "grad"), 0.3, sel, None),
        ("grad_g0.5", target(grads, dirs, "grad"), 0.5, sel, None),
        ("grad_g0.75", target(grads, dirs, "grad"), 0.75, sel, None),
        ("dense_removal", None, None, select_all(attr, only_positive=False), 1.0),
    ]

    out = {"configs": {}, "accept_words": payload["accept_words"],
           "refuse_words": payload["refuse_words"]}
    snapshot = WeightSnapshot(lm)
    for name, a, gamma, atoms, strength in configs:
        if a is None:
            editor = Editor(lm)
            editor.set_directions(dirs)
            info = editor.apply(atoms, renorm=True, strength=strength)
        else:
            info = apply_redirect(lm, atoms, dirs, a, gamma, snapshot)
        lp = logprobs(lm, good_eval_prompts())
        d = decompose(baseline, lp, mask)
        d["perplexity_wikitext2"] = perplexity(lm, text)
        d["edit"] = info
        d["atoms"] = len(atoms.atoms)
        out["configs"][name] = d
        print(f"[{time.time()-t0:6.1f}s] {name:14s} kl={d['kl_total']:.4f} "
              f"in_set={d['kl_in_accept_refuse_set']:.4f} "
              f"({d['in_set_share']*100:5.1f}%) "
              f"outside={d['kl_outside_set']:.4f} "
              f"ppl={d['perplexity_wikitext2']:.2f}")
        if a is None:
            editor.reset()
        else:
            snapshot.restore()
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
