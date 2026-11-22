"""Export the shipped family point as a checkpoint, and measure it after reload.

The family in ``41_dense_hybrid.py`` is measured in memory.  This writes the
chosen point to a normal HF checkpoint so the capability suite and any other
harness can load it, then measures the reloaded model so the reported numbers
include the round trip through safetensors and bfloat16.

The patch container is not written: its schema holds one index set per matrix
kind per layer, and this edit needs two disjoint sets (the removed atoms at
``gamma = 0`` and the target atoms at ``gamma``).  The checkpoint is the
artifact.

    PYTHONPATH=. .venv/bin/python scripts/45_export_family.py \
        --q 0.95 --gamma 0.45 --name taa_target_q0.95_g0.45
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
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="1.5b", choices=("1.5b", "0.5b", "3.5b"))
    ap.add_argument("--plane", type=int, default=0)
    ap.add_argument("--q", type=float, default=0.95)
    ap.add_argument("--gamma", type=float, default=0.45)
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--skip-measure", action="store_true")
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
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    dirs = dirs[plane].unsqueeze(0).repeat(dirs.shape[0], 1)
    attr = attribute(lm, good, bad, dirs)
    atoms = select_all(attr, only_positive=False).atoms
    baseline = logprobs(lm, good_eval_prompts())
    accept = token_ids(lm.tokenizer, ACCEPT_WORDS)
    refuse = token_ids(lm.tokenizer, REFUSE_WORDS)
    grads, _ = acceptance_gradient(lm, bad_fit_prompts()[: args.fit_prompts],
                                   accept, refuse)
    a = target(grads, dirs, "grad")

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
    k = int(round(args.q * len(ranked)))
    removal, redirect = ranked[:k], ranked[k:]

    editor = HybridEdit(lm)
    info = editor.apply(removal, redirect, dirs, a, args.gamma)
    print(f"[{time.time()-t0:6.1f}s] {len(removal):,} removed, {len(redirect):,} redirected, "
          f"relative change {info['relative_change']:.4f}")

    model_dir = OUT / "export" / args.name
    model_dir.mkdir(parents=True, exist_ok=True)
    lm.model.save_pretrained(model_dir, safe_serialization=True)
    lm.tokenizer.save_pretrained(model_dir)
    print(f"[{time.time()-t0:6.1f}s] saved {model_dir}")

    result = {
        "model": spec.name,
        "plane": plane,
        "q": args.q,
        "gamma": args.gamma,
        "removal_atoms": len(removal),
        "redirect_atoms": len(redirect),
        "edit": info,
        "checkpoint": str(model_dir),
    }
    # The editor holds the LoadedModel, so it must go first or the second copy
    # of the weights cannot fit beside the first.
    del editor, lm
    torch.cuda.empty_cache()

    if not args.skip_measure:
        reloaded = load_model(model_dir)
        res = measure(reloaded, baseline, dirs[-1], args.name)
        hold = refusal_rate(reloaded, bad_extended_prompts())
        result.update({
            "refusal_100": res["refusal"],
            "heldout_300": hold,
            "kl_divergence": res["kl_divergence"],
            "perplexity_wikitext2": res["perplexity_wikitext2"],
        })
        print(f"[{time.time()-t0:6.1f}s] reloaded: "
              f"ref={res['refusal']['refusal_rate']:.3f} "
              f"held={hold['refusal_rate']:.3f} "
              f"kl={res['kl_divergence']:.4f} "
              f"ppl={res['perplexity_wikitext2']:.2f}")

    path = OUT / "export" / f"{args.name}.json"
    path.write_text(json.dumps(result, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    sys.exit(main())
