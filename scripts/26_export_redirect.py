"""Export an acceptance redirect as a checkpoint and an operator patch.

The sweep in ``24_gradient_redirect.py`` measures the redirect in memory.  This
script turns one operating point into the two artifacts the method ships:

* a normal HF checkpoint, written by ``save_pretrained``;
* a patch in operator form, one direction and one target per layer plus an
  index and a norm scale per atom.

``--verify-patch`` replays the patch on a pristine checkpoint and compares
first-token logits with the exported model.  The redirect recomputes
``d_f - alpha_f r + gamma alpha_f a`` from the base column, so a zero difference
shows that the stored operator, and not the stored weights, carries the edit.

The exported model is measured again after reloading.  That number is the one
to quote: it includes the round trip through safetensors and bfloat16.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from libershift import (
    ACCEPT_WORDS,
    REFUSE_WORDS,
    acceptance_gradient,
    apply_redirect,
    attribute,
    collect,
    directions,
    load_model,
    select,
    target,
    token_ids,
)
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_extended_prompts, bad_fit_prompts, good_fit_prompts
from libershift.evaluate import measure, refusal_rate
from libershift.patch import apply_patch, write_redirect_patch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="artifact name, e.g. redirect_g0.3")
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--mode", default="grad", choices=["grad", "grad_orth", "neg_r"])
    ap.add_argument("--target", type=float, default=0.90, help="fraction of attributed load")
    ap.add_argument("--plane", type=int, default=18)
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--grad-batch", type=int, default=8)
    ap.add_argument("--baseline",
                    default=str(OUT / f"{QWEN_1_5B.name}_main" / "baseline_logprobs.pt"))
    ap.add_argument("--reuse-grad", action="store_true",
                    help="reuse OUT/acceptance_gradient.pt instead of recomputing it")
    ap.add_argument("--verify-patch", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    export = OUT / "export"
    export.mkdir(parents=True, exist_ok=True)
    model_dir = export / args.name
    patch_json = export / f"{args.name}.json"
    grad_path = OUT / "acceptance_gradient.pt"
    baseline = torch.load(args.baseline, weights_only=True)

    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    dirs = dirs[args.plane].unsqueeze(0).repeat(dirs.shape[0], 1).float()
    attr = attribute(lm, good, bad, dirs)
    sel = select(attr, ("feature", "head"), load_target=args.target, min_t=2.0)
    print(f"[{time.time()-t0:6.1f}s] {len(sel.atoms)} atoms, "
          f"captured {sel.captured_fraction:.3f}")

    if args.reuse_grad and grad_path.exists():
        payload = torch.load(grad_path, weights_only=False)
        grads = payload["gradient"]
        print(f"[{time.time()-t0:6.1f}s] reused {grad_path}")
    else:
        accept_ids = token_ids(lm.tokenizer, ACCEPT_WORDS)
        refuse_ids = token_ids(lm.tokenizer, REFUSE_WORDS)
        grads, score = acceptance_gradient(
            lm, bad_fit_prompts()[: args.fit_prompts], accept_ids, refuse_ids, args.grad_batch
        )
        torch.save(
            {
                "gradient": grads,
                "plane": args.plane,
                "direction": dirs,
                "accept_ids": accept_ids,
                "refuse_ids": refuse_ids,
                "accept_words": ACCEPT_WORDS,
                "refuse_words": REFUSE_WORDS,
                "score": score,
            },
            grad_path,
        )
        print(f"[{time.time()-t0:6.1f}s] acceptance gradient ready, mean score {score:.3f}")

    a = target(grads, dirs, args.mode)
    ops: dict[int, dict[str, dict]] = {}
    info = apply_redirect(lm, sel, dirs, a, args.gamma, ops=ops)
    print(f"[{time.time()-t0:6.1f}s] redirect applied at gamma {args.gamma:g}, "
          f"relative weight change {info['relative_change']:.4f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    lm.model.save_pretrained(model_dir, safe_serialization=True)
    lm.tokenizer.save_pretrained(model_dir)
    print(f"[{time.time()-t0:6.1f}s] saved {model_dir}")
    del lm
    torch.cuda.empty_cache()

    patch = write_redirect_patch(
        patch_json,
        ops,
        {
            "model": QWEN_1_5B.name,
            "method": "acceptance-redirect",
            "gamma": args.gamma,
            "selection": {
                "target": args.target,
                "atoms": len(sel.atoms),
                "captured_fraction": sel.captured_fraction,
                "kind": args.mode,
                "plane": args.plane,
                "edit": info,
            },
        },
    )
    print(f"[{time.time()-t0:6.1f}s] patch: {patch['layer_count']} layers, "
          f"{patch['atoms']} atoms, {patch['array_bytes']} bytes")

    result = {"patch": {k: v for k, v in patch.items() if k != "matrices"},
              "atoms": len(sel.atoms), "gamma": args.gamma, "mode": args.mode,
              "selection": sel.summary(), "edit": info}

    if args.verify_patch:
        patched_dir = export / f"{args.name}_from_patch"
        applied = apply_patch(QWEN_1_5B.path, patch_json, patched_dir)
        print(f"[{time.time()-t0:6.1f}s] replayed patch: {applied['applied_atoms']} atoms")

        exported = load_model(model_dir)
        replayed = load_model(patched_dir)
        prompts = good_fit_prompts()[:16]
        la = exported.first_token_logits(prompts)
        lb = replayed.first_token_logits(prompts)
        diff = float((la - lb).abs().max())
        print(f"[{time.time()-t0:6.1f}s] max |logit difference| exported vs patched: {diff:.3e}")
        del replayed
        torch.cuda.empty_cache()

        heldout = bad_extended_prompts()
        res = measure(exported, baseline, dirs[-1], args.name, keep_texts=False)
        hold = refusal_rate(exported, heldout)
        result["replay"] = {
            "apply": applied,
            "prompts": len(prompts),
            "max_logit_difference": diff,
        }
        result["exported"] = {
            "refusal_100": res["refusal"],
            "heldout_300": hold,
            "kl_divergence": res["kl_divergence"],
            "perplexity_wikitext2": res["perplexity_wikitext2"],
            "bad_projection": res["bad_projection"],
            "good_projection": res["good_projection"],
        }
        print(f"[{time.time()-t0:6.1f}s] exported model: "
              f"refusal={res['refusal']['refusal_rate']:.3f} "
              f"holdout={hold['refusal_rate']:.3f} "
              f"kl={res['kl_divergence']:.4f} "
              f"ppl={res['perplexity_wikitext2']:.2f}")
        del exported
        torch.cuda.empty_cache()

    verify_path = export / f"{args.name}_verify.json"
    verify_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {verify_path}")


if __name__ == "__main__":
    sys.exit(main())
