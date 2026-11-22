"""Redirect toward a concrete acceptance phrase instead of a token set.

The acceptance target in the main study is the gradient of

    logsumexp(logit[accept set]) - logsumexp(logit[refuse set])

over nine acceptance openers and six refusal openers.  That raises the whole
acceptance family, but it does not name the language the model should write.

This script targets one phrase, "I can do that.".  For each harmful prompt it
teacher-forces the chat prompt plus the phrase, computes the log probability of
the phrase tokens, and differentiates with respect to the residual planes at the
last prompt position.  The mean gradient is the direction that raises that
phrase, including the effect of every later block.  The redirect then uses it as
``a`` in

    d_f' = d_f - alpha_f r + gamma alpha_f a.

The phrase log probability before and after the edit measures the language
replacement directly, next to refusal, held-out refusal, KL and perplexity.

    PYTHONPATH=. .venv/bin/python scripts/37_phrase_redirect.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from libershift import collect, directions, load_model, target
from libershift.attribute import attribute
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import (
    bad_eval_prompts,
    bad_extended_prompts,
    bad_fit_prompts,
    good_eval_prompts,
    good_fit_prompts,
)
from libershift.edit import select
from libershift.evaluate import logprobs, measure, refusal_rate
from libershift.redirect import WeightSnapshot, apply_redirect


def phrase_logprob(lm, prompt: str, phrase: str) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Log probability of ``phrase`` after ``prompt``, plus the token ids."""
    prompt_text = lm.chat([prompt])[0]
    prompt_ids = lm.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(lm.device)
    full_ids = lm.tokenizer(prompt_text + phrase, return_tensors="pt")["input_ids"].to(lm.device)
    prompt_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_len:
        raise ValueError("phrase produced no new tokens")
    if not torch.equal(full_ids[:, :prompt_len], prompt_ids):
        raise ValueError("tokenizer changed the prompt when the phrase was appended")
    out = lm.model(input_ids=full_ids, output_hidden_states=True, use_cache=False)
    logits = out.logits[0, :-1, :].float()
    target_ids = full_ids[0, 1:]
    logprobs = F.log_softmax(logits, dim=-1)
    positions = torch.arange(prompt_len - 1, full_ids.shape[1] - 1, device=lm.device)
    score = logprobs[positions].gather(1, target_ids[positions, None]).sum()
    return score, prompt_len, out.hidden_states


def phrase_gradient(lm, prompts: list[str], phrase: str) -> tuple[torch.Tensor, float]:
    """Mean ``d log P(phrase) / d h[p]`` at the last prompt position."""
    total = torch.zeros(lm.n_layers + 1, lm.hidden_size, dtype=torch.float64)
    score_sum = 0.0
    count = 0
    for prompt in prompts:
        score, prompt_len, hidden = phrase_logprob(lm, prompt, phrase)
        grads = torch.autograd.grad(score, hidden)
        for plane, grad in enumerate(grads):
            total[plane] += grad[0, prompt_len - 1, :].float().double().cpu()
        score_sum += float(score.detach())
        count += 1
    return (total / max(count, 1)).float(), score_sum / max(count, 1)


@torch.no_grad()
def mean_phrase_score(lm, prompts: list[str], phrase: str) -> float:
    scores = []
    for prompt in prompts:
        score, _, _ = phrase_logprob(lm, prompt, phrase)
        scores.append(float(score))
    return sum(scores) / len(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", default="I can do that.")
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--eval-prompts", type=int, default=100)
    ap.add_argument("--over", type=float, default=0.90)
    ap.add_argument("--plane", type=int, default=18)
    ap.add_argument("--gammas", default="0.3,0.75")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "redirect" / "phrase_target.json"))
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
    eval_prompts = bad_eval_prompts()[: args.eval_prompts]
    print(f"[{time.time()-t0:6.1f}s] {len(sel.atoms)} atoms, phrase {args.phrase!r}")

    grads, fit_score = phrase_gradient(lm, bad_fit_prompts()[: args.fit_prompts], args.phrase)
    a = target(grads, dirs, "grad")
    before_score = mean_phrase_score(lm, eval_prompts, args.phrase)
    before_refusal = refusal_rate(lm, eval_prompts, keep_texts=False)
    print(f"[{time.time()-t0:6.1f}s] phrase score before {before_score:.3f} "
          f"(fit {fit_score:.3f}), refusal {before_refusal['refusal_rate']:.3f}")

    results = {
        "model": QWEN_1_5B.name,
        "phrase": args.phrase,
        "plane": args.plane,
        "selection": sel.summary(),
        "fit_phrase_logprob": fit_score,
        "before": {
            "phrase_logprob": before_score,
            "refusal_100": before_refusal,
        },
        "results": {},
    }
    snapshot = WeightSnapshot(lm)
    for gamma in [float(x) for x in args.gammas.split(",")]:
        info = apply_redirect(lm, sel, dirs, a, gamma, snapshot=snapshot)
        res = measure(lm, baseline, dirs[-1], f"phrase_g{gamma}", keep_texts=False)
        hold = refusal_rate(lm, bad_extended_prompts())
        after_score = mean_phrase_score(lm, eval_prompts, args.phrase)
        results["results"][f"g{gamma:g}"] = {
            "gamma": gamma,
            "edit": info,
            "phrase_logprob": after_score,
            "phrase_delta": after_score - before_score,
            "refusal_100": res["refusal"],
            "heldout_300": hold,
            "kl_divergence": res["kl_divergence"],
            "perplexity_wikitext2": res["perplexity_wikitext2"],
        }
        if args.samples:
            results["results"][f"g{gamma:g}"]["samples"] = [
                {"prompt": p, "response": t}
                for p, t in zip(eval_prompts[: args.samples],
                                lm.generate(eval_prompts[: args.samples], max_new_tokens=60))
            ]
        print(f"[{time.time()-t0:6.1f}s] gamma {gamma:g}: "
              f"ref={res['refusal']['refusal_rate']:.3f} "
              f"held={hold['refusal_rate']:.3f} "
              f"kl={res['kl_divergence']:.4f} "
              f"phrase {before_score:.3f}->{after_score:.3f}")
        snapshot.restore()
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
