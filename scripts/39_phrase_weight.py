"""Sweep the phrase weight against the acceptance direction.

The target is a convex mix of two unit directions:

    base   = unit(grad logsumexp(accept) - grad logsumexp(refuse))
    phrase = unit(grad log P("I can do that."))
    a      = unit(lambda * phrase + base)

``lambda = 0`` is the acceptance gradient.  Large ``lambda`` approaches the
phrase-only target, which does not convert.  This script finds the point that
keeps conversion while making the phrase more likely, and measures the literal
phrase in the generated responses.

    PYTHONPATH=. .venv/bin/python scripts/39_phrase_weight.py
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
    collect,
    directions,
    load_model,
    target,
    token_ids,
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
from libershift.edit import select
from libershift.evaluate import logprobs, measure, refusal_rate
from libershift.redirect import WeightSnapshot, apply_redirect


def phrase_logprob(lm, prompt: str, phrase: str):
    prompt_text = lm.chat([prompt])[0]
    prompt_ids = lm.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(lm.device)
    full_ids = lm.tokenizer(prompt_text + phrase, return_tensors="pt")["input_ids"].to(lm.device)
    prompt_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_len:
        raise ValueError("phrase produced no new tokens")
    out = lm.model(input_ids=full_ids, output_hidden_states=True, use_cache=False)
    logits = out.logits[0, :-1, :].float()
    ids = full_ids[0, 1:]
    positions = torch.arange(prompt_len - 1, full_ids.shape[1] - 1, device=lm.device)
    score = F.log_softmax(logits, dim=-1)[positions].gather(1, ids[positions, None]).sum()
    return score, prompt_len, out.hidden_states


def phrase_gradient(lm, prompts, phrase):
    total = torch.zeros(lm.n_layers + 1, lm.hidden_size, dtype=torch.float64)
    count = 0
    for prompt in prompts:
        score, prompt_len, hidden = phrase_logprob(lm, prompt, phrase)
        grads = torch.autograd.grad(score, hidden)
        for plane, grad in enumerate(grads):
            total[plane] += grad[0, prompt_len - 1, :].float().double().cpu()
        count += 1
    return (total / max(count, 1)).float()


def set_gradient(lm, prompts, ids, batch_size=8):
    total = torch.zeros(lm.n_layers + 1, lm.hidden_size, dtype=torch.float64)
    count = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = lm.encode(batch)
        out = lm.model(**enc, output_hidden_states=True, use_cache=False)
        logits = out.logits[:, -1, :].float()
        score = torch.logsumexp(logits[:, ids], dim=-1).sum()
        grads = torch.autograd.grad(score, out.hidden_states)
        for plane, grad in enumerate(grads):
            total[plane] += grad[:, -1, :].float().double().cpu().sum(0)
        count += len(batch)
    return (total / max(count, 1)).float()


@torch.no_grad()
def mean_phrase_score(lm, prompts, phrase):
    scores = []
    for prompt in prompts:
        score, _, _ = phrase_logprob(lm, prompt, phrase)
        scores.append(float(score))
    return sum(scores) / len(scores)


def phrase_rates(responses, phrase):
    low = phrase.lower().strip()
    starts = [r.strip().lower().startswith(low) for r in responses]
    anywhere = [low in r.lower() for r in responses]
    return sum(starts) / len(starts), sum(anywhere) / len(anywhere)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", default="I can do that.")
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--eval-prompts", type=int, default=100)
    ap.add_argument("--phrase-eval", type=int, default=50)
    ap.add_argument("--over", type=float, default=0.90)
    ap.add_argument("--plane", type=int, default=18)
    ap.add_argument("--gammas", default="0.3,0.75")
    ap.add_argument("--lambdas", default="0,0.25,0.5,1,2,4")
    ap.add_argument("--out", default=str(OUT / "redirect" / "phrase_weight.json"))
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
    phrase_prompts = eval_prompts[: args.phrase_eval]
    print(f"[{time.time()-t0:6.1f}s] {len(sel.atoms)} atoms, phrase {args.phrase!r}")

    phrase_grad = phrase_gradient(lm, bad_fit_prompts()[: args.fit_prompts], args.phrase)
    accept = token_ids(lm.tokenizer, ACCEPT_WORDS)
    refuse = token_ids(lm.tokenizer, REFUSE_WORDS)
    grad_accept = set_gradient(lm, bad_fit_prompts()[: args.fit_prompts], accept)
    grad_refuse = set_gradient(lm, bad_fit_prompts()[: args.fit_prompts], refuse)
    base = target(grad_accept - grad_refuse, dirs, "grad")
    phrase_dir = target(phrase_grad, dirs, "grad")
    before_score = mean_phrase_score(lm, phrase_prompts, args.phrase)
    before_refusal = refusal_rate(lm, eval_prompts)
    print(f"[{time.time()-t0:6.1f}s] phrase score before {before_score:.3f}, "
          f"refusal {before_refusal['refusal_rate']:.3f}")

    results = {
        "model": QWEN_1_5B.name,
        "phrase": args.phrase,
        "plane": args.plane,
        "selection": sel.summary(),
        "before": {"phrase_logprob": before_score, "refusal_100": before_refusal},
        "results": {},
    }
    snapshot = WeightSnapshot(lm)
    for gamma in [float(x) for x in args.gammas.split(",")]:
        for lam in [float(x) for x in args.lambdas.split(",")]:
            a = target(base + lam * phrase_dir, dirs, "grad")
            info = apply_redirect(lm, sel, dirs, a, gamma, snapshot=snapshot)
            res = measure(lm, baseline, dirs[-1], f"lambda{lam:g}_g{gamma:g}",
                          keep_texts=True)
            texts = [r["response"] for r in res.pop("responses")]
            start_rate, any_rate = phrase_rates(texts, args.phrase)
            hold = refusal_rate(lm, bad_extended_prompts())
            after_score = mean_phrase_score(lm, phrase_prompts, args.phrase)
            key = f"lambda{lam:g}_g{gamma:g}"
            results["results"][key] = {
                "lambda": lam,
                "gamma": gamma,
                "edit": info,
                "phrase_logprob": after_score,
                "phrase_delta": after_score - before_score,
                "phrase_start_rate": start_rate,
                "phrase_anywhere_rate": any_rate,
                "refusal_100": res["refusal"],
                "heldout_300": hold,
                "kl_divergence": res["kl_divergence"],
                "perplexity_wikitext2": res["perplexity_wikitext2"],
            }
            print(f"[{time.time()-t0:6.1f}s] {key:18s} "
                  f"ref={res['refusal']['refusal_rate']:.3f} "
                  f"held={hold['refusal_rate']:.3f} "
                  f"kl={res['kl_divergence']:.4f} "
                  f"phrase={after_score:.2f} "
                  f"start={start_rate:.2f} any={any_rate:.2f}")
            snapshot.restore()
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
