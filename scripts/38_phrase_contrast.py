"""Phrase target with the refusal set suppressed.

A phrase-only target raises "I can do that." but leaves refusal at 0.55/0.50.
This script tests whether adding the refusal-set contrast recovers conversion:

    phrase                 grad log P(phrase)
    phrase_refuse          grad log P(phrase) - grad logsumexp(refuse set)
    phrase_accept          grad log P(phrase) + grad logsumexp(accept set)
                           - grad logsumexp(refuse set)
    accept                 grad logsumexp(accept set) - grad logsumexp(refuse set)

Each target is applied at gamma 0.3 and 0.75 on the same 35,043 atoms.

    PYTHONPATH=. .venv/bin/python scripts/38_phrase_contrast.py
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", default="I can do that.")
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--eval-prompts", type=int, default=100)
    ap.add_argument("--phrase-eval", type=int, default=50)
    ap.add_argument("--over", type=float, default=0.90)
    ap.add_argument("--plane", type=int, default=18)
    ap.add_argument("--gammas", default="0.3,0.75")
    ap.add_argument("--targets", default="phrase,phrase_refuse,phrase_accept,accept")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "redirect" / "phrase_contrast.json"))
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

    targets = {
        "phrase": phrase_grad,
        "phrase_refuse": phrase_grad - grad_refuse,
        "phrase_accept": phrase_grad + grad_accept - grad_refuse,
        "accept": grad_accept - grad_refuse,
    }
    targets = {k: target(v, dirs, "grad") for k, v in targets.items()}
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
    for name, a in targets.items():
        if name not in args.targets.split(","):
            continue
        for gamma in [float(x) for x in args.gammas.split(",")]:
            info = apply_redirect(lm, sel, dirs, a, gamma, snapshot=snapshot)
            res = measure(lm, baseline, dirs[-1], f"{name}_g{gamma}", keep_texts=False)
            hold = refusal_rate(lm, bad_extended_prompts())
            after_score = mean_phrase_score(lm, phrase_prompts, args.phrase)
            key = f"{name}_g{gamma:g}"
            results["results"][key] = {
                "target": name,
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
                results["results"][key]["samples"] = [
                    {"prompt": p, "response": t}
                    for p, t in zip(eval_prompts[: args.samples],
                                    lm.generate(eval_prompts[: args.samples], max_new_tokens=60))
                ]
            print(f"[{time.time()-t0:6.1f}s] {key:20s} "
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
