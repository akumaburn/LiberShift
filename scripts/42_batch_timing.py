"""Measure the evaluation cost and whether a larger batch preserves results.

Every reported number comes from ``measure``: 100 greedy generations of 100
tokens, 100 first-token logits, one wikitext-2 perplexity, and two projection
passes.  If a larger batch gives the same refusal count, KL and perplexity at a
lower wall clock, the whole study gets that much cheaper.

    PYTHONPATH=. .venv/bin/python scripts/42_batch_timing.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from libershift import load_model
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_eval_prompts, good_eval_prompts
from libershift.evaluate import logprobs, measure, refusal_rate


def timed(fn, reps: int):
    times = []
    result = None
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return result, times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--batches", default="32,64,96,128")
    ap.add_argument("--out", default=str(OUT / "batch_timing.json"))
    args = ap.parse_args()

    lm = load_model(QWEN_1_5B.path)
    bad = bad_eval_prompts()
    good = good_eval_prompts()
    baseline = logprobs(lm, good)
    out = {"model": QWEN_1_5B.name, "batches": {}}

    for batch in [int(x) for x in args.batches.split(",")]:
        lm.batch_size = batch
        _, gen_times = timed(lambda: refusal_rate(lm, bad), args.reps)
        _, lp_times = timed(lambda: logprobs(lm, good), args.reps)
        res, full_times = timed(
            lambda: measure(lm, baseline, torch.zeros(lm.hidden_size), "probe"),
            args.reps,
        )
        out["batches"][str(batch)] = {
            "refusal_seconds": gen_times,
            "logits_seconds": lp_times,
            "measure_seconds": full_times,
            "measure_median": statistics.median(full_times),
            "refusal_rate": res["refusal"]["refusal_rate"],
            "kl_divergence": res["kl_divergence"],
            "perplexity_wikitext2": res["perplexity_wikitext2"],
        }
        print(
            f"batch {batch:4d}  gen {statistics.median(gen_times):6.2f}s  "
            f"logits {statistics.median(lp_times):5.2f}s  "
            f"measure {statistics.median(full_times):6.2f}s  "
            f"ref={res['refusal']['refusal_rate']:.3f} "
            f"kl={res['kl_divergence']:.4f} ppl={res['perplexity_wikitext2']:.2f}"
        )
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
