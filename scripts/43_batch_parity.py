"""Evaluate every comparison checkpoint at two batch sizes.

The frozen study used 32 prompts per forward pass.  Heretic auto-selects the
largest batch that fits, up to 128, so its published numbers were measured at a
different batch size than ours.  This script re-evaluates the base model, the
dense removal edit, and both Heretic modes at 32 and 128 in the same harness.

Two things come out of it:

* the batch-size cost, so the speed-up is measured rather than assumed;
* a parity table that shows whether the protocol change moves any number.  If
  Heretic's two modes reproduce their published refusal rates here, the harness
  agrees with Heretic and the batch size does not decide the comparison.

    PYTHONPATH=. .venv/bin/python scripts/43_batch_parity.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from libershift import load_model
from libershift.constants import MODELS, OUT, QWEN_0_5B, QWEN_1_5B
from libershift.data import bad_extended_prompts, good_eval_prompts
from libershift.evaluate import logprobs, measure, refusal_rate

# Heretic's exported modes live beside the checkpoints this study loaded,
# under freedom/out.  ``models`` is a symlink to ``freedom/models``.
FREEDOM = MODELS.resolve().parent


def checkpoints(which: str) -> dict[str, Path]:
    other = FREEDOM / "out"
    if which == "1.5b":
        return {
            "base": QWEN_1_5B.path,
            "dense_removal": other / "export" / "taa_p18",
            "heretic_mpoa": other / "heretic_mpoa" / "model",
            "heretic_ara": other / "heretic_ara" / "model",
        }
    return {
        "base": QWEN_0_5B.path,
        "dense_removal": other / "export" / "Qwen2.5-0.5B-Instruct_taa",
        "heretic_mpoa": other / "heretic_mpoa_05b" / "model",
        "heretic_ara": other / "heretic_ara_05b" / "model",
    }


def scores(lm, baseline) -> dict:
    start = time.perf_counter()
    measured = measure(lm, baseline, torch.zeros(lm.hidden_size), "parity")
    seconds = time.perf_counter() - start
    hold = refusal_rate(lm, bad_extended_prompts())
    return {
        "refusal_100": measured["refusal"]["refusal_rate"],
        "opening_refusal_100": measured["refusal"]["opening_refusal_rate"],
        "heldout_300": hold["refusal_rate"],
        "heldout_opening_300": hold["opening_refusal_rate"],
        "kl_divergence": measured["kl_divergence"],
        "perplexity_wikitext2": measured["perplexity_wikitext2"],
        "measure_seconds": seconds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b", choices=("1.5b", "0.5b"))
    ap.add_argument("--batches", default="32,128")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths = checkpoints(args.model)
    out: dict = {
        "model": args.model,
        "checkpoints": {k: str(v.resolve()) for k, v in paths.items()},
        "results": {},
    }

    for batch in [int(x) for x in args.batches.split(",")]:
        baseline = None
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(path)
            lm = load_model(path)
            lm.batch_size = batch
            if baseline is None:
                baseline = logprobs(lm, good_eval_prompts())
            res = scores(lm, baseline)
            del lm
            torch.cuda.empty_cache()
            out["results"].setdefault(name, {})[str(batch)] = res
            print(
                f"{time.strftime('%H:%M:%S')} batch {batch:3d} {name:14s} "
                f"ref={res['refusal_100']:.3f} open={res['opening_refusal_100']:.3f} "
                f"held={res['heldout_300']:.3f}/{res['heldout_opening_300']:.3f} "
                f"kl={res['kl_divergence']:.4f} ppl={res['perplexity_wikitext2']:.2f} "
                f"measure={res['measure_seconds']:.1f}s"
            )

    target = Path(args.out) if args.out else OUT / f"batch_parity_{args.model}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2))
    print("wrote", target)


if __name__ == "__main__":
    sys.exit(main())
