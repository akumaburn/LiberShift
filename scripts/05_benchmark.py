"""Capability check with lm-eval, identical tasks and limits for every model.

KL divergence on first-token distributions is sensitive but indirect; a small
accuracy suite catches damage that KL misses (and vice versa).  Every model in
the comparison runs the same tasks with the same seed and sample limit.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from libershift.constants import OUT

TASKS = ["arc_easy", "piqa", "winogrande"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tokenizer = AutoTokenizer.from_pretrained(args.path)
    config = AutoConfig.from_pretrained(args.path)
    architectures = " ".join(getattr(config, "architectures", None) or [])
    loader = (AutoModelForImageTextToText if "ConditionalGeneration" in architectures
              else AutoModelForCausalLM)
    model = loader.from_pretrained(args.path, dtype=torch.bfloat16, device_map="cuda")
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=16)

    results = {}
    for task in TASKS:
        out = lm_eval.simple_evaluate(model=hflm, tasks=[task], limit=args.limit)
        metrics = dict(out["results"][task])
        for boilerplate in ("name", "alias"):
            metrics.pop(boilerplate, None)
        results[task] = metrics
        print(f"{args.name} {task}: {metrics}")

    path = OUT / f"bench_{args.name}.json"
    path.write_text(json.dumps({"model": args.name, "path": args.path,
                                "limit": args.limit, "tasks": results}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
