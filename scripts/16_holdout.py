"""Refusal on 300 held-out harmful prompts from two unseen families.

JailbreakBench (HarmBench-derived goals) and the unsafe prompts of BeaverTails.
Neither was used for the direction, the calibration, or the report split.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from libershift.constants import OUT, spec as get_spec, taa_export
from libershift.data import bad_extended_prompts
from libershift.evaluate import refusal_rate, save_result
from libershift.model import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--names", default="base,taa,heretic_mpoa,heretic_ara")
    args = ap.parse_args()

    s = get_spec(args.model)
    tag = "_05b" if "0.5B" in s.name else "_35b" if "3.5" in s.name else ""
    paths = {
        "base": str(s.path),
        "taa": str(taa_export(s.name)),
        "heretic_mpoa": str(OUT / f"heretic_mpoa{tag}" / "model"),
        "heretic_ara": str(OUT / f"heretic_ara{tag}" / "model"),
    }
    prompts = bad_extended_prompts()
    results = {}
    for name in args.names.split(","):
        lm = load_model(paths[name])
        res = refusal_rate(lm, prompts)
        results[name] = res
        save_result(OUT / f"holdout_{s.name}_{name}.json", res)
        print(f"{name:14s} holdout refusal={res['refusal_rate']:.3f} "
              f"({res['refusals']}/{res['n']}) opening={res['opening_refusal_rate']:.3f} "
              f"chars={res['mean_response_chars']:.0f}")
        del lm
        torch.cuda.empty_cache()

    (OUT / f"holdout_{s.name}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    sys.exit(main())
