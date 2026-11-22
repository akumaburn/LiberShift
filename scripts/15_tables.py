"""Build the reported tables from stored generations, on disjoint prompt splits.

`measure` evaluates the full 100-prompt harmful slice.  Thirty of those prompts
were used to calibrate the edit plane, so they cannot appear in the reported
number.  This script recomputes every rate from the stored per-prompt responses
using prompts 30..99, and pairs it with the 300 held-out prompts that no method
saw during calibration or evaluation.

It also prints the markdown rows used in RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import sys

from libershift.constants import OUT, spec as get_spec
from libershift.evaluate import opening_refusal


def slice_stats(responses: list[dict], start: int) -> dict:
    rows = responses[start:]
    n = len(rows)
    return {
        "n": n,
        "refusals": sum(1 for r in rows if r["refusal"]),
        "refusal_rate": sum(1 for r in rows if r["refusal"]) / n,
        "opening_refusals": sum(1 for r in rows if opening_refusal(r["response"])),
        "opening_rate": sum(1 for r in rows if opening_refusal(r["response"])) / n,
        "mean_chars": sum(len(r["response"].strip()) for r in rows) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--calibration", type=int, default=30)
    args = ap.parse_args()

    s = get_spec(args.model)
    tag = "_05b" if "0.5B" in s.name else None
    names = ["base", "taa", "heretic_mpoa", "heretic_ara"]
    tables = {}
    print(f"model: {s.name}\n")
    print("| model | refusal (70) | refusal (300 held out) | opening (70) | KL | PPL |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name in names:
        path = OUT / f"eval_{s.name}_{name}.json"
        data = json.loads(path.read_text())
        stats = slice_stats(data["responses"], args.calibration)
        ext = data.get("extended_refusal", {})
        tables[name] = {
            "report": stats,
            "extended": ext,
            "kl_divergence": data["kl_divergence"],
            "perplexity": data["perplexity_wikitext2"],
        }
        label = {"base": "base", "taa": "**LiberShift**",
                 "heretic_mpoa": "Heretic MPOA, 40 trials",
                 "heretic_ara": "Heretic ARA, 40 trials"}[name]
        print(f"| {label} | {stats['refusal_rate']:.2f} ({stats['refusals']}/{stats['n']}) "
              f"| {ext.get('refusal_rate', float('nan')):.2f} "
              f"({ext.get('refusals', 0)}/{ext.get('n', 0)}) "
              f"| {stats['opening_rate']:.2f} | {data['kl_divergence']:.4f} "
              f"| {data['perplexity_wikitext2']:.2f} |")

    out = OUT / f"tables_{tag or 'main'}.json"
    out.write_text(json.dumps(tables, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
