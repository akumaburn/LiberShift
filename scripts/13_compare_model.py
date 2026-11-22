"""Cross-model comparison table from the saved artifacts.

Every model is measured by the same code: Heretic's refusal markers, Heretic's
first-token KL against the base checkpoint, a held-out perplexity, the 300
held-out harmful prompts, and the same capability tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from libershift.constants import OUT, ROOT, spec as get_spec, taa_export
from libershift.evaluate import logprobs, measure, save_result
from libershift.model import load_model


def paths_for(spec_name: str) -> dict[str, str]:
    s = get_spec(spec_name)
    tag = "_05b" if "0.5B" in spec_name else "_35b" if "3.5" in spec_name else ""
    return {
        "base": str(s.path),
        "taa": str(taa_export(s.name)),
        "heretic_mpoa": str(OUT / f"heretic_mpoa{tag}" / "model"),
        "heretic_ara": str(OUT / f"heretic_ara{tag}" / "model"),
    }


def baseline_dir(spec_name: str) -> str:
    name = get_spec(spec_name).name
    for candidate in (OUT / f"{name}_main", OUT / name):
        if (candidate / "baseline_logprobs.pt").exists():
            return str(candidate)
    # The Qwen3.5-4B study keeps its fit statistics in cache/qwen35_stats.pt
    # (keys good/bad/dirs instead of one stats.pt); the baseline logprobs are
    # computed once from the base checkpoint and saved beside it.
    if (ROOT / "cache" / "qwen35_stats.pt").exists():
        out_dir = OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (out_dir / "baseline_logprobs.pt").exists():
            from libershift.data import good_eval_prompts

            lm = load_model(get_spec(spec_name).path)
            torch.save(logprobs(lm, good_eval_prompts()),
                       out_dir / "baseline_logprobs.pt")
            del lm
            torch.cuda.empty_cache()
        return str(out_dir)
    raise FileNotFoundError(f"no stats for {spec_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--models", default="")
    ap.add_argument("--extended", action="store_true", default=True)
    args = ap.parse_args()

    s = get_spec(args.model)
    d = baseline_dir(args.model)
    if (Path(d) / "stats.pt").exists():
        stats = torch.load(f"{d}/stats.pt", weights_only=False)
    else:
        stats = torch.load(ROOT / "cache" / "qwen35_stats.pt", weights_only=False)
    ref = torch.load(f"{d}/baseline_logprobs.pt", weights_only=True)
    readout = (stats["directions"] if "directions" in stats else stats["dirs"])[-1]

    from libershift.data import bad_extended_prompts
    from libershift.evaluate import refusal_rate

    paths = paths_for(args.model)
    names = args.models.split(",") if args.models else list(paths)
    results = {}
    for name in names:
        lm = load_model(paths[name])
        res = measure(lm, ref, readout, name, keep_texts=True)
        if name == "base":
            res["kl_sanity"] = abs(res["kl_divergence"]) < 1e-6
        if args.extended:
            ext = refusal_rate(lm, bad_extended_prompts())
            res["extended_refusal"] = {k: v for k, v in ext.items() if k != "responses"}
        results[name] = res
        save_result(OUT / f"eval_{s.name}_{name}.json", res)
        print(f"{name:14s} report={res['refusal']['refusal_rate']:.3f} "
              f"opening={res['refusal']['opening_refusal_rate']:.3f} "
              f"extended={res.get('extended_refusal', {}).get('refusal_rate', float('nan')):.3f} "
              f"kl={res['kl_divergence']:.4f} ppl={res['perplexity_wikitext2']:.2f} "
              f"chars={res['refusal']['mean_response_chars']:.0f}")
        del lm
        torch.cuda.empty_cache()

    (OUT / f"comparison_{s.name}.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT / f'comparison_{s.name}.json'}")


if __name__ == "__main__":
    sys.exit(main())
