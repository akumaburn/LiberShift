"""Run a grid of method configurations and measure each one.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/02_run.py --fit-prompts 400 \
      --configs local_0.90_1,local_0.90_2,global_0.90_1,dense
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from libershift import collect, directions, load_model
from libershift.attribute import concentration
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_eval_prompts, bad_fit_prompts, good_eval_prompts, good_fit_prompts
from libershift.evaluate import logprobs, measure, save_result
from libershift.patch import write_patch
from libershift.pipeline import Config, build_directions, fused_selection, run


def parse_config(name: str) -> Config:
    if name == "dense_all":
        return Config(direction_scope="local", load_target=1.0, rounds=1, min_t=0.0,
                      include_negative=True)
    if name == "dense":
        return Config(direction_scope="local", load_target=1.0, rounds=1, min_t=0.0)
    scope, target, rounds = name.rsplit("_", 2)
    return Config(direction_scope=scope, load_target=float(target), rounds=int(rounds))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--configs", default="local_0.90_1")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--save-patch", action="store_true")
    args = ap.parse_args()

    out = OUT / (QWEN_1_5B.name + args.out_tag)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lm = load_model(QWEN_1_5B.path)
    good_prompts = good_fit_prompts()[: args.fit_prompts]
    bad_prompts = bad_fit_prompts()[: args.fit_prompts]
    good = collect(lm, good_prompts)
    bad = collect(lm, bad_prompts)
    print(f"[{time.time()-t0:6.1f}s] stats: {len(good_prompts)} prompts per class")

    dirs_local = directions(bad.h, good.h)
    torch.save({"directions": dirs_local, "stats_good": good, "stats_bad": bad}, out / "stats.pt")

    baseline_path = out / "baseline_logprobs.pt"
    if baseline_path.exists():
        baseline = torch.load(baseline_path, weights_only=True)
    else:
        baseline = logprobs(lm, good_eval_prompts())
        torch.save(baseline, baseline_path)

    readout = dirs_local[-1]
    base = measure(lm, baseline, readout, "baseline", keep_texts=True)
    save_result(out / "baseline.json", base)
    print(f"[{time.time()-t0:6.1f}s] baseline refusal={base['refusal']['refusal_rate']:.2f} "
          f"kl={base['kl_divergence']:.4f} ppl={base['perplexity_wikitext2']:.2f}")

    results = {"baseline": {k: v for k, v in base.items() if k != "responses"},
               "configs": {}}
    for name in args.configs.split(","):
        cfg = parse_config(name)
        dirs = build_directions(good, bad, cfg.direction_scope)
        runner, report = run(lm, good_prompts, bad_prompts, cfg, direction=dirs, stats=(good, bad))
        res = measure(lm, baseline, readout, name, keep_texts=True)
        res["method"] = report.summary()
        save_result(out / f"{name}.json", res)
        results["configs"][name] = {k: v for k, v in res.items() if k != "responses"}
        print(f"[{time.time()-t0:6.1f}s] {name:18s} refusal={res['refusal']['refusal_rate']:.2f} "
              f"kl={res['kl_divergence']:.4f} ppl={res['perplexity_wikitext2']:.1f} "
              f"atoms={len(report.atoms)} rel_dw={_rel(report):.4f} "
              f"captured={_captured(report):.3f}")
        if args.save_patch:
            write_patch(out / f"patch_{name}.json", lm, report.atoms, dirs,
                        {"model": QWEN_1_5B.name, "method": name, "selection": report.summary()})
        runner.reset()
        torch.cuda.empty_cache()

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[{time.time()-t0:6.1f}s] wrote {out / 'results.json'}")


def _rel(report) -> float:
    return sum(r.edit["relative_change"] for r in report.rounds)


def _captured(report) -> float:
    before = sum(r.load_before for r in report.rounds)
    got = sum(r.selection["captured_load"] for r in report.rounds)
    return got / before if before else 0.0


if __name__ == "__main__":
    sys.exit(main())
