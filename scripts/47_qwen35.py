"""Qwen3.5-4B: the model Heretic's ARA PR (#211) reports as resistant to
classic directional ablation -- more than 50 of 100 harmful prompts still
refused after abliteration, against 4 of 100 for ARA at 0.1396 KL.

Qwen3.5-4B is a hybrid decoder: 24 of 32 layers are linear attention
(``linear_attn.out_proj``), 8 are full attention (``self_attn.o_proj``), and
the two families do not share a head layout.  The harness therefore treats one
attention-output input channel as one atom (``attention_atoms = "columns"``).

Stages are run separately because each stage loads the model fresh; the fit
statistics are cached in ``cache/`` across stages.

  PYTHONPATH=. .venv/bin/python scripts/47_qwen35.py --stage base
  PYTHONPATH=. .venv/bin/python scripts/47_qwen35.py --stage calibrate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from libershift import collect, directions, load_model
from libershift.attribute import attribute
from libershift.constants import CACHE, OUT, QWEN_3_5_4B
from libershift.data import (
    bad_eval_prompts,
    bad_extended_prompts,
    bad_fit_prompts,
    good_eval_prompts,
    good_fit_prompts,
)
from libershift.edit import select_all
from libershift.evaluate import logprobs, measure, refusal_rate, save_result
from libershift.redirect import HybridEdit

STATS = CACHE / "qwen35_stats.pt"
PLANES = (16, 18, 20, 22, 24)
# Provenance for the docs: Heretic PR #211 discussion (GhostWithAHat, 2026),
# measured with Heretic's own default protocol on Qwen/Qwen3.5-4B.
HERETIC_PR = {
    "source": "https://github.com/p-e-w/heretic/pull/211",
    "claim": "Qwen3.5-4B still refuses >50/100 after directional ablation (MPOA); "
             "ARA reaches 4/100 at 0.1396 KL",
}


def fit(lm) -> tuple:
    """Per-plane refusal directions and prompt statistics, cached on disk."""
    if STATS.exists():
        d = torch.load(STATS, weights_only=False)
        return d["good"], d["bad"], d["dirs"]
    good = collect(lm, good_fit_prompts()[:400])
    bad = collect(lm, bad_fit_prompts()[:400])
    dirs = directions(bad.h, good.h)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"good": good, "bad": bad, "dirs": dirs}, STATS)
    return good, bad, dirs


def plane_dirs(dirs: torch.Tensor, plane: int) -> torch.Tensor:
    return dirs[plane].unsqueeze(0).repeat(dirs.shape[0], 1)


def stage_base() -> None:
    t0 = time.time()
    lm = load_model(QWEN_3_5_4B.path)
    good, bad, dirs = fit(lm)
    print(f"[{time.time()-t0:6.1f}s] stats cached; {atoms_of(lm)} edit columns per layer")
    baseline = logprobs(lm, good_eval_prompts())
    res = measure(lm, baseline, dirs[-1], "base", keep_texts=True)
    hold = refusal_rate(lm, bad_extended_prompts())
    from collections import Counter

    import json as _json

    config_path = Path(QWEN_3_5_4B.path) / "config.json"
    layer_types = dict(Counter(_json.loads(config_path.read_text())["text_config"]["layer_types"]))
    payload = {
        "model": QWEN_3_5_4B.name,
        "n_layers": lm.n_layers,
        "hidden_size": lm.hidden_size,
        "attention_atoms": lm.attention_atoms,
        "atoms_per_layer": lm.n_heads,
        "layer_types": layer_types,
        "base": {k: v for k, v in res.items() if k != "responses"},
        "base_heldout_300": hold,
        "heretic_pr": HERETIC_PR,
    }
    save_result(OUT / "qwen35_base.json", payload)
    print(f"[{time.time()-t0:6.1f}s] base refusal={res['refusal']['refusal_rate']:.2f} "
          f"heldout={hold['refusal_rate']:.2f} ppl={res['perplexity_wikitext2']:.2f}")


def stage_calibrate() -> None:
    t0 = time.time()
    lm = load_model(QWEN_3_5_4B.path)
    good, bad, dirs = fit(lm)
    slice70 = bad_eval_prompts()[:70]
    results = {}
    editor = HybridEdit(lm)
    for plane in PLANES:
        dirs_p = plane_dirs(dirs, plane)
        attr = attribute(lm, good, bad, dirs_p)
        atoms = select_all(attr, only_positive=False).atoms
        editor.apply(atoms, [], dirs_p, dirs_p, 0.0)
        rate = refusal_rate(lm, slice70)
        results[str(plane)] = rate
        print(f"[{time.time()-t0:6.1f}s] plane {plane}: dense removal refusal "
              f"{rate['refusal_rate']:.3f}")
        editor.restore()
    chosen = min(results, key=lambda p: results[p]["refusal_rate"])
    payload = {
        "model": QWEN_3_5_4B.name,
        "protocol": "dense removal (every atom, gamma=0), refusal on the 70-prompt slice",
        "planes": results,
        "chosen_plane": int(chosen),
    }
    save_result(OUT / "qwen35_calibration.json", payload)
    print(f"[{time.time()-t0:6.1f}s] chosen plane {chosen}; wrote qwen35_calibration.json")


def atoms_of(lm) -> int:
    return lm.ffn_down(0).weight.shape[1] + lm.attn_out(0).weight.shape[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("base", "calibrate"))
    args = ap.parse_args()
    {"base": stage_base, "calibrate": stage_calibrate}[args.stage]()


if __name__ == "__main__":
    sys.exit(main())
