"""Replay a rank-1 or operator patch against a pristine checkpoint.

``apply_patch`` rebuilds the edited model from the base checkpoint and the
patch. This script then loads the exported model and the rebuilt one and
compares their first-token logits, so the "zero logit difference" claim has a
file behind it for any model.

    PYTHONPATH=. .venv/bin/python scripts/31_verify_patch.py \
        --model models/Qwen2.5-0.5B-Instruct \
        --patch out/export/Qwen2.5-0.5B-Instruct_taa.json \
        --exported out/export/Qwen2.5-0.5B-Instruct_taa \
        --out out/patch_verify_05b.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from libershift.constants import OUT
from libershift.data import good_eval_prompts
from libershift.model import load_model
from libershift.patch import apply_patch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pristine base checkpoint")
    ap.add_argument("--patch", required=True, help="patch JSON sidecar")
    ap.add_argument("--exported", required=True, help="model exported by the edit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=16)
    args = ap.parse_args()

    rebuilt = OUT / "patch_replay_tmp"
    if rebuilt.exists():
        shutil.rmtree(rebuilt)
    result = apply_patch(args.model, Path(args.patch), rebuilt)
    print("re-applied patch:", result)

    exported = load_model(args.exported)
    patched = load_model(rebuilt)
    prompts = good_eval_prompts()[: args.prompts]
    a = exported.first_token_logits(prompts)
    b = patched.first_token_logits(prompts)
    difference = float((a - b).abs().max())

    payload = {
        "apply": result,
        "max_logit_difference": difference,
        "patch_bytes": result["patch_bytes"],
        "representation": result["representation"],
        "prompts": len(prompts),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"max |logit difference| over {len(prompts)} prompts: {difference:.3e}")
    print("wrote", args.out)
    shutil.rmtree(rebuilt, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
