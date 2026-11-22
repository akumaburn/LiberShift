"""Export a method variant as a standalone HF checkpoint plus a patch.

The patch is the point: the edit is a list of atom indices and their new
vectors, so it can be inspected and re-applied without re-running attribution.
`--verify-patch` re-applies the patch to a pristine checkpoint and checks that
the exported model and the patched model produce identical logits.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from libershift import attribute, collect, directions, load_model
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_fit_prompts, good_fit_prompts, good_eval_prompts
from libershift.edit import Editor
from libershift.evaluate import logprobs
from libershift.patch import apply_patch, write_patch
from libershift.pipeline import atoms_for_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--layers", default="all")  # "all" or "top:N"
    ap.add_argument("--kinds", default="feature,head")
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--verify-patch", action="store_true")
    ap.add_argument("--direction", default="local",
                    help="local | global (last plane) | plane:N (one plane for every layer)")
    args = ap.parse_args()

    out = OUT / "export"
    out.mkdir(parents=True, exist_ok=True)
    target = out / args.name

    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    if args.direction == "global":
        dirs = dirs[-1].unsqueeze(0).repeat(dirs.shape[0], 1)
    elif args.direction.startswith("plane:"):
        plane = int(args.direction.split(":")[1])
        dirs = dirs[plane].unsqueeze(0).repeat(dirs.shape[0], 1)
    attr = attribute(lm, good, bad, dirs)

    if args.layers == "all":
        layers = list(range(lm.n_layers))
    elif args.layers.startswith("top:"):
        k = int(args.layers.split(":")[1])
        per_layer = (attr.feature_load.sum(1) + attr.head_load.sum(1)).tolist()
        layers = sorted(sorted(range(lm.n_layers), key=lambda l: per_layer[l], reverse=True)[:k])
    else:
        layers = [int(x) for x in args.layers.split(",")]
    kinds = tuple(args.kinds.split(","))

    atoms = atoms_for_layers(attr, kinds, layers)
    editor = Editor(lm)
    editor.set_directions(dirs)
    info = editor.apply(atoms, renorm=True)
    print(f"edited {len(atoms)} atoms over {len(layers)} layers, relative weight change "
          f"{info['relative_change']:.4f}")

    target.mkdir(parents=True, exist_ok=True)
    lm.model.save_pretrained(target, safe_serialization=True)
    lm.tokenizer.save_pretrained(target)
    print(f"saved {target}")

    payload = write_patch(out / f"{args.name}.json", editor.ops,
                          {"model": QWEN_1_5B.name, "method": args.name,
                           "selection": {"layers": layers, "kinds": list(kinds),
                                         "atoms": len(atoms), "edit": info}})
    print(f"patch: {payload['layer_count']} layers, {payload['representation']} form, "
          f"{payload['array_bytes']} bytes, sha256 {payload['array_sha256'][:16]}")

    if args.verify_patch:
        patched_dir = out / f"{args.name}_from_patch"
        result = apply_patch(QWEN_1_5B.path, out / f"{args.name}.json", patched_dir)
        print(f"re-applied patch: {result}")
        a = load_model(target)
        b = load_model(patched_dir)
        prompts = good_eval_prompts()[:16]
        la = a.first_token_logits(prompts)
        lb = b.first_token_logits(prompts)
        print(f"max |logit difference| between exported and patched model: "
              f"{float((la - lb).abs().max()):.3e}")
        (out / f"{args.name}_verify.json").write_text(json.dumps({
            "apply": result,
            "max_logit_difference": float((la - lb).abs().max()),
            "patch_bytes": payload["array_bytes"],
            "representation": payload["representation"],
        }, indent=2))


if __name__ == "__main__":
    sys.exit(main())
