"""Redirect refusal writers to a compliant teacher's residual shift.

The teacher is the exported dense-removal checkpoint.  Its residual shift on
harmful prompts is the direction that a working compliance edit produces.  This
script rotates the base model's refusal writers toward that shift, instead of
toward the negative refusal direction, and compares the two targets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from libershift import collect, directions, load_model
from libershift.attribute import attribute
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_fit_prompts, good_fit_prompts
from libershift.edit import Editor, select
from libershift.evaluate import measure, save_result


def redirect(lm, sel, r, a, gamma: float, renorm: bool = True) -> dict:
    r = r.float().to(lm.device)
    a = a.float().to(lm.device)
    by_layer: dict[int, list] = {}
    for atom in sel.atoms:
        by_layer.setdefault(atom.layer, []).append(atom)
    changed = 0.0
    for layer, atoms in sorted(by_layer.items()):
        feats = [x for x in atoms if x.kind == "feature"]
        heads = [x for x in atoms if x.kind == "head"]
        if feats:
            module = lm.ffn_down(layer)
            W = module.weight.data.float()
            idx = torch.tensor([x.index for x in feats], device=W.device)
            cols = W[:, idx]
            alpha = cols.T @ r[layer + 1]
            new = cols - torch.outer(r[layer + 1], alpha) + gamma * torch.outer(a[layer + 1], alpha)
            if renorm:
                scale = cols.norm(dim=0) / new.norm(dim=0).clamp_min(1e-12)
                new = new * scale
            W[:, idx] = new
            module.weight.data.copy_(W.to(module.weight.dtype))
            changed += float((new - cols).pow(2).sum())
        if heads:
            module = lm.attn_out(layer)
            W = module.weight.data.float()
            hd = lm.head_dim
            for atom in heads:
                block = W[:, atom.index * hd:(atom.index + 1) * hd]
                beta = block.T @ r[layer + 1]
                new = block - torch.outer(r[layer + 1], beta) + gamma * torch.outer(a[layer + 1], beta)
                if renorm:
                    scale = block.norm() / new.norm().clamp_min(1e-12)
                    new = new * scale
                W[:, atom.index * hd:(atom.index + 1) * hd] = new
                changed += float((new - block).pow(2).sum())
            module.weight.data.copy_(W.to(module.weight.dtype))
    return {"changed_sq_norm": changed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--gammas", default="0.5,1.0,1.5")
    ap.add_argument("--out", default=str(OUT / "teacher_redirect.json"))
    args = ap.parse_args()

    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[:400])
    bad = collect(lm, bad_fit_prompts()[:400])
    dirs = directions(bad.h, good.h)
    dirs = dirs[18].unsqueeze(0).repeat(dirs.shape[0], 1)
    attr = attribute(lm, good, bad, dirs)
    baseline = torch.load(args.baseline, weights_only=True)

    teacher = load_model(args.teacher)
    t_bad = collect(teacher, bad_fit_prompts()[:400])
    shift = t_bad.h - bad.h
    a = F.normalize(shift.float(), p=2, dim=1)
    cos = F.cosine_similarity(a, -dirs, dim=1)
    print(f"[{time.time()-t0:6.1f}s] teacher shift ready; cos(shift, -r) by plane: "
          f"{[round(float(x), 3) for x in cos[14:29]]}")
    del teacher
    torch.cuda.empty_cache()

    sel = select(attr, ("feature", "head"), load_target=args.target, min_t=2.0)
    print(f"selected {len(sel.atoms)} atoms, captured {sel.captured_fraction:.3f}")

    results = {}
    editor = Editor(lm)
    editor.set_directions(dirs)
    for gamma in [float(x) for x in args.gammas.split(",")]:
        info = redirect(lm, sel, dirs, a, gamma)
        res = measure(lm, baseline, dirs[-1], f"teacher_g{gamma}", keep_texts=False)
        results[f"g{gamma}"] = {
            "gamma": gamma,
            "refusal": res["refusal"],
            "kl_divergence": res["kl_divergence"],
            "perplexity_wikitext2": res["perplexity_wikitext2"],
            "edit": info,
        }
        print(f"[{time.time()-t0:6.1f}s] gamma {gamma:4.2f} refusal="
              f"{res['refusal']['refusal_rate']:.3f} kl={res['kl_divergence']:.4f} "
              f"ppl={res['perplexity_wikitext2']:.2f}")
        editor.reset()
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
