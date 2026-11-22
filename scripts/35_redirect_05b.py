"""The redirect and the hybrid on Qwen2.5-0.5B-Instruct.

The 1.5B study is the only model with redirect numbers. This runs the same
pipeline on the 0.5B model, at the plane the 0.5B calibration chose (14), and
adds the efficiency-ranked hybrid at the two operating points found on 1.5B.

    PYTHONPATH=. .venv/bin/python scripts/35_redirect_05b.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from libershift import (
    ACCEPT_WORDS,
    REFUSE_WORDS,
    acceptance_gradient,
    collect,
    directions,
    load_model,
    target,
    token_ids,
)
from libershift.attribute import attribute
from libershift.constants import OUT, spec
from libershift.data import (
    bad_extended_prompts,
    bad_fit_prompts,
    good_eval_prompts,
    good_fit_prompts,
)
from libershift.edit import RENORM_GUARD, select
from libershift.evaluate import logprobs, measure, refusal_rate


class HybridEditor:
    """Removal for one atom set, redirect for a disjoint set."""

    def __init__(self, lm) -> None:
        self.lm = lm
        self.saved: dict[tuple[str, int], torch.Tensor] = {}

    def _keep(self, kind: str, layer: int, module) -> None:
        key = (kind, layer)
        if key not in self.saved:
            self.saved[key] = module.weight.detach().clone()

    def restore(self) -> None:
        for (kind, layer), weight in self.saved.items():
            module = (self.lm.ffn_down(layer) if kind == "feature"
                      else self.lm.attn_out(layer))
            module.weight.data.copy_(
                weight.to(module.weight.device, module.weight.dtype)
            )
        self.saved.clear()

    @staticmethod
    def _renorm(new: torch.Tensor, old: torch.Tensor, dim: int) -> torch.Tensor:
        n0 = old.norm(dim=dim)
        n1 = new.norm(dim=dim).clamp_min(1e-12)
        guard = n1 < RENORM_GUARD * n0
        scale = torch.where(guard, torch.ones_like(n1), n0 / n1)
        return new * (scale.unsqueeze(0) if dim == 0 else scale)

    def apply(self, removal, redirect, r, a, gamma: float) -> dict:
        rem_feat: dict[int, list] = {}
        red_feat: dict[int, list] = {}
        rem_head: dict[int, list] = {}
        red_head: dict[int, list] = {}
        for atom in removal:
            bucket = rem_feat if atom.kind == "feature" else rem_head
            bucket.setdefault(atom.layer, []).append(atom.index)
        for atom in redirect:
            bucket = red_feat if atom.kind == "feature" else red_head
            bucket.setdefault(atom.layer, []).append(atom.index)

        layers = sorted(set(rem_feat) | set(red_feat) | set(rem_head) | set(red_head))
        denominator = 0.0
        for layer in layers:
            if layer in rem_feat or layer in red_feat:
                denominator += float(
                    self.lm.ffn_down(layer).weight.detach().float().pow(2).sum()
                )
            if layer in rem_head or layer in red_head:
                denominator += float(
                    self.lm.attn_out(layer).weight.detach().float().pow(2).sum()
                )

        changed = 0.0
        for layer in layers:
            rp = r[layer + 1].float().to(self.lm.device)
            ap = a[layer + 1].float().to(self.lm.device)
            if layer in rem_feat or layer in red_feat:
                module = self.lm.ffn_down(layer)
                self._keep("feature", layer, module)
                W = module.weight.data.float()
                for indices, mode in ((rem_feat.get(layer, []), "removal"),
                                      (red_feat.get(layer, []), "redirect")):
                    if not indices:
                        continue
                    idx = torch.tensor(indices, device=W.device)
                    cols = W[:, idx]
                    alpha = cols.T @ rp
                    new = cols - torch.outer(rp, alpha)
                    if mode == "redirect":
                        new = new + gamma * torch.outer(ap, alpha)
                    new = self._renorm(new, cols, dim=0)
                    W[:, idx] = new
                    changed += float((new - cols).pow(2).sum())
                module.weight.data.copy_(W.to(module.weight.dtype))
            if layer in rem_head or layer in red_head:
                module = self.lm.attn_out(layer)
                self._keep("head", layer, module)
                W = module.weight.data.float()
                hd = self.lm.head_dim
                for indices, mode in ((rem_head.get(layer, []), "removal"),
                                      (red_head.get(layer, []), "redirect")):
                    for head in indices:
                        start = head * hd
                        block = W[:, start:start + hd]
                        beta = block.T @ rp
                        new = block - torch.outer(rp, beta)
                        if mode == "redirect":
                            new = new + gamma * torch.outer(ap, beta)
                        new = self._renorm(new, block, dim=None)
                        W[:, start:start + hd] = new
                        changed += float((new - block).pow(2).sum())
                module.weight.data.copy_(W.to(module.weight.dtype))
        torch.cuda.empty_cache()
        return {
            "changed_sq_norm": changed,
            "edited_family_sq_norm": denominator,
            "relative_change": changed / denominator if denominator else 0.0,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-0.5B-Instruct")
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--over", type=float, default=0.90)
    ap.add_argument("--plane", type=int, default=14)
    ap.add_argument("--gammas", default="0.3,0.75")
    ap.add_argument("--out", default=str(OUT / "redirect" / "redirect_05b.json"))
    args = ap.parse_args()

    ms = spec(args.model)
    t0 = time.time()
    lm = load_model(ms.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    dirs = dirs[args.plane].unsqueeze(0).repeat(dirs.shape[0], 1)
    attr = attribute(lm, good, bad, dirs)
    sel = select(attr, ("feature", "head"), load_target=args.over, min_t=2.0)
    baseline = logprobs(lm, good_eval_prompts())
    print(f"[{time.time()-t0:6.1f}s] {ms.name}: {len(sel.atoms)} atoms at plane {args.plane}")

    accept = token_ids(lm.tokenizer, ACCEPT_WORDS)
    refuse = token_ids(lm.tokenizer, REFUSE_WORDS)
    grads, _ = acceptance_gradient(
        lm, bad_fit_prompts()[: args.fit_prompts], accept, refuse
    )
    a = target(grads, dirs, "grad")

    n_heads, head_dim = lm.n_heads, lm.head_dim
    proxy: dict[tuple[int, str, int], float] = {}
    for atom in sel.atoms:
        if atom.kind == "feature":
            alpha = float(attr.feature_alpha[atom.layer, atom.index])
            rms = math.sqrt(float(good.act_sq[atom.layer, atom.index]))
            proxy[(atom.layer, atom.kind, atom.index)] = abs(alpha) * rms
        else:
            beta = (lm.attn_out(atom.layer).weight.detach().float().cpu().T
                    @ dirs[atom.layer + 1].float().cpu())
            beta = beta.reshape(n_heads, head_dim)[atom.index]
            sq = good.heads_sq[atom.layer].reshape(n_heads, head_dim)[atom.index]
            proxy[(atom.layer, atom.kind, atom.index)] = math.sqrt(
                float((sq * beta.pow(2)).sum())
            )

    ranked = sorted(
        sel.atoms,
        key=lambda atom: atom.load / max(proxy[(atom.layer, atom.kind, atom.index)], 1e-12),
    )
    editor = HybridEditor(lm)
    results = {
        "model": ms.name,
        "plane": args.plane,
        "selection": sel.summary(),
        "proxy": "gamma * |alpha| * rms(harmless activation)",
        "results": {},
    }

    for gamma in [float(x) for x in args.gammas.split(",")]:
        for q in (0.0, 0.25, 0.30, 0.40, 1.0):
            k = int(round(q * len(ranked)))
            removal = ranked[:k]
            redirect = ranked[k:]
            label = f"efficiency_q{q:g}_g{gamma:g}"
            info = editor.apply(removal, redirect, dirs, a, gamma)
            res = measure(lm, baseline, dirs[-1], label, keep_texts=False)
            hold = refusal_rate(lm, bad_extended_prompts())
            results["results"][label] = {
                "quantile": q,
                "gamma": gamma,
                "removal_atoms": len(removal),
                "redirect_atoms": len(redirect),
                "edit": info,
                "refusal_100": res["refusal"],
                "heldout_300": hold,
                "kl_divergence": res["kl_divergence"],
                "perplexity_wikitext2": res["perplexity_wikitext2"],
            }
            print(
                f"[{time.time()-t0:6.1f}s] {label:22s} "
                f"ref={res['refusal']['refusal_rate']:.3f} "
                f"held={hold['refusal_rate']:.3f} "
                f"kl={res['kl_divergence']:.4f} "
                f"ppl={res['perplexity_wikitext2']:.2f}"
            )
            editor.restore()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
