"""Continuous per-atom gamma: taper the redirect by conversion efficiency.

The binary hybrid assigns ``gamma_f = 0`` to the worst fraction and ``gamma`` to
the rest.  That is a step function.  This script gives every selected atom its
own weight,

    d_f' = d_f - alpha_f r + gamma_f alpha_f a,    gamma_f = gamma * t_f

where ``t_f`` is a monotone function of the atom's refusal load per unit of
first-order harmless write.  The binary hybrid is the step-function member of
the family.

Mappings:

* ``linear``   -- ``t = rank`` (lowest efficiency removed, highest full target)
* ``square``   -- ``t = rank^2`` (more aggressive removal)
* ``sqrt``     -- ``t = rank^0.5`` (more aggressive redirect)
* ``soft_q``   -- ``t = clip((rank - q) / w, 0, 1)`` (flat removal below q,
  full target above q + w)

Run:

    PYTHONPATH=. .venv/bin/python scripts/33_continuous_redirect.py
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
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import (
    bad_extended_prompts,
    bad_fit_prompts,
    good_eval_prompts,
    good_fit_prompts,
)
from libershift.edit import RENORM_GUARD, select
from libershift.evaluate import logprobs, measure, refusal_rate


class PerAtomEditor:
    """Apply ``d - alpha r + gamma_f alpha a`` with a per-atom gamma."""

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

    def apply(self, atoms, r, a, gamma_by_atom: dict) -> dict:
        by_layer: dict[int, dict[str, list]] = {}
        for atom in atoms:
            bucket = by_layer.setdefault(atom.layer, {"feature": [], "head": []})
            bucket[atom.kind].append(atom)

        denominator = 0.0
        for layer, bucket in by_layer.items():
            if bucket["feature"]:
                denominator += float(
                    self.lm.ffn_down(layer).weight.detach().float().pow(2).sum()
                )
            if bucket["head"]:
                denominator += float(
                    self.lm.attn_out(layer).weight.detach().float().pow(2).sum()
                )

        changed = 0.0
        for layer, bucket in sorted(by_layer.items()):
            rp = r[layer + 1].float().to(self.lm.device)
            ap = a[layer + 1].float().to(self.lm.device)
            if bucket["feature"]:
                module = self.lm.ffn_down(layer)
                self._keep("feature", layer, module)
                W = module.weight.data.float()
                idx = torch.tensor([x.index for x in bucket["feature"]],
                                   device=W.device)
                gammas = torch.tensor(
                    [gamma_by_atom[(layer, "feature", x.index)]
                     for x in bucket["feature"]],
                    device=W.device,
                )
                cols = W[:, idx]
                alpha = cols.T @ rp
                new = cols - torch.outer(rp, alpha) + torch.outer(ap, alpha * gammas)
                new = self._renorm(new, cols, dim=0)
                W[:, idx] = new
                changed += float((new - cols).pow(2).sum())
                module.weight.data.copy_(W.to(module.weight.dtype))
            if bucket["head"]:
                module = self.lm.attn_out(layer)
                self._keep("head", layer, module)
                W = module.weight.data.float()
                hd = self.lm.head_dim
                for atom in bucket["head"]:
                    start = atom.index * hd
                    block = W[:, start:start + hd]
                    beta = block.T @ rp
                    gamma = float(gamma_by_atom[(layer, "head", atom.index)])
                    new = block - torch.outer(rp, beta) + gamma * torch.outer(ap, beta)
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
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--over", type=float, default=0.90)
    ap.add_argument("--plane", type=int, default=18)
    ap.add_argument("--gammas", default="0.3,0.75")
    ap.add_argument("--out", default=str(OUT / "redirect" / "continuous_sweep.json"))
    args = ap.parse_args()

    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    dirs = dirs[args.plane].unsqueeze(0).repeat(dirs.shape[0], 1)
    attr = attribute(lm, good, bad, dirs)
    sel = select(attr, ("feature", "head"), load_target=args.over, min_t=2.0)
    baseline = logprobs(lm, good_eval_prompts())
    print(f"[{time.time()-t0:6.1f}s] {len(sel.atoms)} atoms selected")

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

    atoms = sorted(
        sel.atoms,
        key=lambda atom: atom.load / max(proxy[(atom.layer, atom.kind, atom.index)], 1e-12),
    )
    rank = {id(atom): i / max(len(atoms) - 1, 1) for i, atom in enumerate(atoms)}

    def gamma_map(mapping: str, gamma: float) -> dict:
        out = {}
        for atom in atoms:
            key = (atom.layer, atom.kind, atom.index)
            t = rank[id(atom)]
            if mapping == "linear":
                factor = t
            elif mapping == "square":
                factor = t * t
            elif mapping == "sqrt":
                factor = math.sqrt(t)
            elif mapping == "soft_q":
                factor = min(1.0, max(0.0, (t - 0.25) / 0.10))
            else:
                raise ValueError(mapping)
            out[key] = gamma * factor
        return out

    editor = PerAtomEditor(lm)
    results = {
        "model": QWEN_1_5B.name,
        "plane": args.plane,
        "selection": sel.summary(),
        "proxy": "gamma * |alpha| * rms(harmless activation), rank-normalised",
        "results": {},
    }
    mappings = ("linear", "square", "sqrt", "soft_q")
    gammas = [float(x) for x in args.gammas.split(",")]

    for gamma in gammas:
        for mapping in mappings:
            label = f"{mapping}_g{gamma:g}"
            info = editor.apply(atoms, dirs, a, gamma_map(mapping, gamma))
            res = measure(lm, baseline, dirs[-1], label, keep_texts=False)
            hold = refusal_rate(lm, bad_extended_prompts())
            results["results"][label] = {
                "mapping": mapping,
                "gamma": gamma,
                "edit": info,
                "refusal_100": res["refusal"],
                "heldout_300": hold,
                "kl_divergence": res["kl_divergence"],
                "perplexity_wikitext2": res["perplexity_wikitext2"],
            }
            print(
                f"[{time.time()-t0:6.1f}s] {label:16s} "
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
