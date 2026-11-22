"""Refusal-to-acceptance edge redirect.

Abliteration removes the refusal component from a write atom:

    d_f <- d_f - alpha_f r,        alpha_f = d_f . r

The atom keeps firing, and what it writes no longer points at refusal.  The
redirect keeps the same removal and adds a target:

    d_f <- d_f - alpha_f r + gamma * alpha_f a_l

``a_l`` is a unit direction in plane ``l + 1``.  The gate row, the up row and
the column norm stay fixed, so the feature keeps its detector and its
magnitude, and only its edge target moves.  ``a_l = -r`` is the pilot's
sign flip; ``a_l`` from :func:`acceptance_gradient` is the direction that
raises the acceptance score.

The acceptance score is

    s = logsumexp(logit[accept set]) - logsumexp(logit[refuse set])

at the last prompt position.  Its gradient at plane ``p`` includes the effect of
every later block, so it is the direction in which block ``p - 1``'s write must
move to raise acceptance.

``gamma`` scales the acceptance component against the refusal component it
replaces.  ``gamma = 0`` is plain removal; ``gamma = 1`` gives the replacement
the same magnitude as the removal.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .edit import Selection
from .model import LoadedModel

RENORM_GUARD = 0.25

# First-token openers, read off the base model's own top-8 next tokens: harmful
# prompts put "I", "As", "Sorry" and "Unfortunately" first on every prompt;
# harmless prompts put "The", "Here" and "Sure" first.  These sets define the
# acceptance score only.  They never take part in atom selection.
ACCEPT_WORDS = ["Sure", "Here", "Of", "Yes", "Absolutely", "Certainly",
                "Great", "Okay", "Alright"]
REFUSE_WORDS = ["I", "Sorry", "Unfortunately", "As", "No", "Cannot"]


def token_ids(tokenizer, words: list[str]) -> list[int]:
    """First token id of every word in ``words``, deduplicated and sorted."""
    return sorted({tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in words})


def acceptance_score(logits: Tensor, accept_ids: list[int],
                     refuse_ids: list[int]) -> Tensor:
    """``[batch]`` log-probability mass of acceptance minus refusal."""
    return (torch.logsumexp(logits[:, accept_ids], dim=-1)
            - torch.logsumexp(logits[:, refuse_ids], dim=-1))


def acceptance_gradient(lm: LoadedModel, prompts: list[str], accept_ids: list[int],
                        refuse_ids: list[int], batch_size: int | None = None) -> tuple[Tensor, float]:
    """Mean ``d s / d h[p]`` over prompts, for every plane ``p``, plus the mean score.

    Plane ``p`` is the residual entering block ``p``, so the result has
    ``n_layers + 1`` rows and row ``l + 1`` is the target for block ``l``.
    """
    if batch_size is None:
        batch_size = lm.grad_batch_size
    n = lm.n_layers
    total = torch.zeros(n + 1, lm.hidden_size, dtype=torch.float64)
    score_sum = 0.0
    count = 0
    for i in range(0, len(prompts), batch_size):
        enc = lm.encode(prompts[i : i + batch_size])
        with torch.enable_grad():
            out = lm.model(**enc, output_hidden_states=True, use_cache=False)
            logits = out.logits[:, -1, :].float()
            score = acceptance_score(logits, accept_ids, refuse_ids)
            grads = torch.autograd.grad(score.sum(), out.hidden_states)
        for plane, grad in enumerate(grads):
            total[plane] += grad[:, -1, :].float().double().cpu().sum(dim=0)
        score_sum += float(score.detach().sum())
        count += int(score.shape[0])
    return (total / count).float(), score_sum / max(count, 1)


def unit(v: Tensor) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def top_token_gradient(lm: LoadedModel, prompts: list[str],
                       batch_size: int | None = None) -> tuple[Tensor, Tensor]:
    """Gradient of the model's own top-1 log-probability, per prompt and mean.

    For each prompt the model's greedy next token is the one response the
    harmless first-token distribution puts most mass on, so ``d log p(top1) /
    d h[p]`` is the direction in which plane ``p`` most directly raises or
    lowers the answer a harmless prompt would actually produce.  Returns the
    mean ``[n_planes, hidden]`` and the per-prompt ``[prompts, n_planes, hidden]``
    gradients.
    """
    # A gradient pass carries the whole graph for every layer, so the batch is
    # kept small independently of the generation batch.
    if batch_size is None:
        batch_size = lm.grad_batch_size
    n = lm.n_layers
    rows: list[Tensor] = []
    for i in range(0, len(prompts), batch_size):
        enc = lm.encode(prompts[i : i + batch_size])
        with torch.enable_grad():
            out = lm.model(**enc, output_hidden_states=True, use_cache=False)
            logits = out.logits[:, -1, :].float()
            top = logits.argmax(dim=-1)
            chosen = logits.gather(1, top[:, None]).squeeze(1)
            logp = chosen - torch.logsumexp(logits, dim=-1)
            grads = torch.autograd.grad(logp.sum(), out.hidden_states)
        # [batch, n_planes, hidden], last prompt position only.
        rows.append(torch.stack([g[:, -1, :].float() for g in grads], dim=1).cpu())
    per_prompt = torch.cat(rows, dim=0)
    assert per_prompt.shape[1] == n + 1
    return per_prompt.mean(dim=0), per_prompt


def harmlessness_basis(per_prompt: Tensor, rank: int) -> Tensor:
    """Top-``rank`` principal directions of the harmless first-token gradient.

    ``per_prompt`` is ``[prompts, n_planes, hidden]``.  The rows of the logit
    Jacobian are these gradients, so the leading right singular vectors are the
    residual directions to which the harmless next-token distribution is most
    sensitive.  A redirect target orthogonal to them cannot move that
    distribution to first order along its strongest axes.
    """
    basis = torch.empty(per_prompt.shape[1], rank, per_prompt.shape[2])
    for plane in range(per_prompt.shape[1]):
        x = per_prompt[:, plane, :].float()
        x = x - x.mean(dim=0, keepdim=True)
        # Gram matrix is [prompts, prompts]; its eigenvectors give the row space.
        gram = x @ x.T
        values, vectors = torch.linalg.eigh(gram.double())
        order = values.argsort(descending=True)[:rank]
        vectors = vectors[:, order].float()
        basis[plane] = unit((vectors.T @ x))
    return basis


def project_out(grads: Tensor, basis: Tensor) -> Tensor:
    """Remove from ``grads`` [n_planes, hidden] the span of ``basis`` [n_planes, rank, hidden]."""
    out = grads.clone()
    for plane in range(basis.shape[0]):
        v = basis[plane]
        out[plane] = grads[plane] - (grads[plane] @ v.T) @ v
    return out


def target(grads: Tensor, r: Tensor, mode: str = "grad") -> Tensor:
    """Unit redirect target ``[n_planes, hidden]`` from a gradient and ``r``.

    ``grad`` is the acceptance gradient.  ``grad_orth`` removes the component
    along ``r`` first, because the removal term has already taken ``r`` out and
    adding anything back along it would partly undo the removal.  ``neg_r`` is
    the pilot's target and exists as the control.
    """
    if mode == "neg_r":
        a = -r
    elif mode == "grad":
        a = unit(grads)
    elif mode == "grad_orth":
        a = unit(grads - (grads * r).sum(dim=-1, keepdim=True) * r)
    else:
        raise ValueError(f"unknown target mode {mode!r}; known: neg_r, grad, grad_orth")
    a = a.clone()
    a[0] = 0.0  # plane 0 is the embedding; no block writes into it
    return a


class HybridEdit:
    """Remove one atom set and redirect a disjoint one, one write per matrix.

    The two sets are written in a single pass per matrix, so the result equals
    applying removal to one set and the redirect to the other, without the
    intermediate write.  ``gamma_by_layer`` overrides ``gamma`` per block, which
    is how a per-plane allocation of the acceptance component is expressed.
    """

    def __init__(self, lm: LoadedModel) -> None:
        self.lm = lm
        self.saved: dict[tuple[str, int], Tensor] = {}

    def _keep(self, kind: str, layer: int, module) -> None:
        key = (kind, layer)
        if key not in self.saved:
            # CPU copy: 32 hybrid layers snapshot about 2.2 GB, which crowds a
            # 16 GB GPU during the next generation pass.  The round trip is
            # bit-exact.
            self.saved[key] = module.weight.detach().to("cpu", copy=True)

    def restore(self) -> None:
        for (kind, layer), weight in self.saved.items():
            module = (self.lm.ffn_down(layer) if kind == "feature"
                      else self.lm.attn_out(layer))
            module.weight.data.copy_(
                weight.to(module.weight.device, module.weight.dtype)
            )
        self.saved.clear()

    @staticmethod
    def _renorm(new: Tensor, old: Tensor, dim: int | None) -> Tensor:
        n0 = old.norm(dim=dim)
        n1 = new.norm(dim=dim).clamp_min(1e-12)
        guard = n1 < RENORM_GUARD * n0
        scale = torch.where(guard, torch.ones_like(n1), n0 / n1)
        if dim is None:
            return new * scale
        return new * (scale.unsqueeze(0) if dim == 0 else scale)

    def apply(self, removal, redirect, r: Tensor, a: Tensor, gamma: float,
              gamma_by_layer: dict[int, float] | None = None) -> dict:
        rem_feat: dict[int, list[int]] = {}
        red_feat: dict[int, list[int]] = {}
        rem_head: dict[int, list[int]] = {}
        red_head: dict[int, list[int]] = {}
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
            gl = gamma if gamma_by_layer is None else gamma_by_layer.get(layer, gamma)
            if layer in rem_feat or layer in red_feat:
                module = self.lm.ffn_down(layer)
                self._keep("feature", layer, module)
                W = module.weight.data.float()
                for indices, gain in ((rem_feat.get(layer, []), 0.0),
                                      (red_feat.get(layer, []), gl)):
                    if not indices:
                        continue
                    idx = torch.tensor(indices, device=W.device)
                    cols = W[:, idx]
                    alpha = cols.T @ rp
                    new = cols - torch.outer(rp, alpha)
                    if gain:
                        new = new + gain * torch.outer(ap, alpha)
                    new = self._renorm(new, cols, dim=0)
                    W[:, idx] = new
                    changed += float((new - cols).pow(2).sum())
                module.weight.data.copy_(W.to(module.weight.dtype))
            if layer in rem_head or layer in red_head:
                module = self.lm.attn_out(layer)
                self._keep("head", layer, module)
                W = module.weight.data.float()
                hd = self.lm.head_dim
                for indices, gain in ((rem_head.get(layer, []), 0.0),
                                      (red_head.get(layer, []), gl)):
                    for head in indices:
                        start = head * hd
                        block = W[:, start:start + hd]
                        beta = block.T @ rp
                        new = block - torch.outer(rp, beta)
                        if gain:
                            new = new + gain * torch.outer(ap, beta)
                        new = self._renorm(new, block, dim=None)
                        W[:, start:start + hd] = new
                        changed += float((new - block).pow(2).sum())
                module.weight.data.copy_(W.to(module.weight.dtype))
        torch.cuda.empty_cache()
        return {
            "changed_sq_norm": changed,
            "edited_family_sq_norm": denominator,
            "relative_change": (changed / denominator) ** 0.5 if denominator else 0.0,
            "gamma": gamma,
            "removal_atoms": len(removal),
            "redirect_atoms": len(redirect),
        }


class WeightSnapshot:
    """Restores exactly the matrices an edit touched, so runs can be compared."""

    def __init__(self, lm: LoadedModel) -> None:
        self.lm = lm
        self.saved: dict[tuple[str, int], Tensor] = {}

    def keep(self, kind: str, layer: int, module) -> None:
        key = (kind, layer)
        if key not in self.saved:
            # CPU copy, same reason as HybridEdit._keep.
            self.saved[key] = module.weight.detach().to("cpu", copy=True)

    def restore(self) -> None:
        for (kind, layer), weight in self.saved.items():
            module = self.lm.ffn_down(layer) if kind == "feature" else self.lm.attn_out(layer)
            module.weight.data.copy_(weight.to(module.weight.device, module.weight.dtype))
        self.saved.clear()


def apply_redirect(lm: LoadedModel, sel: Selection | list, r: Tensor, a: Tensor,
                   gamma: float = 1.0, snapshot: WeightSnapshot | None = None,
                   ops: dict[int, dict[str, dict]] | None = None) -> dict:
    """Rotate the selected atoms' refusal component into the acceptance target.

    ``r`` and ``a`` are indexed by plane, so layer ``l`` uses row ``l + 1``.
    Every edited column is rescaled to its original norm, with the same guard
    the ablation uses: a column whose norm would fall below a quarter of its
    original keeps its scale instead.

    Pass a dict as ``ops`` to record the edit in operator form.  The record is
    what ``libershift.patch.write_redirect_patch`` stores, and it needs no
    replacement vectors: ``(direction, target, gamma, indices, scale)`` per
    layer reproduces the edit from a pristine checkpoint.
    """
    atoms = sel.atoms if hasattr(sel, "atoms") else sel
    by_layer: dict[int, list] = {}
    for atom in atoms:
        by_layer.setdefault(atom.layer, []).append(atom)
    snapshot = snapshot or WeightSnapshot(lm)

    kinds = {atom.kind for atom in atoms}
    denominator = 0.0
    for layer in sorted(by_layer):
        if "feature" in kinds:
            denominator += float(lm.ffn_down(layer).weight.detach().float().pow(2).sum())
        if "head" in kinds:
            denominator += float(lm.attn_out(layer).weight.detach().float().pow(2).sum())

    changed = 0.0
    for layer, layer_atoms in sorted(by_layer.items()):
        rp = r[layer + 1].float().to(lm.device)
        ap = a[layer + 1].float().to(lm.device)
        feats = [x for x in layer_atoms if x.kind == "feature"]
        if feats:
            module = lm.ffn_down(layer)
            snapshot.keep("feature", layer, module)
            W = module.weight.data.float()
            idx = torch.tensor([x.index for x in feats], device=W.device)
            cols = W[:, idx]
            alpha = cols.T @ rp
            new = cols - torch.outer(rp, alpha) + gamma * torch.outer(ap, alpha)
            n0 = cols.norm(dim=0)
            n1 = new.norm(dim=0).clamp_min(1e-12)
            scale = torch.where(n1 < RENORM_GUARD * n0, torch.ones_like(n1), n0 / n1)
            new = new * scale
            W[:, idx] = new
            module.weight.data.copy_(W.to(module.weight.dtype))
            changed += float((new - cols).pow(2).sum())
            if ops is not None:
                ops.setdefault(layer, {})["feature"] = {
                    "direction": rp.detach().float().cpu(),
                    "target": ap.detach().float().cpu(),
                    "indices": idx.detach().cpu(),
                    "scale": scale.detach().float().cpu(),
                    "gamma": gamma,
                    "full": len(feats) == module.weight.shape[1],
                }
        heads = [x for x in layer_atoms if x.kind == "head"]
        if heads:
            module = lm.attn_out(layer)
            snapshot.keep("head", layer, module)
            W = module.weight.data.float()
            hd = lm.head_dim
            head_idx, head_scale = [], []
            for atom in heads:
                start = atom.index * hd
                block = W[:, start : start + hd]
                beta = block.T @ rp
                new = block - torch.outer(rp, beta) + gamma * torch.outer(ap, beta)
                n0 = block.norm()
                n1 = new.norm().clamp_min(1e-12)
                factor = 1.0 if float(n1) < RENORM_GUARD * float(n0) else float(n0 / n1)
                new = new * factor
                W[:, start : start + hd] = new
                changed += float((new - block).pow(2).sum())
                head_idx.append(atom.index)
                head_scale.append(factor)
            module.weight.data.copy_(W.to(module.weight.dtype))
            if ops is not None:
                ops.setdefault(layer, {})["head"] = {
                    "direction": rp.detach().float().cpu(),
                    "target": ap.detach().float().cpu(),
                    "indices": torch.tensor(head_idx),
                    "scale": torch.tensor(head_scale, dtype=torch.float32),
                    "gamma": gamma,
                    "full": len(heads) == lm.n_heads,
                }

    torch.cuda.empty_cache()
    return {
        "gamma": gamma,
        "changed_sq_norm": changed,
        "edited_family_sq_norm": denominator,
        "relative_change": (changed / denominator) ** 0.5 if denominator else 0.0,
    }
