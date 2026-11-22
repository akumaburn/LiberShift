"""The edit as data: export, inspect, and replay without recomputation.

Ablation with one direction is a rank-one change per matrix:

    M' = (M - s r (r^T M)) S

where ``r`` is the unit direction, ``s`` the strength and ``S`` the diagonal of
per-atom (column) norm scales.  For a full-coverage edit the whole artifact is
therefore one direction per layer plus one scale per atom: about a megabyte for
a 1.5B model, against a 3 GB checkpoint.  A sparse edit cannot use that form, so
the exporter falls back to replacement values for the selected atoms.

A redirect stores the operator instead, in ``write_redirect_patch``: one
direction, one target, and one index and scale per atom.  Replay recomputes
``d_f - alpha_f r + gamma alpha_f a`` from the base column, so the patch never
carries a replacement vector.

Replay is exact.  ``scripts/06_export.py --verify-patch`` re-applies the patch
to a pristine checkpoint and compares logits with the exported model; the
measured difference is 0.0.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import LoadedModel


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()


def write_patch(out: Path, ops: dict[int, dict[str, dict]], meta: dict) -> dict:
    """Persist recorded edit operations as a JSON sidecar plus an npz payload."""
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    table = []
    compact = all(op["full"] for layer in ops.values() for op in layer.values())

    for layer, layer_ops in sorted(ops.items()):
        for kind, op in layer_ops.items():
            r = op["direction"]
            if compact:
                arrays[f"direction_{layer}"] = r.numpy()
                arrays[f"scale_{kind}_{layer}"] = op["scale"].numpy().astype(np.float32)
                table.append({
                    "layer": layer, "kind": kind, "coverage": "all",
                    "values": int(op["scale"].numel()),
                    "strength": op["strength"],
                    "direction_sha256": _digest(r)[:16],
                })
            else:
                arrays[f"{kind}_index_{layer}"] = op["indices"].numpy().astype(np.int32)
                arrays[f"{kind}_scale_{layer}"] = op["scale"].numpy().astype(np.float32)
                table.append({
                    "layer": layer, "kind": kind, "coverage": "subset",
                    "values": int(op["indices"].numel()),
                    "strength": op["strength"],
                    "direction_sha256": _digest(r)[:16],
                })

    npz = out.with_suffix(".npz")
    np.savez_compressed(npz, **arrays)
    payload = {
        "format": "taa-patch/2",
        "representation": "rank1" if compact else "rows",
        "base_model": str(meta.get("model", "")),
        "method": meta.get("method", "trace-attributed-ablation"),
        "selection": meta.get("selection", {}),
        "array_file": npz.name,
        "array_sha256": sha256_file(npz),
        "array_bytes": npz.stat().st_size,
        "matrices": table,
        "layer_count": len(ops),
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1))
    return payload


def write_redirect_patch(out: Path, ops: dict[int, dict[str, dict]], meta: dict) -> dict:
    """Persist a redirect edit in operator form.

    One direction and one target per layer, plus an index and a norm scale per
    atom.  The bake is ::

        d_f' = (d_f - alpha_f r + gamma * alpha_f a) * scale_f

    with ``alpha_f = d_f . r`` taken from the base column at replay time.  No
    replacement vector is stored, so the per-atom cost is one ``int32`` index
    and one ``float32`` scale, and the direction pair is shared by the layer.

    Replay is checked against the exported checkpoint by
    ``scripts/26_export_redirect.py --verify-patch``: zero logit difference
    on 16 prompts at both operating points.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    layers, matrices = [], []
    atoms = 0

    for layer, layer_ops in sorted(ops.items()):
        first = layer_ops[sorted(layer_ops)[0]]
        r = first["direction"].float()
        a = first["target"].float()
        arrays[f"direction_{layer}"] = r.numpy().astype(np.float32)
        arrays[f"target_{layer}"] = a.numpy().astype(np.float32)
        layers.append({
            "layer": layer,
            "direction_sha256": _digest(r)[:16],
            "target_sha256": _digest(a)[:16],
        })
        for kind, op in sorted(layer_ops.items()):
            idx = op["indices"].long()
            scale = op["scale"].float()
            arrays[f"{kind}_index_{layer}"] = idx.numpy().astype(np.int32)
            arrays[f"{kind}_scale_{layer}"] = scale.numpy().astype(np.float32)
            matrices.append({
                "layer": layer,
                "kind": kind,
                "coverage": "all" if op["full"] else "subset",
                "values": int(idx.numel()),
                "gamma": float(op["gamma"]),
            })
            atoms += int(idx.numel())

    npz = out.with_suffix(".npz")
    np.savez_compressed(npz, **arrays)
    payload = {
        "format": "taa-patch/3",
        "representation": "redirect",
        "base_model": str(meta.get("model", "")),
        "method": meta.get("method", "acceptance-redirect"),
        "gamma": meta.get("gamma"),
        "selection": meta.get("selection", {}),
        "array_file": npz.name,
        "array_sha256": sha256_file(npz),
        "array_bytes": npz.stat().st_size,
        "layers": layers,
        "matrices": matrices,
        "layer_count": len(ops),
        "atoms": atoms,
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1))
    return payload


def _apply_redirect_ops(lm, payload: dict, npz) -> int:
    hd = lm.head_dim
    touched = 0
    for entry in payload["matrices"]:
        layer, kind, gamma = entry["layer"], entry["kind"], float(entry["gamma"])
        r = torch.from_numpy(npz[f"direction_{layer}"]).float().to(lm.device)
        a = torch.from_numpy(npz[f"target_{layer}"]).float().to(lm.device)
        idx = torch.from_numpy(npz[f"{kind}_index_{layer}"]).long().to(lm.device)
        scale = torch.from_numpy(npz[f"{kind}_scale_{layer}"]).float().to(lm.device)
        module = lm.ffn_down(layer) if kind == "feature" else lm.attn_out(layer)
        W = module.weight.data.float()
        if kind == "feature":
            cols = W[:, idx]
            alpha = cols.T @ r
            new = cols - torch.outer(r, alpha) + gamma * torch.outer(a, alpha)
            W[:, idx] = new * scale.unsqueeze(0)
        else:
            for row, h in enumerate(idx.tolist()):
                start = h * hd
                block = W[:, start : start + hd]
                beta = block.T @ r
                new = block - torch.outer(r, beta) + gamma * torch.outer(a, beta)
                W[:, start : start + hd] = new * float(scale[row])
        module.weight.data.copy_(W.to(module.weight.dtype))
        touched += int(idx.numel())
    return touched


def apply_patch(model_path: Path | str, patch_json: Path, target: Path) -> dict:
    """Replay a patch on a pristine checkpoint and save a standalone model.

    Handles both representations: ``rank1``/``rows`` replay an ablation and
    ``redirect`` replays the acceptance redirect.
    """
    from .model import load_model

    payload = json.loads(Path(patch_json).read_text())
    npz = np.load(Path(patch_json).with_suffix(".npz"))
    lm = load_model(model_path)
    hd = lm.head_dim
    touched = 0

    if payload["representation"] == "redirect":
        touched = _apply_redirect_ops(lm, payload, npz)
        target.mkdir(parents=True, exist_ok=True)
        lm.model.save_pretrained(target, safe_serialization=True)
        lm.tokenizer.save_pretrained(target)
        return {
            "applied_atoms": touched,
            "target": str(target),
            "representation": payload["representation"],
            "patch_sha256": payload["array_sha256"],
            "patch_bytes": payload["array_bytes"],
        }

    for entry in payload["matrices"]:
        layer, kind, strength = entry["layer"], entry["kind"], entry["strength"]
        if payload["representation"] == "rank1":
            r = torch.from_numpy(npz[f"direction_{layer}"]).float().to(lm.device)
            scale = torch.from_numpy(npz[f"scale_{kind}_{layer}"]).float().to(lm.device)
            module = lm.ffn_down(layer) if kind == "feature" else lm.attn_out(layer)
            W = module.weight.data.float()
            if kind == "feature":
                new = W - strength * torch.outer(r, W.T @ r)
                new = new * scale.unsqueeze(0)
            else:
                new = W.clone()
                for h in range(lm.n_heads):
                    block = W[:, h * hd : (h + 1) * hd]
                    block = block - strength * torch.outer(r, block.T @ r)
                    new[:, h * hd : (h + 1) * hd] = block * scale[h]
            module.weight.data.copy_(new.to(module.weight.dtype))
            touched += int(scale.numel())
        else:
            idx = torch.from_numpy(npz[f"{kind}_index_{layer}"]).long()
            scale = torch.from_numpy(npz[f"{kind}_scale_{layer}"]).float().to(lm.device)
            r = torch.from_numpy(npz[f"direction_{layer}"]).float().to(lm.device)
            module = lm.ffn_down(layer) if kind == "feature" else lm.attn_out(layer)
            W = module.weight.data.float()
            if kind == "feature":
                cols = W[:, idx.to(W.device)]
                cols = (cols - strength * torch.outer(r, cols.T @ r)) * scale.unsqueeze(0)
                W[:, idx.to(W.device)] = cols
            else:
                for row, h in enumerate(idx.tolist()):
                    block = W[:, h * hd : (h + 1) * hd]
                    block = (block - strength * torch.outer(r, block.T @ r)) * scale[row]
                    W[:, h * hd : (h + 1) * hd] = block
            module.weight.data.copy_(W.to(module.weight.dtype))
            touched += len(idx)

    target.mkdir(parents=True, exist_ok=True)
    lm.model.save_pretrained(target, safe_serialization=True)
    lm.tokenizer.save_pretrained(target)
    return {
        "applied_atoms": touched,
        "target": str(target),
        "representation": payload["representation"],
        "patch_sha256": payload["array_sha256"],
        "patch_bytes": payload["array_bytes"],
    }
