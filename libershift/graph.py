"""The model as a graph, computed from the weights.

Nodes are FFN feature slots ``(layer, feature)``. An edge is where a node
writes: column ``f`` of ``down_proj`` scored against the unembedding gives the
tokens that column pushes toward, and the top-k of that score is the feature's
target list. That is the whole graph, and it is derived, not stored, so it can
never disagree with the weights.

Three queries cover what the edit needs:

* :meth:`EdgeGraph.edges` -- the targets of a feature set. Run it before and
  after a redirect to see the edge move.
* :meth:`EdgeGraph.walk` -- which features fire for a prompt, by gate score.
  The gate rows are not edited, so this is the control: the same nodes fire.
* :meth:`EdgeGraph.feature_loads` -- a cheap approximation of refusal load,
  used to pick a feature set without the full attribution.

Edges are scored against the unembedding used at inference, so for a model with
tied embeddings the two are the same matrix.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .model import LoadedModel


class EdgeGraph:
    """Live edge queries over a loaded model."""

    def __init__(self, lm: LoadedModel) -> None:
        self.lm = lm
        self.unembed = _unembedding(lm)  # [vocab, hidden], f32
        self.embed = _embedding(lm)  # [vocab, hidden], f32 (for walk)

    # ── edges ────────────────────────────────────────────────────────────────

    def scores(self, layer: int, features: list[int] | Tensor) -> Tensor:
        """``[vocab, n]`` dot products of the unembedding against each column."""
        idx = torch.as_tensor(list(features), device=self.lm.device, dtype=torch.long)
        down = self.lm.ffn_down(layer).weight.detach().float().to(self.lm.device)
        return self.unembed @ down[:, idx]

    def edges(self, layer: int, features: list[int] | Tensor, k: int = 5) -> list[dict]:
        """Top-``k`` target tokens for each feature, highest first."""
        feats = [int(f) for f in features]
        if not feats:
            return []
        scores = self.scores(layer, feats)
        values, ids = scores.topk(k, dim=0)
        rows = []
        for col, feature in enumerate(feats):
            tokens = self.lm.tokenizer.convert_ids_to_tokens(ids[:, col].tolist())
            rows.append(
                {
                    "layer": layer,
                    "feature": feature,
                    "target": tokens[0],
                    "also": tokens[1:],
                    "score": float(values[0, col]),
                }
            )
        return rows

    def diff(self, before: list[dict], layer: int, features: list[int], k: int = 5) -> dict:
        """Compare the edges now with ``before``, a list of rows from this layer.

        The graph reads live weights, so the "before" side has to be captured
        before the edit: :meth:`edges` returns plain rows for exactly that
        reason.
        """
        old = {row["feature"]: row for row in before if row["layer"] == layer}
        new = {row["feature"]: row for row in self.edges(layer, features, k)}
        shared = [f for f in new if f in old]
        target_changed = [f for f in shared if old[f]["target"] != new[f]["target"]]
        list_changed = [
            f for f in shared
            if [old[f]["target"], *old[f]["also"]] != [new[f]["target"], *new[f]["also"]]
        ]
        examples = [
            {"feature": f, "before": old[f]["target"], "after": new[f]["target"]}
            for f in target_changed[:10]
        ]
        return {
            "layer": layer,
            "features": len(new),
            "target_changed": len(target_changed),
            "target_changed_fraction": len(target_changed) / max(len(shared), 1),
            "list_changed": len(list_changed),
            "list_changed_fraction": len(list_changed) / max(len(shared), 1),
            "examples": examples,
        }

    # ── walk ─────────────────────────────────────────────────────────────────

    def walk(self, prompt: str, layers: list[int] | None = None, top_k: int = 5) -> list[dict]:
        """Features that fire for ``prompt``'s last token, by gate score."""
        ids = self.lm.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.lm.device)
        last = ids[0, -1]
        x = self.embed[last]
        if layers is None:
            layers = list(range(self.lm.n_layers))
        hits = []
        for layer in layers:
            gate = self.lm.layers[layer].mlp.gate_proj.weight.detach().float()
            scores = gate @ x
            values, idx = scores.topk(top_k)
            meta = self.edges(layer, idx.tolist(), k=3)
            for rank, (value, feature) in enumerate(zip(values.tolist(), idx.tolist())):
                row = meta[rank]
                hits.append(
                    {
                        "layer": layer,
                        "feature": feature,
                        "gate": value,
                        "target": row["target"],
                        "also": row["also"],
                    }
                )
        return hits

    # ── selection helper ─────────────────────────────────────────────────────

    def feature_loads(self, layer: int, direction: Tensor) -> Tensor:
        """``down[:, f] . direction`` for every feature of ``layer``."""
        down = self.lm.ffn_down(layer).weight.detach().float()
        return down.T @ direction.float().to(down.device)


def _unembedding(lm: LoadedModel) -> Tensor:
    head = getattr(lm.model, "lm_head", None)
    if head is None:
        head = lm.model.get_output_embeddings()
    if head is None:
        head = lm.model.get_input_embeddings()
    return head.weight.detach().float().to(lm.device)


def _embedding(lm: LoadedModel) -> Tensor:
    return lm.model.get_input_embeddings().weight.detach().float().to(lm.device)
