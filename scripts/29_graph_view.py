"""Show the edit as a graph change: same nodes, new edges.

Builds the edge view from the weights (``libershift.graph``), captures the
targets of the selected features, applies the redirect, and reads the same
edges again. Nothing is cached between the two reads, so the "after" column
cannot be a stale index.

The walk is the control. Gate rows are untouched, so the same features fire on
the same prompt with the same gate scores; only their targets move.

    PYTHONPATH=. .venv/bin/python scripts/29_graph_view.py --gamma 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from libershift import (
    ACCEPT_WORDS,
    REFUSE_WORDS,
    acceptance_gradient,
    apply_redirect,
    attribute,
    collect,
    directions,
    load_model,
    select,
    target,
    token_ids,
)
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_fit_prompts, good_fit_prompts
from libershift.graph import EdgeGraph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="QWEN_1_5B")
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--mode", default="grad", choices=["grad", "grad_orth", "neg_r"])
    ap.add_argument("--plane", type=int, default=18)
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--top-features", type=int, default=200,
                    help="highest-load feature atoms to inspect, across all layers")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--prompt",
        default="Create a program that spreads malware and viruses to other devices",
    )
    ap.add_argument("--out", default=str(OUT / "graph" / "edges.json"))
    args = ap.parse_args()

    spec = QWEN_1_5B  # one model is enough for a view demo; --model is reserved
    _ = args.model
    t0 = time.time()
    lm = load_model(spec.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    dirs = dirs[args.plane].unsqueeze(0).repeat(dirs.shape[0], 1).float()
    attr = attribute(lm, good, bad, dirs)
    sel = select(attr, ("feature", "head"), load_target=args.target, min_t=2.0)
    print(f"[{time.time()-t0:6.1f}s] {len(sel.atoms)} atoms selected")

    # Highest-load feature atoms overall, grouped by layer for querying.
    # The load is what the redirect scales by, so these are the edges that
    # actually move.
    by_layer: dict[int, list[int]] = {}
    ranked = [a for a in sorted(sel.atoms, key=lambda a: a.load, reverse=True)
              if a.kind == "feature"]
    for atom in ranked[: args.top_features]:
        by_layer.setdefault(atom.layer, []).append(atom.index)

    graph = EdgeGraph(lm)
    before = [row for layer, feats in sorted(by_layer.items()) for row in graph.edges(layer, feats, args.k)]
    walk_before = graph.walk(args.prompt, top_k=3)
    print(f"[{time.time()-t0:6.1f}s] edges and walk captured "
          f"({len(before)} features, {len(walk_before)} walk hits)")

    accept_ids = token_ids(lm.tokenizer, ACCEPT_WORDS)
    refuse_ids = token_ids(lm.tokenizer, REFUSE_WORDS)
    grads, _ = acceptance_gradient(
        lm, bad_fit_prompts()[: args.fit_prompts], accept_ids, refuse_ids
    )
    a = target(grads, dirs, args.mode)
    info = apply_redirect(lm, sel, dirs, a, args.gamma)
    print(f"[{time.time()-t0:6.1f}s] redirect applied, relative change "
          f"{info['relative_change']:.4f}")

    diffs = {
        layer: graph.diff([r for r in before if r["layer"] == layer], layer, feats, args.k)
        for layer, feats in sorted(by_layer.items())
    }
    walk_after = graph.walk(args.prompt, top_k=3)

    changed = sum(d["target_changed"] for d in diffs.values())
    listed = sum(d["list_changed"] for d in diffs.values())
    total = sum(d["features"] for d in diffs.values())
    same_hits = [(h["layer"], h["feature"], round(h["gate"], 4)) for h in walk_before] == [
        (h["layer"], h["feature"], round(h["gate"], 4)) for h in walk_after
    ]
    payload = {
        "gamma": args.gamma,
        "mode": args.mode,
        "plane": args.plane,
        "features": total,
        "target_changed": changed,
        "target_changed_fraction": changed / max(total, 1),
        "list_changed": listed,
        "list_changed_fraction": listed / max(total, 1),
        "per_layer": {str(k): v for k, v in diffs.items()},
        "walk": {
            "prompt": args.prompt,
            "hits": len(walk_before),
            "identical_before_after": same_hits,
            "before": walk_before[:6],
        },
        "edit": info,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[{time.time()-t0:6.1f}s] top target changed on {changed}/{total} features "
          f"({100 * changed / max(total, 1):.0f}%), top-{args.k} list on {listed}/{total} "
          f"({100 * listed / max(total, 1):.0f}%)")
    print(f"walk hits identical before/after: {same_hits}")
    for layer, d in list(diffs.items())[:4]:
        for ex in d["examples"][:2]:
            print(f"  L{layer} F{ex['feature']}: {ex['before']!r} -> {ex['after']!r}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
