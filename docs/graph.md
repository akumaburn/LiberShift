# The model as a graph

Nodes are FFN feature slots `(layer, feature)`. An edge is where a node writes:
column `f` of `down_proj`, scored against the unembedding, names the tokens that
column pushes toward. `libershift/graph.py` computes that from the weights, so
the view and the edit read the same numbers.

## Why derive it

A stored index can disagree with the weights. This one cannot: every query is a
matrix product against the current `down_proj`, so an edit is visible on the
next read. There is no rebuild step and no cache to invalidate.

The cost is a forward product per query. The unembedding is `[vocab, hidden]`
and a feature column is `hidden`, so scoring `n` features on one layer costs
`vocab * hidden * n`. On a 1.5B model that is 233M multiply-adds per feature;
scoring 200 features across the layers they sit in takes about a second on one
GPU.

## Three queries

| query | what it answers | cost |
| --- | --- | --- |
| `edges(layer, features, k)` | top-k target tokens and their scores for each feature | `vocab * hidden * n` |
| `walk(prompt, layers, top_k)` | which features fire for the prompt's last token, by gate score | `hidden * intermediate` per layer |
| `diff(before, layer, features)` | which features changed target since `before` | one `edges` call |

`walk` uses the gate rows, which no edit in this repository touches, so it is
the control. `edges` uses the down columns, which both edits change.

`feature_loads(layer, direction)` is the cheap selection helper: `down[:, f] .
direction` for every feature, without the statistical attribution.

## What the edit looks like on the graph

Qwen2.5-1.5B-Instruct, plane 18, the 200 highest-load feature atoms, redirect
at `gamma = 0.3` (`results/graph/graph_grad_g0.3.json`):

| measure | value |
| --- | --- |
| top target changed | 31 of 200 (16%) |
| top-5 target list changed | 125 of 200 (62.5%) |
| walk hits before against after | 84 of 84 identical, all gate scores equal |

The top target moves only when the rotated column outscores the old winner,
which is a threshold, not a rate. The list moves far more often because every
score in it is re-weighted. At `gamma = 1.0` the list changes for 143 of 200
features with the acceptance target and 157 of 200 with the mirrored one
(`results/graph/graph_grad_g1.json`, `results/graph/graph_neg_r_g1.json`). The
probe in `results/redirect/pilot_probe.json` records the same 157 as
`down_top_token_changed`; that field counts the top-5 list, not the top token.
The top target changes for 50 of the 200 at that setting.

The walk result is the claim that matters: the same nodes fire. The edit
re-points edges, it does not remove nodes.

## Reproduce

```sh
PYTHONPATH=. .venv/bin/python scripts/29_graph_view.py --gamma 0.3
```

It loads the model, selects the 90% load set, captures the edges, applies the
redirect in memory, and reads the edges again. `--mode` picks the target
(`grad`, `grad_orth`, `neg_r`), `--top-features` sets how many atoms to inspect.

## Where the decomposition comes from

The graph is a consequence of the additive residual contract: a decoder block
writes `h[l+1] = h[l] + attn[l] + ffn[l]`, both writes split exactly into atoms,
and a direction's projection onto the residual is the sum of the atoms'
contributions. `docs/method.md` §1–3 states that decomposition and the identity
that checks it. The graph view is the same object read from the other side: the
identity says an atom's load is `activity * (write . direction)`, and `edges`
shows what the write points at.
