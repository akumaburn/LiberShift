"""Which plane's direction should be ablated, and over which layers?

The trace answers both questions without searching:

* A direction written at plane ``p`` only matters if later layers still write
  along it.  The measured propagation score

      S(p) = sum over l > p of  load_l * max(0, cos(r_p, r_{l+1}))

  is the amount of later refusal write that is aligned with ``r_p``.  The plane
  with the largest score is the best single edit direction, because ablating it
  removes both its own write and the later writes that continue it.

* The causal check: ablate exactly one layer and measure.  Then compare the
  measured curve with the load the trace predicts for that layer.  The gap
  between "writes the direction" and "matters to behaviour" is the reason the
  attribution alone is not a causal claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from libershift import attribute, collect, directions, load_model
from libershift.constants import OUT, QWEN_1_5B
from libershift.data import bad_eval_prompts, bad_fit_prompts, good_fit_prompts
from libershift.edit import Editor
from libershift.evaluate import logprobs, measure, refusal_rate, save_result
from libershift.pipeline import atoms_for_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-prompts", type=int, default=400)
    ap.add_argument("--tag", default="_main")
    args = ap.parse_args()

    out = OUT / (QWEN_1_5B.name + args.tag)
    t0 = time.time()
    lm = load_model(QWEN_1_5B.path)
    good = collect(lm, good_fit_prompts()[: args.fit_prompts])
    bad = collect(lm, bad_fit_prompts()[: args.fit_prompts])
    dirs = directions(bad.h, good.h)
    attr = attribute(lm, good, bad, dirs)
    load = (attr.feature_load.sum(1) + attr.head_load.sum(1))
    baseline = torch.load(out / "baseline_logprobs.pt", weights_only=True)
    readout = dirs[-1]
    prompts = bad_eval_prompts()

    # --- propagation score per candidate plane.
    planes = dirs[1:]  # plane p corresponds to r_p (p = 1..L)
    scores = []
    for p in range(planes.shape[0]):
        r = planes[p]
        cos = (planes @ r).clamp_min(0.0)
        later = torch.zeros_like(load)
        later[p:] = load[p:]  # layers at or after p
        scores.append(float((later * cos).sum()))
    best_plane = 1 + int(torch.tensor(scores).argmax())
    print("propagation score by plane:", " ".join(f"{s:.0f}" for s in scores))
    print(f"best plane by propagation: {best_plane} (r_{best_plane})")

    editor = Editor(lm)
    results = {"propagation": scores, "best_plane": best_plane,
               "load": load.tolist(), "per_layer": {}, "ranges": {}}

    # --- single-layer ablation, using the best plane's direction everywhere.
    editor.set_directions(dirs[best_plane].unsqueeze(0).repeat(dirs.shape[0], 1))
    base_rate = refusal_rate(lm, prompts)["refusal_rate"]
    results["baseline_refusal"] = base_rate
    print(f"baseline refusal {base_rate:.2f}")
    for layer in range(lm.n_layers):
        atoms = atoms_for_layers(attr, ("feature", "head"), [layer])
        editor.apply(atoms, renorm=True)
        rate = refusal_rate(lm, prompts)["refusal_rate"]
        results["per_layer"][layer] = rate
        editor.reset()
        print(f"[{time.time()-t0:6.1f}s] single layer {layer:2d}: refusal={rate:.2f} "
              f"(load {float(load[layer]):.1f})")
    torch.cuda.empty_cache()

    # --- layer-range ablation using the chosen direction.
    def run_range(name: str, layers: list[int]) -> None:
        atoms = atoms_for_layers(attr, ("feature", "head"), layers)
        info = editor.apply(atoms, renorm=True)
        res = measure(lm, baseline, readout, name, keep_texts=True)
        res["edit"] = info
        res["layers"] = layers
        results["ranges"][name] = res
        save_result(out / f"range_{name}.json", res)
        print(f"[{time.time()-t0:6.1f}s] {name:14s} refusal={res['refusal']['refusal_rate']:.2f} "
              f"kl={res['kl_divergence']:.4f} ppl={res['perplexity_wikitext2']:.2f} "
              f"rel_dw={info['relative_change']:.4f} layers={len(layers)}")
        editor.reset()
        torch.cuda.empty_cache()

    p = best_plane
    run_range(f"single_{p}", [p - 1])  # layer index = plane - 1
    run_range(f"from_{p}", list(range(p - 1, lm.n_layers)))
    run_range(f"range_{max(p-4,0)}_{p+8}", list(range(max(p - 4, 0), min(p + 8, lm.n_layers))))
    run_range(f"range_{max(p-8,0)}_{p+8}", list(range(max(p - 8, 0), min(p + 8, lm.n_layers))))
    run_range("all", list(range(lm.n_layers)))

    (out / "causal_layers.json").write_text(json.dumps(results, indent=2))
    print(f"[{time.time()-t0:6.1f}s] wrote {out / 'causal_layers.json'}")


if __name__ == "__main__":
    sys.exit(main())
