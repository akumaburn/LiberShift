"""Parity check: LARQL's own layer dump against the capture this study uses.

`larql shannon layer-dump` writes one f32 plane per end-of-layer residual from
the LARQL engine.  This script runs the same token ids through HF transformers
and compares plane by plane.  Agreement means the residual planes, and therefore
the additive decomposition built on them, are the same object in both engines.

Two details decide the result:

* dtype.  LARQL dumps f32.  A bfloat16 transformers run agrees only to about
  1%, which is the forward pass's own precision, so the default here is f32.
* the last plane.  Transformers 5.x returns `hidden_states[-1]` after the
  model's closing norm.  LARQL's last plane is the pre-norm residual, equal to
  `h[-2] + attn + ffn` at the final position.  The comparison records all three
  candidates so the difference is visible rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from libershift.constants import OUT, QWEN_1_5B


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(OUT / "larql_parity" / "dump"))
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    dump = Path(args.dump)
    manifest = json.loads((dump / "layer_dump.json").read_text())
    ids = torch.tensor([manifest["token_ids"]], dtype=torch.long)
    n_layers = manifest["num_layers"]
    seq = manifest["seq_len"]
    print(f"larql dump: {n_layers} layers, seq={seq}, engine={manifest.get('engine')}")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        str(QWEN_1_5B.path), dtype=dtype, device_map="cuda", attn_implementation="eager"
    )
    buf: dict[str, torch.Tensor] = {}
    handles = [
        model.model.layers[-1].self_attn.register_forward_hook(
            lambda _m, _i, o: buf.__setitem__("a", o[0] if isinstance(o, tuple) else o)
        ),
        model.model.layers[-1].mlp.register_forward_hook(lambda _m, _i, o: buf.__setitem__("f", o)),
    ]
    with torch.no_grad():
        out = model(input_ids=ids.to(model.device), output_hidden_states=True)
    for h in handles:
        h.remove()
    hidden = out.hidden_states

    rows = []
    for plane in range(n_layers + 1):
        path = dump / f"layer_{plane:03d}.f32"
        if not path.exists():
            continue
        arr = np.fromfile(path, dtype=np.float32).reshape(seq, -1)
        ref = hidden[plane].detach().float().cpu().numpy()
        err = float(np.linalg.norm(arr - ref) / np.linalg.norm(ref))
        row = {"plane": plane, "rel_err": err}
        if plane == n_layers:
            recon = (hidden[plane - 1] + buf["a"] + buf["f"]).detach().float().cpu().numpy()
            post = model.model.norm(hidden[plane]).detach().float().cpu().numpy()
            row["rel_err_reconstruction"] = float(np.linalg.norm(arr - recon) / np.linalg.norm(recon))
            row["rel_err_post_norm"] = float(np.linalg.norm(arr - post) / np.linalg.norm(post))
        rows.append(row)

    pre_norm = [r["rel_err"] for r in rows if r["plane"] < n_layers]
    last = rows[-1]
    print(f"dtype: {args.dtype}")
    print("plane 0 relative error:", f"{rows[0]['rel_err']:.1e}")
    print("planes 1..%d worst relative error:" % (n_layers - 1), f"{max(pre_norm[1:]):.1e}")
    print("last plane vs returned hidden state:", f"{last['rel_err']:.1e}")
    print("last plane vs reconstruction:      ", f"{last.get('rel_err_reconstruction', float('nan')):.1e}")
    print("last plane vs post-norm value:     ", f"{last.get('rel_err_post_norm', float('nan')):.1e}")

    payload = {
        "manifest": {k: v for k, v in manifest.items() if k != "token_ids"},
        "dtype": args.dtype,
        "plane0_rel_err": rows[0]["rel_err"],
        "pre_norm_worst_rel_err": max(pre_norm[1:]),
        "last_plane": last,
        "planes": rows,
    }
    (dump.parent / "parity.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {dump.parent / 'parity.json'}")


if __name__ == "__main__":
    sys.exit(main())
