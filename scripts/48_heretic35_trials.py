"""Parse Heretic's own per-trial scores from the Qwen3.5-4B run logs.

The exported checkpoints are Heretic's best trials; this records what the
optimiser saw when it chose them, per its own scorers.  The harness
measurements of those checkpoints live in comparison_Qwen3.5-4B.json; the two
files agree on the exports but score different protocols (Heretic's KL slice is
its calibration set, ours is the harmless evaluation slice).

    PYTHONPATH=. .venv/bin/python scripts/48_heretic35_trials.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
RESULTS = ROOT / "results"


def trials(log_path: Path) -> dict[int, dict]:
    trials: dict[int, dict] = {}
    current = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = re.search(r"Running trial (\d+) of 40", line)
        if m:
            current = int(m.group(1))
            continue
        m = re.search(r"Refusals: (\d+)/100", line)
        if m and current is not None:
            trials.setdefault(current, {})["refusals"] = int(m.group(1))
            continue
        m = re.search(r"KL divergence: ([\d.]+)", line)
        if m and current is not None:
            trials.setdefault(current, {})["kl"] = float(m.group(1))
    return {t: v for t, v in trials.items() if "refusals" in v and "kl" in v}


def main() -> None:
    payload = {}
    for run, modifier in (("mpoa", "MPOA"), ("ara", "ARA")):
        log = OUT / f"heretic_{run}_35b.log"
        pairs = trials(log)
        best = min(pairs, key=lambda t: (pairs[t]["refusals"], pairs[t]["kl"]))
        payload[run] = {
            "modifier": modifier,
            "trials_scored": len(pairs),
            "best_trial": best,
            "refusals": pairs[best]["refusals"],
            "kl_divergence": pairs[best]["kl"],
            "log": str(log.relative_to(ROOT)),
        }
        print(f"{run}: best trial {best} -> {pairs[best]['refusals']}/100, "
              f"KL {pairs[best]['kl']}")
    payload["_source"] = "per-trial scores parsed from the Heretic run logs; " \
                         "the exported checkpoint is the restored best trial"
    path = RESULTS / "heretic_35b_trials.json"
    path.write_text(json.dumps(payload, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    sys.exit(main())
