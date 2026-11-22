"""Prompt loading, pinned to the dataset commits Heretic uses."""

from __future__ import annotations

import json
from pathlib import Path

from .constants import (
    BAD_DATASET,
    BAD_EVAL_SPLIT,
    BAD_FIT_SPLIT,
    CACHE,
    GOOD_DATASET,
    GOOD_EVAL_SPLIT,
    GOOD_FIT_SPLIT,
)


def _load(dataset: tuple[str, str], split: str, column: str) -> list[str]:
    from datasets import load_dataset

    name, commit = dataset
    ds = load_dataset(name, split=split, revision=commit)
    return [row[column] for row in ds]


def _cached(key: str, dataset: tuple[str, str], split: str, column: str) -> list[str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = _load(dataset, split, column)
    path.write_text(json.dumps(rows, indent=1))
    return rows


def good_fit_prompts() -> list[str]:
    return _cached("good_fit", GOOD_DATASET, GOOD_FIT_SPLIT, "text")


def bad_fit_prompts() -> list[str]:
    return _cached("bad_fit", BAD_DATASET, BAD_FIT_SPLIT, "text")


def good_eval_prompts() -> list[str]:
    return _cached("good_eval", GOOD_DATASET, GOOD_EVAL_SPLIT, "text")


def bad_eval_prompts() -> list[str]:
    return _cached("bad_eval", BAD_DATASET, BAD_EVAL_SPLIT, "text")


def capability_text(chunks: int = 32, chunk_tokens: int = 512) -> str:
    """Held-out natural text for perplexity, from wikitext-2 test."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    return text[: chunks * chunk_tokens * 6]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def bad_extended_prompts() -> list[str]:
    """Held-out harmful prompts from two families the method never touched.

    Heretic's harmful slice has only 104 rows in its test split, so there is no
    room for a large holdout inside it.  These come from JailbreakBench
    (HarmBench-derived goals) and from the unsafe prompts of BeaverTails.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "bad_holdout.json"
    if path.exists():
        return json.loads(path.read_text())

    from datasets import load_dataset

    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    goals = [row["Goal"] for row in jbb]
    tails = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    unsafe = [row["prompt"] for row in tails if not row["is_safe"]][:200]
    rows = goals + unsafe
    path.write_text(json.dumps(rows, indent=1))
    return rows


def good_extended_prompts() -> list[str]:
    return _cached("good_extended", GOOD_DATASET, "test[100:400]", "text")
