"""Persist DPO pairs to a JSONL file."""

from __future__ import annotations

import json
from pathlib import Path


def append_pair(pair: dict, dataset_path: Path) -> None:
    """Append exactly {"prompt": ..., "chosen": ..., "rejected": ...} to the JSONL file."""
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"prompt": pair["prompt"], "chosen": pair["chosen"], "rejected": pair["rejected"]},
        ensure_ascii=False,
    )
    with dataset_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
