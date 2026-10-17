"""Small helpers shared across the pipeline."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

PROBLEM = "problem"
SOLUTION = "solution"

EDGE_ALIASES = {"backtracking": "repetition"}


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def node_key(node_id: str) -> float:
    """Sort key putting the problem node first, then steps in generation order."""
    if node_id == PROBLEM:
        return -1
    return int(node_id) if str(node_id).isdigit() else float("inf")


def sorted_nodes(nodes) -> list[str]:
    return sorted(nodes, key=node_key)


def fill(template: str, **values) -> str:
    """Fill ``{name}`` placeholders without interpreting other braces."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def read_text(path: str | Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(obj), f, indent=indent, ensure_ascii=False)


def read_jsonl(path: str | Path) -> dict:
    """Read a JSONL file of ``{id: entry}`` lines into one dict."""
    data = {}
    path = Path(path)
    if not path.exists():
        return data
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.update(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def append_jsonl(path: str | Path, key: str, entry: dict) -> None:
    """Append one ``{key: entry}`` line and flush it, so runs can resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({key: to_serializable(entry)}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def judge_of(entry: dict) -> dict:
    """Judge labels of an entry (also reads the legacy ``judge_advanced`` key)."""
    return entry.get("judge") or entry.get("judge_advanced") or {}


def is_correct(entry: dict) -> int:
    """Binary correctness label: Correct and Partial count as correct."""
    status = judge_of(entry).get("final_answer", {}).get("parsed", {}).get("final_answer_correct")
    return int(str(status).lower() in {"correct", "partial"})


def edge_class(edge: dict) -> str:
    cls = edge.get("class", "none")
    return EDGE_ALIASES.get(cls, cls)


def cot_steps(entry: dict) -> dict:
    return entry.get("cot", {}).get("parsed", {})
