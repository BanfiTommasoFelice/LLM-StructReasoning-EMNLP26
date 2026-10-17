"""Benchmark loading and question formatting."""

from __future__ import annotations

import random

import pandas as pd

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_benchmark(cfg) -> pd.DataFrame:
    """Load the configured benchmark from the Hugging Face Hub."""
    from datasets import load_dataset

    spec = cfg.datasets[cfg.dataset]
    ds = load_dataset(spec.hf_id, spec.hf_config, split=spec.split)
    df = ds.to_pandas()
    df.columns = [c.replace(" ", "") for c in df.columns]

    if cfg.dataset == "MMLU-Pro" and spec.num_questions:
        df = balanced_sample(df, spec.num_questions, cfg.seed)
    elif spec.num_questions:
        df = df.head(spec.num_questions)
    return df.reset_index(drop=True)


def balanced_sample(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    """Sample ``total`` questions spread evenly over the ``category`` column."""
    categories = sorted(df["category"].unique())
    per_cat, remainder = divmod(total, len(categories))
    parts = []
    for i, cat in enumerate(categories):
        cat_df = df[df["category"] == cat]
        n = min(per_cat + (1 if i < remainder else 0), len(cat_df))
        parts.append(cat_df.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed)


def _with_options(question: str, options: list[str]) -> str:
    lines = [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    return f"{question}\n\nOptions:\n" + "\n".join(lines)


def format_question(row: dict, dataset: str, rng: random.Random) -> dict:
    """Turn one benchmark row into the fields stored with each trace."""
    if dataset == "GPQA":
        correct = row["CorrectAnswer"]
        wrong = [row["IncorrectAnswer1"], row["IncorrectAnswer2"], row["IncorrectAnswer3"]]
        options = [correct] + wrong
        rng.shuffle(options)
        return {
            "problem": _with_options(row["Question"], options),
            "solution": row.get("Explanation"),
            "answer": correct,
            "options_raw": {"correct": correct, "opt1": wrong[0], "opt2": wrong[1], "opt3": wrong[2]},
        }
    if dataset == "MMLU-Pro":
        options = list(row["options"])
        return {
            "problem": _with_options(row["question"], options),
            "solution": row.get("cot_content"),
            "answer": row["answer"],
            "answer_index": row.get("answer_index"),
            "options_raw": options,
        }
    raise ValueError(f"Unknown dataset: {dataset}")
