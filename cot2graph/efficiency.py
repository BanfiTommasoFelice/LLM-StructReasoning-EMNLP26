"""Efficiency decomposition of long CoTs (Section 5.4, Appendix F)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .config import run_dir, stage_file
from .utils import PROBLEM, cot_steps, edge_class, is_correct, judge_of, node_key, read_json

IRRELEVANT = {"redundant", "irrelevant"}


def drop_irrelevant(matrix: dict, relevance: dict) -> dict:
    """Remove steps the judge marked as redundant or irrelevant."""
    removed = {n for n, info in relevance.items() if info.get("label") in IRRELEVANT}
    return {src: {t: e for t, e in tgts.items() if t not in removed}
            for src, tgts in matrix.items() if src not in removed}


def prerequisites(graph: dict, node: str, types: set) -> list[str]:
    return [src for src, tgts in graph.items() if node in tgts and edge_class(tgts[node]) in types]


def tsp_traversal(graph: dict, types: list[str], relaxation_order: list[str]) -> tuple[list[str], int]:
    """Visit each step only once its prerequisites are visited."""
    graph = copy.deepcopy(graph)
    types = set(types)
    path, visited = [PROBLEM], {PROBLEM}
    remaining = set(graph) - {PROBLEM}
    relaxations = 0
    while remaining:
        available = [n for n in remaining if all(p in visited for p in prerequisites(graph, n, types))]
        if available:
            node = min(available, key=node_key)
        else:
            blocked = sorted(((n, prerequisites(graph, n, types)) for n in remaining), key=lambda x: len(x[1]))
            node = None
            for candidate, prereqs in blocked:
                for dropped in relaxation_order:
                    relaxations += 1
                    prereqs = [p for p in prereqs if edge_class(graph[p][candidate]) != dropped]
                    if all(p in visited for p in prereqs):
                        node = candidate
                        break
                if node is not None:
                    break
            if node is None:
                raise ValueError("No dependency-consistent traversal exists.")
        path.append(node)
        visited.add(node)
        remaining.remove(node)
    return path, relaxations


def trace_metrics(entry: dict, cfg) -> tuple[dict, list[dict]]:
    eff = cfg.efficiency
    graph = entry["matrix_compressed"]
    matrix, rep = graph["matrix"], graph["representative"]
    judge = judge_of(entry)
    hallucination = judge.get("hallucinated", {}).get("parsed", {})
    steps = list(cot_steps(entry))

    seen, noloop = set(), []
    for s in steps:
        r = rep.get(s, s)
        if r not in seen:
            seen.add(r)
            noloop.append(r)
    path, relaxations = tsp_traversal(
        drop_irrelevant(matrix, judge.get("relevance", {}).get("parsed", {})),
        eff.prerequisite_types, eff.relaxation_order,
    )
    l_cot, l_loops, l_tsp = len(steps) + 1, len(noloop) + 1, len(path)

    types, visited, skips = set(eff.prerequisite_types), {PROBLEM}, []
    for pos, s in enumerate(steps):
        r = rep.get(s, s)
        missing = [p for p in prerequisites(matrix, r, types) if p not in visited]
        skips.append({"position_norm": pos / max(1, len(steps) - 1),
                      "hallucination": hallucination.get(s, {}).get("label", "unknown"),
                      "skipped": bool(missing)})
        visited.add(r)

    row = {
        "L_CoT": l_cot, "L_loops": l_loops, "L_TSP": l_tsp, "relaxations": relaxations,
        "efficient": l_tsp / l_cot,
        "verification": (l_cot - l_loops) / l_cot,
        "inefficiency": (l_loops - l_tsp) / l_cot,
        "skip_rate": float(np.mean([s["skipped"] for s in skips])) if skips else 0.0,
        "correct": bool(is_correct(entry)),
    }
    return row, skips


def plot(df: pd.DataFrame, steps: pd.DataFrame, out_dir) -> None:
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    from .plotting import COLORS, save, setup

    setup()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, col, title in zip(axes, ["L_loops", "L_TSP"], ["Loops collapsed", "Minimal valid path (TSP*)"]):
        ratio = df[col] / df["L_CoT"]
        for flag, name in ((True, "correct"), (False, "incorrect")):
            sub = df["correct"] == flag
            ax.scatter(df.loc[sub, "L_CoT"], ratio[sub], s=14, alpha=0.4, color=COLORS[name], label=name.capitalize())
        trend = lowess(ratio, df["L_CoT"], frac=0.5)
        ax.plot(trend[:, 0], trend[:, 1], color=COLORS["trend"], lw=2)
        ax.set(title=title, xlabel="L_CoT (steps)", ylim=(0, 1.05))
    axes[0].set_ylabel("Length / L_CoT")
    axes[0].legend()
    save(fig, out_dir / "length_ratios.pdf")

    shares = df.groupby("length_bin", observed=True)[["efficient", "verification", "inefficiency"]].mean()
    shares.loc["all"] = df[["efficient", "verification", "inefficiency"]].mean()
    fig, ax = plt.subplots(figsize=(6, 3))
    left = np.zeros(len(shares))
    for col, color, label in [("efficient", COLORS["neutral"], "Efficient (TSP*)"),
                              ("verification", COLORS["correct"], "Exploration / verification"),
                              ("inefficiency", COLORS["incorrect"], "Path inefficiency")]:
        ax.barh(shares.index.astype(str), shares[col] * 100, left=left, color=color, label=label)
        left += shares[col].values * 100
    ax.invert_yaxis()
    ax.set_xlabel("Share of L_CoT (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=3, frameon=False)
    save(fig, out_dir / "decomposition.pdf")

    rates = steps.groupby("hallucination")["skipped"].mean().reindex(["correct", "partial", "hallucinated"])
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(rates.index, rates.values * 100, color=[COLORS["correct"], COLORS["amber"], COLORS["incorrect"]])
    ax.set_ylabel("Steps with a skipped prerequisite (%)")
    save(fig, out_dir / "skips_by_hallucination.pdf")


def run(cfg) -> None:
    rows, all_steps = [], []
    for qid, entry in read_json(stage_file(cfg, "graphs", 0)).items():
        row, steps = trace_metrics(entry, cfg)
        rows.append({"id": qid, **row})
        all_steps += steps
    df, steps = pd.DataFrame(rows), pd.DataFrame(all_steps)
    df["length_bin"] = pd.qcut(df["L_CoT"], len(cfg.efficiency.length_bins), labels=cfg.efficiency.length_bins)

    cols = ["efficient", "verification", "inefficiency"]
    print("Share of L_CoT by length bin (%):")
    print((df.groupby("length_bin", observed=True)[cols].mean() * 100).round(1).to_string())
    print(f"all     {' '.join(f'{df[c].mean() * 100:5.1f}' for c in cols)}")
    print("\nSteps with a skipped prerequisite, by hallucination label (%):")
    print((steps.groupby("hallucination")["skipped"].mean() * 100).round(2).to_string())

    out_dir = run_dir(cfg) / "efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "traces.csv", index=False)
    plot(df, steps, out_dir)
    print(f"Saved metrics and plots to {out_dir}")
