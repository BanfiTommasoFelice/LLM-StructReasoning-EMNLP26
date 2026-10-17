"""What separates correct from failed reasoning: GNNExplainer analysis (Section 5.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .config import run_dir
from .features import node_feature_table
from .train_graph import load_entries
from .utils import cot_steps, is_correct, read_json

LABELS = {
    "position": "Absolute position",
    "position_norm": "Normalized position",
    "in_degree": "In-degree",
    "out_degree": "Out-degree",
    "deg_balance": "Degree balance",
    "deg_ratio": "Degree ratio",
}


def node_table(entries: dict, predictions: dict, graph_key: str) -> pd.DataFrame:
    rows = []
    for uid, entry in entries.items():
        pred = predictions.get(uid, {})
        importance = pred.get("node_importance")
        if not importance:
            continue
        mean_mask = np.mean(list(importance.values())) or 1.0
        truth, guess = is_correct(entry), pred["majority_vote"]
        outcome = {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(truth, guess)]
        table = node_feature_table(entry[graph_key]["matrix"], len(cot_steps(entry)))
        for node, value in importance.items():
            feats = dict(table[node])
            feats["in_degree"], feats["out_degree"] = feats.pop("in_degree_raw"), feats.pop("out_degree_raw")
            rows.append({"uid": uid, "node": node, "importance": value / mean_mask,
                         "outcome": outcome, "correct": bool(truth), **feats})
    return pd.DataFrame(rows)


def feature_contributions(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Share (%) of |standardised coefficient| per feature, for correct and incorrect traces."""
    out = {}
    for name, group in (("correct", df[df.correct]), ("incorrect", df[~df.correct])):
        X = StandardScaler().fit_transform(group[features].values)
        coef = np.abs(LinearRegression().fit(X, group["importance"].values).coef_)
        out[name] = coef / coef.sum() * 100
    return pd.DataFrame(out, index=features)


def plot(df: pd.DataFrame, contrib: pd.DataFrame, out_dir) -> None:
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    from .plotting import COLORS, save, setup

    setup()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for outcome, color, label in (("TP", COLORS["correct"], "Correct (TP)"), ("TN", COLORS["incorrect"], "Incorrect (TN)")):
        sub = df[(df.outcome == outcome) & (df.node != "problem")]
        if len(sub) < 5:
            continue
        ax.scatter(sub.position_norm, sub.importance, s=8, alpha=0.2, color=color)
        trend = lowess(sub.importance, sub.position_norm, frac=0.35)
        ax.plot(trend[:, 0], trend[:, 1], color=color, lw=3, label=label)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set(xlabel="Normalized step position", ylabel="Normalized node importance")
    ax.legend()
    save(fig, out_dir / "importance_vs_position.pdf")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(contrib))
    for offset, col, color in ((-0.18, "correct", COLORS["correct"]), (0.18, "incorrect", COLORS["incorrect"])):
        bars = ax.bar(x + offset, contrib[col], 0.36, color=color, label=col.capitalize())
        ax.bar_label(bars, fmt="%.1f%%", fontsize=7)
    ax.set_xticks(x, [LABELS.get(f, f) for f in contrib.index], rotation=25, ha="right")
    ax.set_ylabel("Contribution (%)")
    ax.legend()
    save(fig, out_dir / "feature_contributions.pdf")


def run(cfg) -> None:
    graph_key = cfg.explainer_analysis.graph
    predictions = read_json(run_dir(cfg) / "predictions" / f"{graph_key}.json")
    df = node_table(load_entries(cfg, cfg.gnn.runs), predictions, graph_key)
    if df.empty:
        raise SystemExit("No node importances found: run `train-graph` with gnn.explain=true first.")

    contrib = feature_contributions(df, cfg.explainer_analysis.regression_features)
    print("Feature contribution to node importance (%):")
    print(contrib.round(1).rename(index=LABELS).to_string())

    out_dir = run_dir(cfg) / "explainer"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "nodes.csv", index=False)
    contrib.to_csv(out_dir / "feature_contributions.csv")
    plot(df, contrib, out_dir)
    print(f"Saved tables and plots to {out_dir}")
