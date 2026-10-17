"""Graph-level correctness prediction and GNNExplainer attributions (Sections 4.2-4.3)."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from .config import run_dir, stage_file
from .features import graph_statistics, to_pyg
from .models import build_graph_model
from .utils import cot_steps, read_json, set_seed, write_json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_entries(cfg, runs) -> dict:
    """Entries of the requested runs, keyed ``"<run>_<qid>"``."""
    entries = {}
    for run_id in runs:
        for qid, entry in read_json(stage_file(cfg, "graphs", run_id)).items():
            entries[f"{run_id}_{qid}"] = entry
    return entries


def balanced_split(labels, test_size, val_ratio, gamma, tolerance, rng):
    """Random train/val/test split with soft down-sampling of the training majority class."""
    idx = list(range(len(labels)))
    rng.shuffle(idx)
    n_train = len(idx) - test_size
    train_full, test = idx[:n_train], idx[n_train:]
    n_val = int(n_train * val_ratio)
    val, train = train_full[:n_val], train_full[n_val:]

    by_class = defaultdict(list)
    for i in train:
        by_class[labels[i]].append(i)
    if len(by_class) < 2:
        return train, val, test
    minority, majority = sorted(by_class.values(), key=len)
    m, big = len(minority), len(majority)
    gap = big - m
    keep = big
    if gap > tolerance:
        excess = gap - tolerance
        keep = max(big - int(excess * (excess / gap) ** gamma), m + tolerance)
    train = minority + rng.sample(majority, keep)
    rng.shuffle(train)
    return train, val, test


def binary_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per_class = {c: accuracy_score(y_true[y_true == c], y_pred[y_true == c]) if (y_true == c).any() else np.nan
                 for c in (0, 1)}
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "acc_incorrect": per_class[0],
        "acc_correct": per_class[1],
    }


def evaluate(model, loader):
    model.eval()
    preds, gts, uids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds += out.argmax(1).cpu().tolist()
            gts += batch.y.cpu().tolist()
            uids += list(batch.uid)
    return preds, gts, uids


def train_model(cfg, train_graphs, val_graphs, ckpt_path):
    """Train one GATv2 model; early stopping keeps the best validation F1."""
    from torch_geometric.loader import DataLoader

    g = cfg.gnn
    model = build_graph_model(cfg, train_graphs[0].x.size(1), train_graphs[0].edge_attr.size(1)).to(DEVICE)
    y_train = [int(d.y) for d in train_graphs]
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train) ** g.class_weight_alpha
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=g.lr, weight_decay=g.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=g.lr_step, gamma=g.lr_gamma)
    train_loader = DataLoader(train_graphs, batch_size=g.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=g.batch_size)

    best_f1, since_best = -1.0, 0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, g.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch), batch.y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % g.eval_every == 0 or epoch == g.epochs:
            preds, gts, _ = evaluate(model, val_loader)
            val_f1 = f1_score(gts, preds, zero_division=0)
            if val_f1 > best_f1:
                best_f1, since_best = val_f1, 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                since_best += g.eval_every
            if since_best >= g.patience:
                break
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


def explain(model, graph, epochs: int) -> list[float]:
    """GNNExplainer node mask for one graph (node masks only, Eq. 4)."""
    from torch_geometric.explain import Explainer, GNNExplainer

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type=None,
        model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"),
    )
    graph = graph.clone().to(DEVICE)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)
    mask = explainer(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, batch=batch).node_mask
    return mask.detach().cpu().flatten().tolist()


def summarize(name: str, rows: list[dict]) -> dict:
    keys = rows[0].keys()
    summary = {k: (float(np.nanmean([r[k] for r in rows])), float(np.nanstd([r[k] for r in rows]))) for k in keys}
    print(f"{name:<18}" + "".join(f"{k} {m * 100:5.1f} ± {s * 100:4.1f}   " for k, (m, s) in summary.items()))
    return summary


def run(cfg) -> None:
    from torch_geometric.loader import DataLoader

    g = cfg.gnn
    set_seed(g.seed, deterministic=True)
    rng = random.Random(g.seed)
    entries = load_entries(cfg, g.runs)

    graphs, node_ids = [], {}
    for uid, entry in entries.items():
        data, ids = to_pyg(entry, g.graph, cfg.features, uid=uid)
        graphs.append(data)
        node_ids[uid] = ids
    labels = [int(d.y) for d in graphs]
    lengths = np.array([[len(cot_steps(e))] for e in entries.values()])
    stats = np.array([graph_statistics(d, len(cfg.features.edge_types)) for d in graphs])
    y = np.array(labels)
    print(f"{len(graphs)} graphs ({g.graph}) | labels {dict(Counter(labels))} | device {DEVICE}")

    splits = [balanced_split(labels, g.test_size, g.val_ratio, g.balance_gamma, g.balance_tolerance, rng)
              for _ in range(g.num_splits)]

    ckpt_dir = run_dir(cfg) / "checkpoints" / g.graph
    results = {"CoT2Graph" if g.graph == "matrix_compressed" else g.graph: [], "LR (length)": [], "LR (graph stats)": []}
    split_preds, masks = defaultdict(list), defaultdict(list)
    for sid, (train, val, test) in enumerate(splits):
        model = train_model(cfg, [graphs[i] for i in train], [graphs[i] for i in val], ckpt_dir / f"split{sid}.pt")
        preds, gts, uids = evaluate(model, DataLoader([graphs[i] for i in test], batch_size=g.batch_size))
        results[next(iter(results))].append(binary_metrics(gts, preds))
        for uid, p in zip(uids, preds):
            split_preds[uid].append(int(p))
        if g.explain:
            for i in test:
                masks[graphs[i].uid].append(explain(model, graphs[i], g.explainer_epochs))

        for name, X in (("LR (length)", lengths), ("LR (graph stats)", stats)):
            clf = LogisticRegression(max_iter=3000).fit(X[train], y[train])
            results[name].append(binary_metrics(y[test], clf.predict(X[test])))
        print(f"split {sid + 1}/{g.num_splits}: acc {results[next(iter(results))][-1]['acc']:.3f}")

    print(f"\nTest metrics over {g.num_splits} splits (mean ± std, %):")
    summary = {name: summarize(name, rows) for name, rows in results.items()}

    predictions = {}
    for uid, preds in split_preds.items():
        record = {"split_predictions": preds, "majority_vote": Counter(preds).most_common(1)[0][0]}
        if masks.get(uid):
            mean_mask = np.mean(masks[uid], axis=0)
            record["node_importance"] = dict(zip(node_ids[uid], mean_mask.tolist()))
        predictions[uid] = record

    out_dir = run_dir(cfg) / "predictions"
    write_json(predictions, out_dir / f"{g.graph}.json")
    write_json(summary, out_dir / f"{g.graph}_metrics.json")
    print(f"Saved predictions and metrics to {out_dir}")
