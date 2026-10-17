"""Node-level prediction of relevance and hallucination labels from structure (Section 5.2)."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from .config import run_dir
from .features import HALLUCINATION, RELEVANCE, to_pyg
from .models import GINENodeClassifier
from .train_graph import DEVICE, load_entries
from .utils import set_seed, write_json

TASKS = {"hallucinated": (0, HALLUCINATION), "relevance": (1, RELEVANCE)}


def permute_nodes(batch):
    perm = torch.randperm(batch.num_nodes, device=batch.x.device)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(batch.num_nodes, device=batch.x.device)
    batch.x, batch.y_node = batch.x[perm], batch.y_node[perm]
    batch.edge_index = inv[batch.edge_index]
    return batch


def train_split(cfg, train_graphs):
    from torch_geometric.loader import DataLoader

    c = cfg.node_gnn
    model = GINENodeClassifier(train_graphs[0].x.size(1), train_graphs[0].edge_attr.size(1), c.hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs, eta_min=c.eta_min)
    loader = DataLoader(train_graphs, batch_size=c.batch_size, shuffle=True)
    model.train()
    for _ in range(c.epochs):
        for batch in loader:
            batch = batch.to(DEVICE)
            if c.node_permute_prob > 0 and random.random() < c.node_permute_prob:
                batch = permute_nodes(batch)
            hall, rel = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = F.cross_entropy(hall, batch.y_node[:, 0]) + F.cross_entropy(rel, batch.y_node[:, 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def predict(model, graphs):
    model.eval()
    preds = {task: [] for task in TASKS}
    with torch.no_grad():
        for g in graphs:
            g = g.clone().to(DEVICE)
            for task, logits in zip(TASKS, model(g.x, g.edge_index, g.edge_attr)):
                preds[task] += logits.argmax(1).cpu().tolist()
    return preds


def report(name: str, y_true: dict, y_pred: dict) -> dict:
    out = {}
    for task, (_, classes) in TASKS.items():
        cm = confusion_matrix(y_true[task], y_pred[task], labels=list(classes.values()))
        per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
        out[task] = {
            "macro_f1": float(f1_score(y_true[task], y_pred[task], average="macro")),
            "per_class_accuracy": dict(zip(classes, per_class.round(4).tolist())),
        }
        accs = "  ".join(f"{k} {v * 100:.1f}%" for k, v in out[task]["per_class_accuracy"].items())
        print(f"{name:<6} {task:<13} macro-F1 {out[task]['macro_f1']:.3f} | per-class accuracy: {accs}")
    return out


def run(cfg) -> None:
    c = cfg.node_gnn
    set_seed(c.seed)
    entries = load_entries(cfg, cfg.gnn.runs)
    graphs = [to_pyg(e, c.graph, cfg.features, uid=uid, node_labels=True)[0] for uid, e in entries.items()]

    indices = list(range(len(graphs)))
    random.shuffle(indices)
    splits = []
    for _ in range(c.num_splits):
        random.shuffle(indices)
        splits.append((indices[: c.train_size], indices[c.train_size : c.train_size + c.test_size]))

    truth = {name: {t: [] for t in TASKS} for name in ("GNN", "LR")}
    preds = {name: {t: [] for t in TASKS} for name in ("GNN", "LR")}
    for sid, (train, test) in enumerate(splits):
        test_graphs = [graphs[i] for i in test]
        model = train_split(cfg, [graphs[i] for i in train])
        gnn_pred = predict(model, test_graphs)

        x_train = np.vstack([graphs[i].x.numpy() for i in train])
        x_test = np.vstack([g.x.numpy() for g in test_graphs])
        for task, (col, _) in TASKS.items():
            y_train = np.concatenate([graphs[i].y_node[:, col].numpy() for i in train])
            y_test = np.concatenate([g.y_node[:, col].numpy() for g in test_graphs])
            lr = LogisticRegression(max_iter=1000).fit(x_train, y_train)
            for name, p in (("GNN", gnn_pred[task]), ("LR", lr.predict(x_test).tolist())):
                truth[name][task] += y_test.tolist()
                preds[name][task] += list(p)
        print(f"split {sid + 1}/{c.num_splits} done")

    print(f"\nNode-level results pooled over {c.num_splits} splits:")
    results = {name: report(name, truth[name], preds[name]) for name in ("GNN", "LR")}
    out = run_dir(cfg) / "predictions" / "node_level_metrics.json"
    write_json(results, out)
    print(f"Saved metrics to {out}")
