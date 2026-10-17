"""Structural node/edge features and conversion to PyTorch Geometric graphs."""

from __future__ import annotations

import torch

from .utils import PROBLEM, cot_steps, edge_class, is_correct, judge_of

HALLUCINATION = {"correct": 0, "partial": 1, "hallucinated": 2}
RELEVANCE = {"relevant": 0, "redundant": 1, "irrelevant": 2}


def node_feature_table(matrix: dict, n_steps: int) -> dict[str, dict[str, float]]:
    """All structural features of every node, keyed by node id."""
    n = max(1, n_steps)
    in_deg = {node: 0 for node in matrix}
    for targets in matrix.values():
        for tgt in targets:
            if tgt in in_deg:
                in_deg[tgt] += 1

    table = {}
    for node, targets in matrix.items():
        pos = -1 if node == PROBLEM else int(node)
        d_in, d_out = in_deg[node], len(targets)
        table[node] = {
            "position": pos,
            "position_norm": 0.0 if node == PROBLEM else pos / n,
            "in_degree": d_in / n,
            "out_degree": d_out / n,
            "in_degree_raw": d_in,
            "out_degree_raw": d_out,
            "deg_balance": d_in - d_out,
            "deg_ratio": d_in / (d_out + 1),
        }
    return table


def edge_features(edge: dict, edge_types: list[str], onehot: bool = True) -> list[float]:
    vec = [0.0] * len(edge_types)
    cls = edge_class(edge)
    if onehot and cls in edge_types:
        vec[edge_types.index(cls)] = 1.0
    return vec + [float(edge.get("strength", 0.0)) / 10.0]


def to_pyg(entry: dict, graph_key: str, feat_cfg, uid: str = "", node_labels: bool = False):
    """Convert one entry into a ``torch_geometric.data.Data`` graph."""
    from torch_geometric.data import Data

    matrix = entry[graph_key]["matrix"]
    node_ids = list(matrix)
    index = {nid: i for i, nid in enumerate(node_ids)}
    table = node_feature_table(matrix, len(cot_steps(entry)))
    x = torch.tensor([[table[n][f] for f in feat_cfg.node] for n in node_ids], dtype=torch.float)

    edges, attrs = [], []
    for src, targets in matrix.items():
        for tgt, edge in targets.items():
            if tgt in index:
                edges.append([index[src], index[tgt]])
                attrs.append(edge_features(edge, feat_cfg.edge_types, feat_cfg.edge_type_onehot))
    edge_dim = len(feat_cfg.edge_types) + 1
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_dim), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([is_correct(entry)], dtype=torch.long), uid=uid)
    if node_labels:
        judge = judge_of(entry)
        hall = judge.get("hallucinated", {}).get("parsed", {})
        rel = judge.get("relevance", {}).get("parsed", {})
        labels = []
        for n in node_ids:
            if n == PROBLEM:
                labels.append([0, 0])
            else:
                labels.append([HALLUCINATION.get(hall.get(n, {}).get("label"), 0),
                               RELEVANCE.get(rel.get(n, {}).get("label"), 0)])
        data.y_node = torch.tensor(labels, dtype=torch.long)
    return data, node_ids


def graph_statistics(data, n_edge_types: int) -> list[float]:
    """Graph-level summary statistics for the logistic-regression baseline."""
    n = data.num_nodes
    m = data.edge_index.size(1)
    if m:
        out_deg = torch.bincount(data.edge_index[0], minlength=n).float()
        in_deg = torch.bincount(data.edge_index[1], minlength=n).float()
        type_ratio = (data.edge_attr[:, :n_edge_types].sum(0) / m).tolist()
    else:
        in_deg = out_deg = torch.zeros(n)
        type_ratio = [0.0] * n_edge_types
    std = (lambda t: t.std().item()) if n > 1 else (lambda t: 0.0)
    return [
        n, m, m / max(1, n * (n - 1)), m / max(1, n),
        in_deg.mean().item(), out_deg.mean().item(), std(in_deg), std(out_deg),
        in_deg.max().item(), out_deg.max().item(),
        (in_deg - out_deg).abs().mean().item(), (in_deg / (out_deg + 1)).mean().item(),
        *type_ratio,
    ]
