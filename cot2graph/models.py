"""GNN architectures (Section 4.2, Appendix A)."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, global_mean_pool


class GATv2Classifier(nn.Module):
    """Graph-level correctness predictor."""

    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=2, heads=1, dropout=0.2, num_classes=2):
        super().__init__()
        dims = [node_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            GATv2Conv(dims[i], hidden_dim, edge_dim=edge_dim, heads=heads, concat=False)
            for i in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
        return self.head(global_mean_pool(x, batch))


class GINENodeClassifier(nn.Module):
    """Node-level predictor with hallucination and relevance heads (3 classes each)."""

    def __init__(self, node_dim, edge_dim, hidden_dim=128):
        super().__init__()

        def mlp(d_in):
            return nn.Sequential(nn.Linear(d_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        self.conv1 = GINEConv(mlp(node_dim), edge_dim=edge_dim)
        self.conv2 = GINEConv(mlp(hidden_dim), edge_dim=edge_dim)
        self.head_hallucinated = nn.Linear(hidden_dim, 3)
        self.head_relevance = nn.Linear(hidden_dim, 3)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return self.head_hallucinated(x), self.head_relevance(x)


def build_graph_model(cfg, node_dim: int, edge_dim: int) -> GATv2Classifier:
    g = cfg.gnn
    return GATv2Classifier(node_dim, edge_dim, g.hidden_dim, g.num_layers, g.heads, g.dropout)
