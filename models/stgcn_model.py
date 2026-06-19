import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class STGCN(nn.Module):
    def __init__(self, sequence_length, hidden_dim=32):
        super().__init__()
        # Temporal: 1D conv over the time axis per node
        self.temporal = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        # Spatial: two rounds of graph message passing on the temporal features
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        # x: (num_nodes, sequence_length)
        x = x.unsqueeze(1)                           # (num_nodes, 1, sequence_length)
        x = self.activation(self.temporal(x))         # (num_nodes, hidden_dim, sequence_length)
        x = x.mean(dim=-1)                            # (num_nodes, hidden_dim) — temporal pooling
        x = self.activation(self.gcn1(x, edge_index)) # (num_nodes, hidden_dim)
        x = self.activation(self.gcn2(x, edge_index)) # (num_nodes, hidden_dim)
        return self.output(x)                          # (num_nodes, 1)
