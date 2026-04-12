import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class STGCN(nn.Module):
    def __init__(self, sequence_length, hidden_dim=32):
        super().__init__()
        self.gcn1 = GCNConv(sequence_length, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.activation(self.gcn1(x, edge_index))
        x = self.activation(self.gcn2(x, edge_index))
        return self.output(x)
