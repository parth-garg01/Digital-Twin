import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class STGCN(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.gcn = GCNConv(1, 16)
        self.temporal = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x, edge_index):
        # x: (nodes, features)
        x = self.gcn(x, edge_index)
        x = x.unsqueeze(0).permute(0,2,1)  # for conv1d
        x = self.temporal(x)
        x = x.permute(0,2,1).squeeze(0)
        return self.fc(x)