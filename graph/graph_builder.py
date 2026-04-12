import networkx as nx
import numpy as np

class SmartCityGraph:
    def __init__(self, num_traffic, num_energy):
        self.num_traffic = num_traffic
        self.num_energy = num_energy
        self.G = nx.Graph()
        
        self.traffic_nodes = []
        self.energy_nodes = []
        
        self._build_graph()
    
    def _build_graph(self):
        # Traffic nodes
        for i in range(self.num_traffic):
            node_id = f"T{i}"
            self.G.add_node(node_id, type='traffic')
            self.traffic_nodes.append(node_id)
        
        # Energy nodes
        for i in range(self.num_energy):
            node_id = f"E{i}"
            self.G.add_node(node_id, type='energy')
            self.energy_nodes.append(node_id)
        
        # Traffic edges
        for i in range(self.num_traffic - 1):
            self.G.add_edge(self.traffic_nodes[i], self.traffic_nodes[i+1], type='traffic')
        
        for _ in range(self.num_traffic):
            a, b = np.random.choice(self.traffic_nodes, 2, replace=False)
            self.G.add_edge(a, b, type='traffic')
        
        # Energy edges
        for i in range(self.num_energy - 1):
            self.G.add_edge(self.energy_nodes[i], self.energy_nodes[i+1], type='energy')
        
        # Cross-domain edges
        for e in self.energy_nodes:
            connected_traffic = np.random.choice(self.traffic_nodes, size=3, replace=False)
            for t in connected_traffic:
                self.G.add_edge(t, e, type='cross')
    
    def get_neighbors(self, node, edge_type=None):
        neighbors = []
        for nbr in self.G.neighbors(node):
            if edge_type:
                if self.G[node][nbr]['type'] == edge_type:
                    neighbors.append(nbr)
            else:
                neighbors.append(nbr)
        return neighbors
    def to_edge_index(self):
        import torch

        # ✅ include BOTH traffic + energy
        node_list = self.traffic_nodes + self.energy_nodes
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        edges = []

        for u, v, data in self.G.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                edges.append([node_to_idx[u], node_to_idx[v]])
                edges.append([node_to_idx[v], node_to_idx[u]])

        edge_index = torch.tensor(edges, dtype=torch.long).t()

        return edge_index, node_to_idx