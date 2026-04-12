import networkx as nx
import numpy as np


class SmartCityGraph:
    def __init__(self, num_traffic, num_energy, seed=7):
        self.num_traffic = num_traffic
        self.num_energy = num_energy
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.G = nx.Graph()

        self.traffic_nodes = []
        self.energy_nodes = []

        self._build_graph()

    def _build_graph(self):
        for i in range(self.num_traffic):
            node_id = f"T{i}"
            self.G.add_node(node_id, type="traffic")
            self.traffic_nodes.append(node_id)

        for i in range(self.num_energy):
            node_id = f"E{i}"
            self.G.add_node(node_id, type="energy")
            self.energy_nodes.append(node_id)

        for i in range(self.num_traffic - 1):
            self.G.add_edge(self.traffic_nodes[i], self.traffic_nodes[i + 1], type="traffic")

        extra_edges = max(1, self.num_traffic // 2)
        for _ in range(extra_edges):
            a, b = self.rng.choice(self.traffic_nodes, 2, replace=False)
            self.G.add_edge(a, b, type="traffic")

        for i in range(self.num_energy - 1):
            self.G.add_edge(self.energy_nodes[i], self.energy_nodes[i + 1], type="energy")

        for index, energy_node in enumerate(self.energy_nodes):
            anchor = index % len(self.traffic_nodes)
            traffic_choices = self.traffic_nodes[anchor:] + self.traffic_nodes[:anchor]
            for traffic_node in traffic_choices[: min(3, len(traffic_choices))]:
                self.G.add_edge(traffic_node, energy_node, type="cross")

    def get_neighbors(self, node, edge_type=None):
        neighbors = []
        for neighbor in self.G.neighbors(node):
            if edge_type is None or self.G[node][neighbor]["type"] == edge_type:
                neighbors.append(neighbor)
        return neighbors

    def get_node_list(self, include_energy=True):
        return self.traffic_nodes + (self.energy_nodes if include_energy else [])

    def to_edge_index(self, include_energy=True):
        import torch

        node_list = self.get_node_list(include_energy=include_energy)
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        edges = []

        for u, v, _ in self.G.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                edges.append([node_to_idx[u], node_to_idx[v]])
                edges.append([node_to_idx[v], node_to_idx[u]])

        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return edge_index, node_to_idx
