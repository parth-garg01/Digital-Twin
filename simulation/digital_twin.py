from data.real_loader import RealTrafficLoader
from data.simulator import TrafficSimulator, add_stochasticity
from coupling.cross_domain import CrossDomainCoupling
from optimization.optimizer import Optimizer
from evaluation.metrics import compute_travel_time, compute_energy_variance


class DigitalTwin:
    def __init__(self, graph, alpha, w1, w2):
        self.graph = graph

        from data.real_loader import RealTrafficLoader
        self.traffic_sim = RealTrafficLoader("data/metr_la.npz")

        self.coupling = CrossDomainCoupling(graph, alpha)
        self.optimizer = Optimizer(graph, w1, w2)

        from models.stgcn_model import STGCN
        import torch

        # ✅ FIX: initialize model FIRST
        total_nodes = len(graph.traffic_nodes) + len(graph.energy_nodes)
        self.model = STGCN(total_nodes)

        # Graph
        self.edge_index, self.node_to_idx = self.graph.to_edge_index()

        # ✅ Train AFTER model exists
        from models.train_stgcn import train_model
        train_model(self.model, self.traffic_sim, self.edge_index, epochs=3)    
    def run(self, steps=20):
        history = []
        self.model.eval()
        for t in range(steps):
            # 1. Observe
            traffic = self.traffic_sim.get_next()
            import torch

            # Convert dict → tensor
            # Create combined feature vector
            node_values = []

            # Traffic nodes
            for node in self.graph.traffic_nodes:
                node_values.append(traffic[node])

            # Energy nodes (initially 0)
            for node in self.graph.energy_nodes:
                node_values.append(0.0)

            traffic_tensor = torch.tensor(node_values, dtype=torch.float32).unsqueeze(1)

            # TEMP edge index (we fix later)
            edge_index = self.edge_index

            # Run model
            # Run model
            predicted = self.model(traffic_tensor, edge_index)

            # Extract predictions
            traffic_pred = {}
            energy_pred = {}

            for node in self.graph.traffic_nodes:
                idx = self.node_to_idx[node]
                traffic_pred[node] = max(0, float(predicted[idx]))  # ✅ clamp

            for node in self.graph.energy_nodes:
                idx = self.node_to_idx[node]
                energy_pred[node] = max(0, float(predicted[idx]))  # ✅ clamp
            
            # 2. Coupling
            # Combine learned + physical model
            energy_formula = self.coupling.compute_energy_demand(traffic_pred)

            energy = {
                e: max(0, 0.5 * energy_pred[e] + 0.5 * energy_formula[e])
                for e in self.graph.energy_nodes
            }
            traffic_opt, energy_opt = self.optimizer.optimize(traffic_pred, energy)
            
            # 4. Metrics
            travel_time = compute_travel_time(traffic_opt)
            energy_var = compute_energy_variance(energy_opt)
            
            history.append((travel_time, energy_var))
        
        return history