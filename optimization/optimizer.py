import numpy as np

class Optimizer:
    def __init__(self, graph, w1=0.5, w2=0.5):
        self.graph = graph
        self.w1 = w1
        self.w2 = w2
    
    def optimize(self, traffic_flow, energy_demand):
    
        avg_flow = np.mean(list(traffic_flow.values()))
        avg_energy = np.mean(list(energy_demand.values()))
        
        # measure imbalance
        energy_variance = np.var(list(energy_demand.values()))
        
        # --- ENERGY-AWARE TRAFFIC CONTROL ---
        adjusted_traffic = {}

        for t in self.graph.traffic_nodes:
            v = traffic_flow[t]
            
            connected_energy = [
                nbr for nbr in self.graph.get_neighbors(t)
                if nbr in energy_demand
            ]
            
            overload_factor = sum(
                1 for e in connected_energy if energy_demand[e] > avg_energy
            )
            
            # penalty based on energy stress
            penalty = min(0.2, energy_variance / (avg_energy + 1e-6))
            
            if v > avg_flow:
                reduction = 0.75 - 0.08 * overload_factor + penalty
                reduction = min(0.95, max(0.6, reduction))
            else:
                reduction = 0.9
            
            adjusted_traffic[t] = v * reduction

        # --- REALISTIC ENERGY BALANCING ---
        # --- STRONGER ENERGY BALANCING ---
        adjusted_energy = {}

        for e, v in energy_demand.items():
            # stronger pull toward average
            balanced = 0.5 * v + 0.5 * avg_energy
            
            # slightly reduce peaks
            if v > avg_energy:
                balanced *= 0.9
            
            # small noise
            balanced += np.random.normal(0, 0.01 * avg_energy)
            
            adjusted_energy[e] = max(0, balanced)
        return adjusted_traffic, adjusted_energy