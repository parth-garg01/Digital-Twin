class CrossDomainCoupling:
    def __init__(self, graph, alpha=0.3):
        self.graph = graph
        self.alpha = alpha
    
    def compute_energy_demand(self, traffic_flow):
        energy_demand = {}
        
        for e in self.graph.energy_nodes:
            neighbors = self.graph.get_neighbors(e, edge_type='cross')
            total_flow = sum(traffic_flow[t] for t in neighbors)
            
            D_e = max(0.1, self.alpha) * total_flow
            energy_demand[e] = D_e
        
        return energy_demand