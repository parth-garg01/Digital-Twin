import numpy as np

class TrafficSimulator:
    def __init__(self, graph):
        self.graph = graph
    
    def generate_flow(self):
        traffic_flow = {}
        
        for node in self.graph.traffic_nodes:
            traffic_flow[node] = np.random.poisson(lam=20)
        
        return traffic_flow


def add_stochasticity(demand_dict):
    return {
        k: max(0, v + np.random.normal(0, 2))
        for k, v in demand_dict.items()
    }