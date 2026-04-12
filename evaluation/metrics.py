import numpy as np

def compute_travel_time(traffic_flow):
    return np.mean(list(traffic_flow.values()))

def compute_energy_variance(energy_demand):
    return np.var(list(energy_demand.values()))