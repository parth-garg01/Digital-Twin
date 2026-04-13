import numpy as np

from config.config import RANDOM_SEED
from optimization.nsga2 import run_nsga2


class Optimizer:
    def __init__(self, graph):
        self.graph = graph

    def optimize(
        self,
        traffic_flow,
        energy_demand,
        w1,
        w2,
        population_size,
        generations,
        traffic_delta_bounds,
        energy_delta_bounds,
    ):
        traffic_nodes = list(traffic_flow.keys())
        energy_nodes = list(energy_demand.keys())
        traffic_values = np.array([traffic_flow[node] for node in traffic_nodes], dtype=np.float32)
        energy_values = np.array([energy_demand[node] for node in energy_nodes], dtype=np.float32)

        decisions, objectives = run_nsga2(
            traffic_flow=traffic_values,
            energy_demand=energy_values,
            population_size=population_size,
            generations=generations,
            seed=RANDOM_SEED,
            traffic_delta_bounds=traffic_delta_bounds,
            energy_delta_bounds=energy_delta_bounds,
        )

        if decisions is None or objectives is None or len(decisions) == 0:
            raise RuntimeError("NSGA-II failed to return candidate solutions")

        normalized = objectives.copy()
        for column in range(normalized.shape[1]):
            col = normalized[:, column]
            span = max(1e-6, float(col.max() - col.min()))
            normalized[:, column] = (col - col.min()) / span

        score = w1 * normalized[:, 0] + w2 * normalized[:, 1]
        best_index = int(np.argmin(score))
        best_decision = decisions[best_index]

        n_traffic = len(traffic_nodes)
        traffic_scale = best_decision[:n_traffic]
        energy_scale = best_decision[n_traffic:]

        adjusted_traffic = {
            node: max(0.0, float(value * scale))
            for node, value, scale in zip(traffic_nodes, traffic_values, traffic_scale)
        }

        avg_energy = float(np.mean(energy_values))
        adjusted_energy = {}
        for node, value, scale in zip(energy_nodes, energy_values, energy_scale):
            centered = value - avg_energy
            adjusted_energy[node] = max(0.0, float(avg_energy + centered * scale))

        pareto_front = [
            {
                "travel_time": float(travel),
                "energy_variance": float(variance),
            }
            for travel, variance in objectives
        ]

        return adjusted_traffic, adjusted_energy, pareto_front
