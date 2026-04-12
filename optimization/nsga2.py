import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


class TrafficEnergyProblem(Problem):
    def __init__(self, traffic_flow, energy_demand, max_traffic_reduction, max_energy_shift):
        self.traffic_flow = np.asarray(traffic_flow, dtype=np.float32)
        self.energy_demand = np.asarray(energy_demand, dtype=np.float32)
        self.avg_energy = float(np.mean(self.energy_demand))

        n_traffic = len(self.traffic_flow)
        n_energy = len(self.energy_demand)
        lower = np.concatenate(
            [
                np.full(n_traffic, 1.0 - max_traffic_reduction, dtype=np.float32),
                np.full(n_energy, 1.0 - max_energy_shift, dtype=np.float32),
            ]
        )
        upper = np.concatenate(
            [
                np.ones(n_traffic, dtype=np.float32),
                np.full(n_energy, 1.0 + max_energy_shift, dtype=np.float32),
            ]
        )

        super().__init__(n_var=n_traffic + n_energy, n_obj=2, xl=lower, xu=upper)

    def _evaluate(self, x, out, *args, **kwargs):
        n_traffic = len(self.traffic_flow)
        traffic_scale = x[:, :n_traffic]
        energy_scale = x[:, n_traffic:]

        adjusted_traffic = traffic_scale * self.traffic_flow
        centered_energy = self.energy_demand - self.avg_energy
        adjusted_energy = np.clip(self.avg_energy + centered_energy * energy_scale, 0.0, None)

        travel_time = adjusted_traffic.mean(axis=1)
        energy_variance = adjusted_energy.var(axis=1)
        out["F"] = np.column_stack([travel_time, energy_variance])


def run_nsga2(
    traffic_flow,
    energy_demand,
    population_size,
    generations,
    seed,
    max_traffic_reduction,
    max_energy_shift,
):
    problem = TrafficEnergyProblem(
        traffic_flow=traffic_flow,
        energy_demand=energy_demand,
        max_traffic_reduction=max_traffic_reduction,
        max_energy_shift=max_energy_shift,
    )
    algorithm = NSGA2(pop_size=population_size)
    result = minimize(problem, algorithm, ("n_gen", generations), seed=seed, verbose=False)
    return result.X, result.F
