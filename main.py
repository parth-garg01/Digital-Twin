import json
from pathlib import Path

import numpy as np

from config.config import ALPHA, ALPHA_VALUES, NUM_ENERGY_NODES, NUM_TRAFFIC_NODES, W1, W2, get_runtime_settings, is_dev_mode
from graph.graph_builder import SmartCityGraph
from simulation.digital_twin import ExperimentContext
from utils.helpers import ProgressTracker, StageTimer


def summarize_result(simulation_output, optimization_output):
    raw_history = simulation_output["raw_history"]
    return {
        "pre_optimization": {
            "travel_time": float(np.mean([item["travel_time"] for item in raw_history])),
            "energy_variance": float(np.mean([item["energy_variance"] for item in raw_history])),
        },
        "post_optimization": {
            "travel_time": optimization_output["travel_time"],
            "energy_variance": optimization_output["energy_variance"],
        },
        "pareto_front": optimization_output["pareto_front"],
        "losses": simulation_output["losses"],
        "diagnostics": simulation_output["diagnostics"],
    }


def validate_results(results):
    scenario_results = results["scenarios"]
    cross_domain = scenario_results["Cross-domain"]["post_optimization"]
    traffic_only = scenario_results["Traffic-only"]["post_optimization"]

    for name, values in scenario_results.items():
        metrics = values["post_optimization"]
        if metrics["travel_time"] < 0 or metrics["energy_variance"] < 0:
            raise ValueError(f"{name} produced negative metrics")
        if np.isnan(metrics["travel_time"]) or np.isnan(metrics["energy_variance"]):
            raise ValueError(f"{name} produced NaN metrics")

        for loss in values["losses"]:
            if np.isnan(loss):
                raise ValueError(f"{name} produced NaN training loss")

        for item in values["diagnostics"]:
            if item["traffic_min"] < 0 or item["energy_min"] < 0:
                raise ValueError(f"{name} produced negative traffic or energy")

    if not cross_domain["energy_variance"] > 0:
        raise ValueError("Cross-domain energy variance should be positive")

    return {
        "cross_domain_beats_traffic_on_travel": cross_domain["travel_time"] < traffic_only["travel_time"],
        "cross_domain_has_higher_energy_variance": cross_domain["energy_variance"] > traffic_only["energy_variance"],
        "all_losses_decrease": {
            name: values["losses"][-1] <= values["losses"][0]
            for name, values in scenario_results.items()
        },
    }


def main():
    runtime = get_runtime_settings()
    graph = SmartCityGraph(NUM_TRAFFIC_NODES, NUM_ENERGY_NODES)
    context = ExperimentContext(graph)
    timer = StageTimer()

    # All scenarios use identical optimizer settings (w1, w2, bounds) so that
    # differences in results are attributable only to the model and input data,
    # not to the optimization objective or search range.
    scenario_specs = {
        "Traffic-only": {
            "model_kind": "stgcn",
            "alpha": ALPHA,
            "w1": W1,
            "w2": W2,
            "traffic_delta_bounds": (-0.2, 0.2),
            "energy_delta_bounds": (-0.2, 0.2),
        },
        "Energy-only": {
            "model_kind": "cross_domain",
            "alpha": ALPHA,
            "w1": W1,
            "w2": W2,
            "traffic_delta_bounds": (-0.2, 0.2),
            "energy_delta_bounds": (-0.2, 0.2),
        },
        "Cross-domain": {
            "model_kind": "cross_domain",
            "alpha": ALPHA,
            "w1": W1,
            "w2": W2,
            "traffic_delta_bounds": (-0.2, 0.2),
            "energy_delta_bounds": (-0.2, 0.2),
        },
        "Ablation": {
            "model_kind": "cross_domain",
            "alpha": ALPHA,
            "w1": W1,
            "w2": W2,
            "traffic_delta_bounds": (-0.2, 0.2),
            "energy_delta_bounds": (-0.2, 0.2),
        },
    }
    benchmark_specs = {
        "LSTM": {
            "model_kind": "lstm",
            "scenario_name": "Traffic-only",
            "alpha": ALPHA,
            "w1": W1,
            "w2": W2,
            "traffic_delta_bounds": (-0.1, 0.1),
            "energy_delta_bounds": (-0.1, 0.1),
        },
        "STGCN": {
            "model_kind": "stgcn",
            "scenario_name": "Traffic-only",
            "alpha": ALPHA,
            "w1": W1,
            "w2": W2,
            "traffic_delta_bounds": (-0.1, 0.1),
            "energy_delta_bounds": (-0.1, 0.1),
        },
        "Cross-domain GNN": {
            "model_kind": "cross_domain",
            "scenario_name": "Cross-domain",
            "alpha": ALPHA,
            "w1": W1,
            "w2": W2,
            "traffic_delta_bounds": (-0.2, 0.2),
            "energy_delta_bounds": (-0.5, 0.5),
        },
    }

    total_progress_steps = 3 + (len(scenario_specs) * 2) + (len(ALPHA_VALUES) * 2) + (len(benchmark_specs) * 2) + 1
    progress = ProgressTracker(total_progress_steps)

    trained_models = {}
    for model_kind in ("lstm", "stgcn", "cross_domain"):
        progress.advance("Training model", f"(Model: {model_kind})")
        with timer.time_block(f"train_{model_kind}"):
            trained_models[model_kind] = context.train_model(model_kind)[1]

    scenarios = {}
    for name, spec in scenario_specs.items():
        progress.advance("Running simulation", f"(Scenario: {name})")
        with timer.time_block(f"simulate_{name}"):
            simulation_output = context.simulate_scenario(
                scenario_name=name,
                model_kind=spec["model_kind"],
                alpha=spec["alpha"],
                steps=runtime["time_steps"],
            )

        progress.advance("Running NSGA-II optimization", f"(Scenario: {name})")
        with timer.time_block(f"optimize_{name}"):
            optimization_output = context.optimize_scenario(
                simulation_output,
                w1=spec["w1"],
                w2=spec["w2"],
                traffic_delta_bounds=spec["traffic_delta_bounds"],
                energy_delta_bounds=spec["energy_delta_bounds"],
            )
        scenarios[name] = summarize_result(simulation_output, optimization_output)

    sensitivity = {}
    for alpha in ALPHA_VALUES:
        progress.advance("Sensitivity analysis", f"(alpha={alpha})")
        with timer.time_block(f"sensitivity_sim_{alpha}"):
            simulation_output = context.simulate_scenario(
                scenario_name="Cross-domain",
                model_kind="cross_domain",
                alpha=alpha,
                steps=runtime["time_steps"],
            )

        progress.advance("Running NSGA-II optimization", f"(Sensitivity alpha={alpha})")
        with timer.time_block(f"sensitivity_opt_{alpha}"):
            optimization_output = context.optimize_scenario(
                simulation_output,
                w1=W1,
                w2=W2,
                traffic_delta_bounds=(-0.2, 0.2),
                energy_delta_bounds=(-0.5, 0.5),
            )

        sensitivity[str(alpha)] = {
            "travel_time": optimization_output["travel_time"],
            "energy_variance": optimization_output["energy_variance"],
        }

    benchmark = {}
    for label, spec in benchmark_specs.items():
        progress.advance("Benchmarking", f"(Model: {label})")
        with timer.time_block(f"benchmark_sim_{label}"):
            simulation_output = context.simulate_scenario(
                scenario_name=spec["scenario_name"],
                model_kind=spec["model_kind"],
                alpha=spec["alpha"],
                steps=runtime["time_steps"],
            )

        progress.advance("Running NSGA-II optimization", f"(Benchmark: {label})")
        with timer.time_block(f"benchmark_opt_{label}"):
            optimization_output = context.optimize_scenario(
                simulation_output,
                w1=spec["w1"],
                w2=spec["w2"],
                traffic_delta_bounds=spec["traffic_delta_bounds"],
                energy_delta_bounds=spec["energy_delta_bounds"],
            )

        benchmark[label] = {
            "mae": context.evaluate_prediction_mae(spec["model_kind"]),
            "travel_time": optimization_output["travel_time"],
            "energy_variance": optimization_output["energy_variance"],
        }

    results = {
        "mode": "dev" if is_dev_mode() else "full",
        "scenarios": scenarios,
        "sensitivity_analysis": sensitivity,
        "model_comparison": benchmark,
        "timings": {name: round(value, 4) for name, value in sorted(timer.timings.items(), key=lambda item: item[1], reverse=True)},
        "training_loss_logs": {name: [float(item) for item in losses] for name, losses in trained_models.items()},
    }
    results["validation"] = validate_results(results)

    progress.advance("Generating results")
    Path("results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n===== MODE =====")
    print("DEV_MODE" if is_dev_mode() else "FULL_MODE")

    print("\n===== SCENARIO RESULTS =====")
    for name, values in scenarios.items():
        metrics = values["post_optimization"]
        print(f"{name}: Travel Time={metrics['travel_time']:.4f}, Energy Variance={metrics['energy_variance']:.4f}")

    print("\n===== VALIDATION =====")
    for key, value in results["validation"].items():
        print(f"{key}: {value}")

    print("\n===== BOTTLENECKS =====")
    for name, value in list(results["timings"].items())[:5]:
        print(f"{name}: {value:.2f}s")

    print("[100%] Completed")


if __name__ == "__main__":
    main()
