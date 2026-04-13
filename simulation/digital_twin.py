import random
from pathlib import Path

import numpy as np
import torch

from config.config import (
    DATA_PATH,
    GRAD_CLIP,
    LEARNING_RATE,
    MODEL_CACHE_DIR,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TRAINING_ALPHA,
    TRAIN_SPLIT,
    get_runtime_settings,
)
from coupling.cross_domain import CrossDomainCoupling
from data.real_loader import RealTrafficLoader
from evaluation.metrics import compute_energy_variance, compute_travel_time
from models.lstm_model import SimpleLSTM
from models.stgcn_model import STGCN
from models.train_stgcn import evaluate_predictions, train_gnn_model, train_sequence_model
from optimization.optimizer import Optimizer


class ExperimentContext:
    def __init__(self, graph):
        self.graph = graph
        self.device = torch.device("cpu")
        self.runtime = get_runtime_settings()
        self._set_seed(RANDOM_SEED)

        self.loader = RealTrafficLoader(DATA_PATH, len(graph.traffic_nodes))
        self.optimizer = Optimizer(graph)
        self._bundle_cache = {}
        self._model_cache = {}
        self._loss_cache = {}
        self.cache_dir = Path(MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _limit_series(self, series):
        limit = self.runtime["sample_limit"]
        if limit is None:
            return series
        return series[:limit]

    def _build_training_bundle(self, model_kind):
        if model_kind in self._bundle_cache:
            return self._bundle_cache[model_kind]

        raw_traffic = self._limit_series(self.loader.get_raw_series())
        norm_traffic = self._limit_series(self.loader.get_normalized_series())
        coupling = CrossDomainCoupling(self.graph, TRAINING_ALPHA)

        energy_series = []
        for traffic_step in raw_traffic:
            traffic_dict = {
                node: float(value)
                for node, value in zip(self.graph.traffic_nodes, traffic_step.tolist())
            }
            energy_dict = coupling.compute_energy_demand(traffic_dict)
            energy_series.append([energy_dict[node] for node in self.graph.energy_nodes])

        raw_energy = np.asarray(energy_series, dtype=np.float32)
        energy_mean = raw_energy.mean(axis=0, keepdims=True)
        energy_std = raw_energy.std(axis=0, keepdims=True) + 1e-6
        norm_energy = (raw_energy - energy_mean) / energy_std
        include_energy = model_kind == "cross_domain"
        edge_index, _ = self.graph.to_edge_index(include_energy=include_energy)

        samples = []
        targets = []
        target_traffic_raw = []
        target_energy_raw = []

        for index in range(SEQUENCE_LENGTH, len(norm_traffic)):
            traffic_window = norm_traffic[index - SEQUENCE_LENGTH : index].T
            traffic_target = norm_traffic[index]
            energy_target = norm_energy[index]

            if model_kind == "cross_domain":
                energy_window = np.zeros((len(self.graph.energy_nodes), SEQUENCE_LENGTH), dtype=np.float32)
                sample = np.concatenate([traffic_window, energy_window], axis=0)
                target = np.concatenate([traffic_target, energy_target], axis=0)
            else:
                sample = traffic_window
                target = traffic_target

            samples.append(sample.astype(np.float32))
            targets.append(target.astype(np.float32))
            target_traffic_raw.append(raw_traffic[index].astype(np.float32))
            target_energy_raw.append(raw_energy[index].astype(np.float32))

        split_index = max(1, int(len(samples) * TRAIN_SPLIT))
        bundle = {
            "train_inputs": [torch.tensor(item, dtype=torch.float32, device=self.device) for item in samples[:split_index]],
            "train_targets": [
                torch.tensor(item.reshape(-1, 1), dtype=torch.float32, device=self.device)
                for item in targets[:split_index]
            ],
            "test_inputs": [torch.tensor(item, dtype=torch.float32, device=self.device) for item in samples[split_index:]],
            "test_targets": [
                torch.tensor(item.reshape(-1, 1), dtype=torch.float32, device=self.device)
                for item in targets[split_index:]
            ],
            "test_traffic_raw": target_traffic_raw[split_index:],
            "test_energy_raw": target_energy_raw[split_index:],
            "energy_mean": energy_mean.squeeze(0),
            "energy_std": energy_std.squeeze(0),
            "edge_index": edge_index.to(self.device),
        }

        self._bundle_cache[model_kind] = bundle
        return bundle

    def train_model(self, model_kind):
        if model_kind in self._model_cache:
            return self._model_cache[model_kind], self._loss_cache[model_kind]

        bundle = self._build_training_bundle(model_kind)
        cache_path = self.cache_dir / f"{model_kind}.pth"
        cache_signature = {
            "model_kind": model_kind,
            "mode": "dev" if self.runtime["sample_limit"] is not None else "full",
            "train_epochs": self.runtime["train_epochs"],
            "sample_limit": self.runtime["sample_limit"],
            "sequence_length": SEQUENCE_LENGTH,
        }

        if model_kind == "lstm":
            model = SimpleLSTM(hidden_size=32).to(self.device)
        else:
            model = STGCN(sequence_length=SEQUENCE_LENGTH, hidden_dim=32).to(self.device)

        if cache_path.exists():
            checkpoint = torch.load(cache_path, map_location=self.device)
            if checkpoint.get("signature") == cache_signature:
                model.load_state_dict(checkpoint["state_dict"])
                losses = checkpoint.get("losses", [])
                self._model_cache[model_kind] = model
                self._loss_cache[model_kind] = losses
                return model, losses

        if model_kind == "lstm":
            losses = train_sequence_model(
                model,
                bundle["train_inputs"],
                bundle["train_targets"],
                epochs=self.runtime["train_epochs"],
                learning_rate=LEARNING_RATE,
                grad_clip=GRAD_CLIP,
            )
        else:
            losses = train_gnn_model(
                model,
                bundle["train_inputs"],
                bundle["train_targets"],
                bundle["edge_index"],
                epochs=self.runtime["train_epochs"],
                learning_rate=LEARNING_RATE,
                grad_clip=GRAD_CLIP,
            )

        torch.save(
            {
                "signature": cache_signature,
                "state_dict": model.state_dict(),
                "losses": losses,
            },
            cache_path,
        )

        self._model_cache[model_kind] = model
        self._loss_cache[model_kind] = losses
        return model, losses

    def predict(self, model_kind, sample):
        model, _ = self.train_model(model_kind)
        bundle = self._build_training_bundle(model_kind)

        model.eval()
        with torch.no_grad():
            if model_kind == "lstm":
                prediction = model(sample)
            else:
                prediction = model(sample, bundle["edge_index"])
        return prediction.squeeze(-1).cpu().numpy()

    def evaluate_prediction_mae(self, model_kind):
        bundle = self._build_training_bundle(model_kind)
        predictions = []
        targets = []

        for sample, truth in zip(bundle["test_inputs"], bundle["test_traffic_raw"]):
            pred = self.predict(model_kind, sample)
            traffic_pred = pred[: len(self.graph.traffic_nodes)]
            traffic_raw = self.loader.denormalize_traffic(traffic_pred)
            predictions.append(traffic_raw)
            targets.append(truth)

        return evaluate_predictions(predictions, targets)

    def simulate_scenario(self, scenario_name, model_kind, alpha, steps):
        bundle = self._build_training_bundle(model_kind)
        total_steps = min(steps, len(bundle["test_inputs"]))
        coupling = CrossDomainCoupling(self.graph, alpha)
        losses = self._loss_cache[model_kind]

        raw_history = []
        traffic_accumulator = []
        energy_accumulator = []
        diagnostics = []

        for sample, energy_truth in zip(
            bundle["test_inputs"][:total_steps],
            bundle["test_energy_raw"][:total_steps],
        ):
            pred = self.predict(model_kind, sample)
            traffic_pred = self.loader.denormalize_traffic(pred[: len(self.graph.traffic_nodes)])
            traffic_pred = np.clip(traffic_pred, 0.0, None)

            traffic_dict = {
                node: float(value)
                for node, value in zip(self.graph.traffic_nodes, traffic_pred.tolist())
            }
            energy_formula = coupling.compute_energy_demand(traffic_dict)

            if model_kind == "cross_domain":
                energy_pred = pred[len(self.graph.traffic_nodes) :]
                energy_pred = np.clip(
                    energy_pred * bundle["energy_std"] + bundle["energy_mean"],
                    0.0,
                    None,
                )
            else:
                energy_pred = np.array([energy_formula[node] for node in self.graph.energy_nodes], dtype=np.float32)

            if scenario_name == "Energy-only":
                energy_values = {
                    node: float(max(0.0, truth))
                    for node, truth in zip(self.graph.energy_nodes, energy_truth.tolist())
                }
            elif scenario_name == "Ablation":
                energy_values = {node: float(energy_formula[node]) for node in self.graph.energy_nodes}
            elif model_kind == "cross_domain":
                energy_values = {
                    node: float(max(0.0, 0.5 * model_value + 0.5 * energy_formula[node]))
                    for node, model_value in zip(self.graph.energy_nodes, energy_pred.tolist())
                }
            else:
                energy_values = {node: float(max(0.0, energy_formula[node])) for node in self.graph.energy_nodes}

            raw_history.append(
                {
                    "travel_time": float(compute_travel_time(traffic_dict)),
                    "energy_variance": float(compute_energy_variance(energy_values)),
                }
            )
            traffic_accumulator.append([traffic_dict[node] for node in self.graph.traffic_nodes])
            energy_accumulator.append([energy_values[node] for node in self.graph.energy_nodes])
            diagnostics.append(
                {
                    "traffic_min": float(min(traffic_dict.values())),
                    "traffic_max": float(max(traffic_dict.values())),
                    "energy_min": float(min(energy_values.values())),
                    "energy_max": float(max(energy_values.values())),
                }
            )

        aggregated_traffic = np.mean(np.asarray(traffic_accumulator, dtype=np.float32), axis=0)
        aggregated_energy = np.mean(np.asarray(energy_accumulator, dtype=np.float32), axis=0)

        return {
            "raw_history": raw_history,
            "aggregated_traffic": {
                node: float(value) for node, value in zip(self.graph.traffic_nodes, aggregated_traffic.tolist())
            },
            "aggregated_energy": {
                node: float(value) for node, value in zip(self.graph.energy_nodes, aggregated_energy.tolist())
            },
            "diagnostics": diagnostics,
            "losses": [float(loss) for loss in losses],
        }

    def optimize_scenario(self, simulation_output, w1, w2, traffic_delta_bounds, energy_delta_bounds):
        adjusted_traffic, adjusted_energy, pareto_front = self.optimizer.optimize(
            simulation_output["aggregated_traffic"],
            simulation_output["aggregated_energy"],
            w1=w1,
            w2=w2,
            population_size=self.runtime["nsga2_population"],
            generations=self.runtime["nsga2_generations"],
            traffic_delta_bounds=traffic_delta_bounds,
            energy_delta_bounds=energy_delta_bounds,
        )

        return {
            "travel_time": float(compute_travel_time(adjusted_traffic)),
            "energy_variance": float(compute_energy_variance(adjusted_energy)),
            "optimized_traffic": adjusted_traffic,
            "optimized_energy": adjusted_energy,
            "pareto_front": pareto_front,
        }
