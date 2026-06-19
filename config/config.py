import os

# Global configuration for the smart-city digital twin experiments.

NUM_TRAFFIC_NODES = 10
NUM_ENERGY_NODES = 4

CACHE_VERSION = 2  # Bump when model architecture or training data format changes
ADJ_CORRELATION_THRESHOLD = 0.5  # Pearson correlation threshold for data-driven traffic graph

RANDOM_SEED = 7
DATA_PATH = "data/metr_la.npz"
MODEL_CACHE_DIR = "artifacts/models"

DEV_MODE = True

SEQUENCE_LENGTH = 6
TRAIN_SPLIT = 0.8
LEARNING_RATE = 0.003
GRAD_CLIP = 1.0

TRAINING_ALPHA = 0.3
ALPHA = 0.3
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7]

# Scenario selection weights for the Pareto set.
W1 = 0.85  # travel-time importance
W2 = 0.15  # energy-variance importance

DEV_SETTINGS = {
    "train_epochs": 4,
    "time_steps": 12,
    "sample_limit": 320,
    "nsga2_population": 10,
    "nsga2_generations": 5,
}

FULL_SETTINGS = {
    "train_epochs": 8,
    "time_steps": 40,
    "sample_limit": None,
    "nsga2_population": 40,
    "nsga2_generations": 30,
}


def get_runtime_settings():
    return DEV_SETTINGS if is_dev_mode() else FULL_SETTINGS


def is_dev_mode():
    override = os.getenv("DIGITAL_TWIN_DEV_MODE")
    if override is None:
        return DEV_MODE
    return override.lower() not in {"0", "false", "full"}
