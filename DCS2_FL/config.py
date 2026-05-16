import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEAKAGE_METRIC_MODE = "legacy"
NUM_REPEATS = 3

# DCS2 parameters
DCS2_LAMBDA_G = 0.7
DCS2_LAMBDA_X = 1.0
DCS2_LAMBDA_Z = 1.0
DCS2_EPSILON = 0.1
DCS2_SYNTH_STEPS = 15
DCS2_SYNTH_LR = 0.1
DCS2_INIT_MODE = "random"


def get_experiment_config(dataset: str, seed: int) -> dict:
    dataset = dataset.lower()
    if dataset == "fashion":
        cfg = {
            "DATASET": "fashion",
            "NUM_CLIENTS": 100,
            "CLIENTS_PER_ROUND": 10,
            "NUM_ROUNDS": 200,
            "BATCH_SIZE": 16,
            "LEARNING_RATE": 0.01,
            "NUM_CHANNEL": 1,
            "NUM_CLASSES": 10,
        }
    elif dataset == "cifar":
        cfg = {
            "DATASET": "cifar",
            "NUM_CLIENTS": 50,
            "CLIENTS_PER_ROUND": 10,
            "NUM_ROUNDS": 500,
            "BATCH_SIZE": 16,
            "LEARNING_RATE": 0.03,
            "NUM_CHANNEL": 3,
            "NUM_CLASSES": 10,
        }
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'. Use 'fashion' or 'cifar'.")

    cfg["SEED"] = seed
    cfg["FRACTION_FIT"] = cfg["CLIENTS_PER_ROUND"] / cfg["NUM_CLIENTS"]
    cfg["FRACTION_EVALUATE"] = cfg["CLIENTS_PER_ROUND"] / cfg["NUM_CLIENTS"]
    cfg["RESULTS_DIR"] = os.path.join("results", "dcs2_fl", cfg["DATASET"], f"seed_{seed}")
    os.makedirs(cfg["RESULTS_DIR"], exist_ok=True)
    return cfg
