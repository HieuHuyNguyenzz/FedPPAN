import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEAKAGE_METRIC_MODE = "legacy"
NUM_REPEATS = 3

# CVB parameters (paper defaults for CNN)
CVB_POSITION = 1
CVB_KERNEL_SIZE = 5
CVB_SCALE = 0.5
CVB_BETA = 0.1


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
    cfg["RESULTS_DIR"] = os.path.join("results", "cvb_fl", cfg["DATASET"], f"seed_{seed}")
    os.makedirs(cfg["RESULTS_DIR"], exist_ok=True)
    return cfg
