import argparse
import os
import random
import sys

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from flwr.common import Context
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CVB_FL.config import CVB_KERNEL_SIZE, CVB_SCALE, get_experiment_config
from CVB_FL.fl.client import CVBPrivacyClient
from CVB_FL.fl.strategy import FedAvgCVB
from clientmanager.manager import SimpleClientManager
from function_strategy.function_stategy import (
    aggregate_evaluate_metrics,
    aggregate_fit_metrics,
)
from models.cvb import CVBNet, CVBResNet50
from preprocessing.data_handling import (
    get_dataloader,
    get_dataloader_cifar10,
    split_cifar10_dirichlet_flwr,
    split_mnist_dirichlet_flwr,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_protocol(cfg: dict, check_rounds: bool = True) -> None:
    expected = {
        "fashion": {"NUM_CLIENTS": 100, "CLIENTS_PER_ROUND": 10, "NUM_ROUNDS": 200, "LEARNING_RATE": 0.01},
        "cifar": {"NUM_CLIENTS": 50, "CLIENTS_PER_ROUND": 10, "NUM_ROUNDS": 500, "LEARNING_RATE": 0.03},
    }[cfg["DATASET"]]
    for key, value in expected.items():
        if key == "NUM_ROUNDS" and not check_rounds:
            continue
        if cfg[key] != value:
            raise ValueError(f"Protocol mismatch for {cfg['DATASET']}: {key}={cfg[key]} != {value}")
    expected_fraction = cfg["CLIENTS_PER_ROUND"] / cfg["NUM_CLIENTS"]
    if abs(cfg["FRACTION_FIT"] - expected_fraction) > 1e-9:
        raise ValueError("FRACTION_FIT is not consistent with CLIENTS_PER_ROUND / NUM_CLIENTS")


def build_dataset_partition(cfg: dict):
    if cfg["DATASET"] == "fashion":
        return split_mnist_dirichlet_flwr(num_clients=cfg["NUM_CLIENTS"], alpha=0.5, seed=cfg["SEED"])
    return split_cifar10_dirichlet_flwr(num_clients=cfg["NUM_CLIENTS"], alpha=0.5, seed=cfg["SEED"])


def get_client_fn(cfg: dict, federated_data: dict):
    def client_fn(context: Context) -> fl.client.Client:
        partition_id = context.node_config["partition-id"]
        key = f"client_{partition_id}"
        if key not in federated_data:
            raise ValueError(f"Client ID {partition_id} does not exist in federated_data")
        if cfg["DATASET"] == "fashion":
            train_loader = get_dataloader(federated_data[key], batch_size=cfg["BATCH_SIZE"])
            model = CVBNet(cvb_scale=CVB_SCALE, cvb_kernel_size=CVB_KERNEL_SIZE)
        else:
            train_loader = get_dataloader_cifar10(federated_data[key], batch_size=cfg["BATCH_SIZE"])
            model = CVBResNet50(num_classes=cfg["NUM_CLASSES"], cvb_scale=CVB_SCALE, cvb_kernel_size=CVB_KERNEL_SIZE)
        return CVBPrivacyClient(model, train_loader, learning_rate=cfg["LEARNING_RATE"]).to_client()

    return client_fn


def get_evaluate_fn(cfg: dict, testset):
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        if cfg["DATASET"] == "fashion":
            model = CVBNet(cvb_scale=CVB_SCALE, cvb_kernel_size=CVB_KERNEL_SIZE).to(device)
        else:
            model = CVBResNet50(num_classes=cfg["NUM_CLASSES"], cvb_scale=CVB_SCALE, cvb_kernel_size=CVB_KERNEL_SIZE).to(device)
        model.eval()

        model_keys = list(model.state_dict().keys())
        state_dict = {}
        for key, weight in zip(model_keys, parameters):
            model_weight_shape = model.state_dict()[key].shape
            weight_tensor = torch.tensor(weight, device=device)
            if weight_tensor.shape == model_weight_shape:
                state_dict[key] = weight_tensor
        model.load_state_dict(state_dict, strict=False)

        criterion = nn.CrossEntropyLoss()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total else 0.0
        avg_loss = total_loss / total if total else 0.0
        with open(os.path.join(cfg["RESULTS_DIR"], "centralized_accuracy.txt"), "a", encoding="utf-8") as f:
            f.write(f"round={server_round},value={accuracy:.8f}\n")
        with open(os.path.join(cfg["RESULTS_DIR"], "centralized_loss.txt"), "a", encoding="utf-8") as f:
            f.write(f"round={server_round},value={avg_loss:.8f}\n")
        return avg_loss, {"accuracy": accuracy}

    return evaluate


def build_centralized_testset(dataset: str):
    if dataset == "fashion":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    return datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CVB_FL with IWQoS baseline protocol.")
    parser.add_argument("--dataset", choices=["fashion", "cifar"], default="fashion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rounds", type=int, default=None)
    return parser.parse_args()


def make_fit_config(cfg: dict):
    def _fit_config(server_round: int) -> dict:
        return {
            "learning_rate": cfg["LEARNING_RATE"],
            "batch_size": cfg["BATCH_SIZE"],
            "round": server_round,
        }

    return _fit_config


def make_evaluate_config(cfg: dict):
    def _evaluate_config(server_round: int) -> dict:
        return {"batch_size": cfg["BATCH_SIZE"], "round": server_round}

    return _evaluate_config


def main():
    args = parse_args()
    cfg = get_experiment_config(args.dataset, args.seed)
    assert_protocol(cfg, check_rounds=True)
    if args.num_rounds is not None:
        cfg["NUM_ROUNDS"] = args.num_rounds
        assert_protocol(cfg, check_rounds=False)
    set_global_seed(cfg["SEED"])

    _, federated_data = build_dataset_partition(cfg)
    centralized_testset = build_centralized_testset(cfg["DATASET"])

    client_manager = SimpleClientManager()
    strategy = FedAvgCVB(
        fraction_fit=cfg["FRACTION_FIT"],
        fraction_evaluate=cfg["FRACTION_EVALUATE"],
        min_fit_clients=cfg["CLIENTS_PER_ROUND"],
        min_evaluate_clients=cfg["CLIENTS_PER_ROUND"],
        min_available_clients=cfg["NUM_CLIENTS"],
        on_fit_config_fn=make_fit_config(cfg),
        on_evaluate_config_fn=make_evaluate_config(cfg),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        evaluate_fn=get_evaluate_fn(cfg, centralized_testset),
        results_dir=cfg["RESULTS_DIR"],
    )

    fl.simulation.start_simulation(
        client_fn=get_client_fn(cfg, federated_data),
        num_clients=cfg["NUM_CLIENTS"],
        config=fl.server.ServerConfig(num_rounds=cfg["NUM_ROUNDS"]),
        strategy=strategy,
        client_manager=client_manager,
        client_resources={"num_cpus": 1, "num_gpus": 0.1},
    )


if __name__ == "__main__":
    main()
