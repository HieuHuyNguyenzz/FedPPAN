import os

import flwr as fl
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from flwr.common import Context
from torch.utils.data import DataLoader

from CVB_FL.config import (
    BATCH_SIZE,
    CVB_KERNEL_SIZE,
    CVB_SCALE,
    NUM_CLIENTS,
    NUM_ROUNDS,
    RESULTS_DIR,
)
from CVB_FL.fl.client import CVBPrivacyClient
from CVB_FL.fl.strategy import FedAvgCVB
from clientmanager.manager import SimpleClientManager
from function_strategy.function_stategy import (
    aggregate_evaluate_metrics,
    aggregate_fit_metrics,
    evaluate_config,
    fit_config,
)
from models.cvb import CVBNet
from preprocessing.data_handling import get_dataloader, split_mnist_dirichlet_flwr

fds, federated_data = split_mnist_dirichlet_flwr(num_clients=NUM_CLIENTS)


def get_client_fn():
    def client_fn(context: Context) -> fl.client.Client:
        partition_id = context.node_config["partition-id"]
        key = f"client_{partition_id}"
        if key not in federated_data:
            raise ValueError(f"Client ID {partition_id} does not exist in federated_data")
        train_loader = get_dataloader(federated_data[key], batch_size=BATCH_SIZE)
        model = CVBNet(cvb_scale=CVB_SCALE, cvb_kernel_size=CVB_KERNEL_SIZE)
        return CVBPrivacyClient(model, train_loader).to_client()

    return client_fn


def get_evaluate_fn(testset):
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        model = CVBNet(cvb_scale=CVB_SCALE, cvb_kernel_size=CVB_KERNEL_SIZE).to(device)
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

        with open(os.path.join(RESULTS_DIR, "centralized_accuracy.txt"), "a", encoding="utf-8") as f:
            f.write(f"round={server_round},value={accuracy:.8f}\n")
        with open(os.path.join(RESULTS_DIR, "centralized_loss.txt"), "a", encoding="utf-8") as f:
            f.write(f"round={server_round},value={avg_loss:.8f}\n")

        return avg_loss, {"accuracy": accuracy}

    return evaluate


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    centralized_testset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    client_manager = SimpleClientManager()
    strategy = FedAvgCVB(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=max(1, int(0.1 * NUM_CLIENTS)),
        min_evaluate_clients=max(1, int(0.1 * NUM_CLIENTS)),
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        evaluate_fn=get_evaluate_fn(centralized_testset),
    )

    fl.simulation.start_simulation(
        client_fn=get_client_fn(),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_manager=client_manager,
        client_resources={"num_cpus": 1, "num_gpus": 0.1},
    )


if __name__ == "__main__":
    main()
