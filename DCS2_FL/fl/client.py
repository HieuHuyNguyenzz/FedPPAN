from typing import Dict, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import NDArrays, Scalar

from DCS2_FL.config import (
    DCS2_EPSILON,
    DCS2_INIT_MODE,
    DCS2_LAMBDA_G,
    DCS2_LAMBDA_X,
    DCS2_LAMBDA_Z,
    DCS2_SYNTH_LR,
    DCS2_SYNTH_STEPS,
    DEVICE,
)
from metric.metrics import compute_distortion, compute_privacy_leakage
from models.dcs2 import DCS2Defender


class DCS2PrivacyClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, learning_rate: float, num_classes: int):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.defender = DCS2Defender(
            model=self.model,
            criterion=self.criterion,
            lambda_g=DCS2_LAMBDA_G,
            lambda_x=DCS2_LAMBDA_X,
            lambda_z=DCS2_LAMBDA_Z,
            epsilon=DCS2_EPSILON,
            synth_steps=DCS2_SYNTH_STEPS,
            synth_lr=DCS2_SYNTH_LR,
            init_mode=DCS2_INIT_MODE,
            num_classes=num_classes,
        )

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        ndarray_params = [torch.tensor(p, dtype=torch.float32, device=DEVICE) for p in parameters]
        params_dict = zip(self.model.state_dict().keys(), ndarray_params)
        state_dict = {k: v for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        conceal_obj_total = 0.0
        grad_cos_total = 0.0
        proj_ratio_total = 0.0
        num_batches = 0

        for images, labels in self.train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()

            grads, dcs_stats = self.defender.obfuscated_batch_gradient(images, labels)
            for p, g in zip(self.model.parameters(), grads):
                p.grad = g
            self.optimizer.step()

            with torch.no_grad():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (predicted == labels).sum().item()
            conceal_obj_total += dcs_stats["conceal_obj"]
            grad_cos_total += dcs_stats["grad_cosine"]
            proj_ratio_total += dcs_stats["proj_applied_ratio"]
            num_batches += 1

        updated_parameters = self.get_parameters()
        flat_params = np.concatenate([p.flatten() for p in updated_parameters])
        jitter = np.random.normal(0.0, 1e-6, size=flat_params.shape)
        privacy_leakage = float(compute_privacy_leakage(flat_params + jitter, flat_params))
        distortion = float(compute_distortion(flat_params, flat_params + jitter))

        return updated_parameters, total_samples, {
            "loss": float(total_loss / max(1, total_samples)),
            "accuracy": float(total_correct / max(1, total_samples)),
            "privacy_leakage": privacy_leakage,
            "distortion": distortion,
            "conceal_obj": float(conceal_obj_total / max(1, num_batches)),
            "grad_cosine": float(grad_cos_total / max(1, num_batches)),
            "proj_applied_ratio": float(proj_ratio_total / max(1, num_batches)),
        }
