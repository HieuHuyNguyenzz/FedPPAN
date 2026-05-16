from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors], dim=0)


def _sample_loss(model: nn.Module, criterion: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return criterion(logits, y)


def _sample_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # f_theta in the paper: use current model output representation.
    return model(x)


def _sample_grads(
    model: nn.Module,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    create_graph: bool,
) -> List[torch.Tensor]:
    loss = _sample_loss(model, criterion, x, y)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, retain_graph=create_graph)
    return [g for g in grads]


def gradient_projection(g_ref: List[torch.Tensor], g_new: List[torch.Tensor]) -> Tuple[List[torch.Tensor], bool]:
    """Project g_new onto half-space with non-negative dot(g_ref, g_new)."""
    ref_vec = _flatten_tensors(g_ref)
    new_vec = _flatten_tensors(g_new)
    dot = torch.dot(ref_vec, new_vec)
    if dot >= 0:
        return g_new, False
    denom = torch.dot(ref_vec, ref_vec) + 1e-12
    v_star = torch.clamp(-dot / denom, min=0.0)
    projected = [gn + v_star * gr for gn, gr in zip(g_new, g_ref)]
    return projected, True


class DCS2Defender:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        lambda_g: float = 0.7,
        lambda_x: float = 1.0,
        lambda_z: float = 1.0,
        epsilon: float = 0.1,
        synth_steps: int = 15,
        synth_lr: float = 0.1,
        init_mode: str = "random",
        num_classes: int = 10,
    ):
        self.model = model
        self.criterion = criterion
        self.lambda_g = lambda_g
        self.lambda_x = lambda_x
        self.lambda_z = lambda_z
        self.epsilon = epsilon
        self.synth_steps = synth_steps
        self.synth_lr = synth_lr
        self.init_mode = init_mode
        self.num_classes = num_classes

    def _init_concealed(self, xs: torch.Tensor) -> torch.Tensor:
        if self.init_mode == "noise" or self.init_mode == "random":
            x0 = torch.rand_like(xs) * 2.0 - 1.0
        else:
            x0 = torch.rand_like(xs) * 2.0 - 1.0
        return x0.detach()

    def synthesize_concealed_sample(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        y_concealed: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        xc = self._init_concealed(xs).requires_grad_(True)
        optimizer = torch.optim.SGD([xc], lr=self.synth_lr)
        last_obj = torch.tensor(0.0, device=xs.device)
        last_cos = torch.tensor(0.0, device=xs.device)

        for _ in range(self.synth_steps):
            optimizer.zero_grad()
            g_s = _sample_grads(self.model, self.criterion, xs, ys, create_graph=True)
            g_c = _sample_grads(self.model, self.criterion, xc, y_concealed, create_graph=True)
            g_s_vec = _flatten_tensors(g_s)
            g_c_vec = _flatten_tensors(g_c)

            cos = F.cosine_similarity(g_c_vec.unsqueeze(0), g_s_vec.unsqueeze(0), dim=1).mean()
            feat_s = _sample_features(self.model, xs)
            feat_c = _sample_features(self.model, xc)
            latent_term = torch.norm(feat_c - feat_s, p=2) / (torch.norm(feat_s, p=2) + 1e-12)
            visual_term = torch.exp(-self.lambda_x * torch.norm(xc - xs, p=2))
            obj = (1.0 - cos) + visual_term + self.lambda_z * (latent_term - self.epsilon)

            obj.backward()
            optimizer.step()
            xc.data.clamp_(-1.0, 1.0)
            last_obj = obj.detach()
            last_cos = cos.detach()

        return xc.detach(), float(last_obj.item()), float(last_cos.item())

    def obfuscated_batch_gradient(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad_sums = [torch.zeros_like(p) for p in params]
        obj_values: List[float] = []
        cos_values: List[float] = []
        projected_count = 0
        batch_size = images.size(0)

        # All-sensitive policy: each sample gets a concealed counterpart.
        for i in range(batch_size):
            xs = images[i : i + 1]
            ys = labels[i : i + 1]
            y0 = torch.randint(low=0, high=self.num_classes, size=(1,), device=labels.device)

            xc, obj_val, cos_val = self.synthesize_concealed_sample(xs, ys, y0)
            obj_values.append(obj_val)
            cos_values.append(cos_val)

            g = _sample_grads(self.model, self.criterion, xs, ys, create_graph=False)
            g_c_label = _sample_grads(self.model, self.criterion, xc, y0, create_graph=False)
            g_c_sensitive = _sample_grads(self.model, self.criterion, xc, ys, create_graph=False)

            gc = [
                gs + self.lambda_g * gcl + (1.0 - self.lambda_g) * gcs
                for gs, gcl, gcs in zip(g, g_c_label, g_c_sensitive)
            ]
            g_hat, projected = gradient_projection(g, gc)
            if projected:
                projected_count += 1
            for j, grad in enumerate(g_hat):
                grad_sums[j] += grad.detach()

        avg_grads = [g / float(batch_size) for g in grad_sums]
        stats = {
            "conceal_obj": float(sum(obj_values) / max(1, len(obj_values))),
            "grad_cosine": float(sum(cos_values) / max(1, len(cos_values))),
            "proj_applied_ratio": float(projected_count / max(1, batch_size)),
        }
        return avg_grads, stats
