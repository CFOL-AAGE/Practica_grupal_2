"""Modelos y utilidades sobre modelos.

En PyTorch un modelo es un ``nn.Module`` (el equivalente a ``keras.Model``); para redes que son
una simple pila de capas basta con ``nn.Sequential`` (como ``keras.Sequential``).
Los pesos se leen con ``model.state_dict()`` y se escriben con ``model.load_state_dict(sd)``:
es como ``get_weights`` / ``set_weights`` de Keras, pero un diccionario ``{nombre: tensor}``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fl.utils import Config

StateDict = dict[str, torch.Tensor]

CNN_MAX_PARAMS = 100_000


class MLP(nn.Sequential):
    """Perceptrón multicapa de las diapositivas: dos capas ocultas de ``hidden`` neuronas.

    La última capa devuelve *logits* (sin softmax): la entropía cruzada de PyTorch ya la aplica.
    """

    def __init__(self, hidden: int = 200, n_classes: int = 10):
        super().__init__(
            nn.Flatten(),                      # (B, 1, 28, 28) -> (B, 784)
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),      # logits
        )


class CNN(nn.Sequential):
    """Red convolucional para imágenes ``(1, 28, 28)`` con como mucho ``CNN_MAX_PARAMS`` parámetros.

    CNN de FedAvg (McMahan et al., 2017) sin su capa densa de 512 unidades: dos bloques
    conv 5x5-ReLU-maxpool (28 -> 14 -> 7) y una capa lineal. 83.466 parámetros.
    Sin BatchNorm, dropout ni global average pooling: con datos no-IID empeoran FedAvg.
    Justificación completa en decisiones/01-arquitectura-cnn.md.
    """

    def __init__(self, n_classes: int = 10):
        super().__init__(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),   # 832 params,    (B, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              #                (B, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # 51.264 params, (B, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              #                (B, 64, 7, 7)
            nn.Flatten(),                                 #                (B, 3136)
            nn.Linear(64 * 7 * 7, n_classes),             # 31.370 params -> logits
        )


def build_model(cfg: Config) -> nn.Module:
    """Crea el modelo indicado en la sección ``model`` de la configuración."""
    if cfg.name == "mlp":
        return MLP(cfg.hidden)
    if cfg.name == "cnn":
        model = CNN()
        if (n := count_params(model)) > CNN_MAX_PARAMS:
            raise ValueError(f"La CNN tiene {n:,} parámetros; el límite es {CNN_MAX_PARAMS:,}")
        return model
    raise ValueError(f"Modelo desconocido: {cfg.name!r}")


def count_params(model: nn.Module) -> int:
    """Número de parámetros entrenables (``model.count_params()`` en Keras)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: Dataset, device: str = "cpu", batch_size: int = 1000) -> dict[str, float]:
    """Pérdida (entropía cruzada) y accuracy de ``model`` sobre ``dataset`` (``model.evaluate`` en Keras)."""
    model.eval()  # desactiva dropout, BatchNorm en modo inferencia...
    loss, correct = 0.0, 0
    for x, y in DataLoader(dataset, batch_size=batch_size):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
    n = len(dataset)
    return {"loss": loss / n, "acc": correct / n}
