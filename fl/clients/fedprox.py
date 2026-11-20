"""FedProx (Li et al., 2020): cliente de FedAvg con un término proximal en la pérdida."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

from fl.clients.base import Client


class FedProxClient(Client):
    """Penaliza alejarse del modelo global: ``loss + (mu / 2) * ||w - w_global||²``.

    ``w`` son los parámetros locales y ``w_global`` los del modelo recibido. El servidor no
    cambia: FedProx se agrega con FedAvg.
    """

    def __init__(self, cid: int, dataset: Dataset, *, mu: float, **kwargs):
        super().__init__(cid, dataset, **kwargs)
        self.mu = mu

    def loss(self, model: nn.Module, global_model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Simplemente heredamos la clase cliente y le modificamos la funcion de perdida
        # Es la que se ejecuta del lado del cliente.
        # Lo de arriba: a la perdida sumarle el bloque (mu / 2) * ||w - w_global||²

        loss_original = super().loss(model, global_model, x, y)

        w_dif_cuadrado = 0

        for w_local, w_global in zip(model.parameters(), global_model.parameters()):
            w_dif_cuadrado += ((w_local-w_global.detach())**2).sum()

        parte_derecha = (self.mu/2) * w_dif_cuadrado

        return loss_original + parte_derecha


