"""Cliente base: entrena el modelo global con sus datos locales y devuelve su actualización."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fl.models import StateDict


@dataclass
class ClientUpdate:
    """Lo que un cliente envía al servidor después de entrenar."""

    cid: int
    state_dict: StateDict  # pesos del modelo local tras entrenar
    n_samples: int  # nº de ejemplos locales (el peso del cliente en FedAvg)
    metrics: dict[str, float] = field(default_factory=dict)  # p. ej. {"train_loss": 0.53}


class Client:
    """Un participante de la federación (el cliente de FedAvg).

    Para cambiar cómo entrena un cliente, heredad de esta clase y redefinid ``loss``
    (como ``FedProxClient``) o ``local_train``.
    """

    def __init__(
        self,
        cid: int,
        dataset: Dataset,
        *,
        local_epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.01,
        device: str = "cpu",
    ):
        self.cid = cid
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    @property
    def n_samples(self) -> int:
        return len(self.dataset)

    def loss(self, model: nn.Module, global_model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """(DADO) Pérdida de un mini-lote: entropía cruzada. ``global_model`` es el modelo recibido."""
        return F.cross_entropy(model(x), y)

    def local_train(self, global_model: nn.Module) -> ClientUpdate:
        """Entrena una copia del modelo global con los datos locales.

        Es el equivalente federado de ``model.fit(x_local, y_local, epochs=E, batch_size=B)``:
          1. Copia el modelo global (``copy.deepcopy``): ``global_model`` no debe modificarse.
          2. Entrena ``local_epochs`` épocas con SGD (``lr``) en mini-lotes barajados de
             ``batch_size`` ejemplos, usando ``self.loss(model, global_model, x, y)`` como pérdida
             (así las subclases solo tienen que cambiar la pérdida).
          3. Devuelve un ``ClientUpdate`` con el ``state_dict`` final, ``self.n_samples`` y la
             pérdida media de entrenamiento en ``metrics["train_loss"]``.
        """
        pesos, metrics, perdidas = {}, {}, []

        # Copiamos modelo general y def necesario
        modelo_local = copy.deepcopy(global_model)
        optimizador = torch.optim.SGD(modelo_local.parameters(), lr=self.lr)
        lotes = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)

        # Entrenamos
        modelo_local.train()
        for epoch in range(self.local_epochs):

            for x, y in lotes:

                x, y = x.to(self.device), y.to(self.device)

                # Entrenamos
                optimizador.zero_grad()
                perdida = self.loss(model=modelo_local, global_model=global_model, x=x, y=y)
                perdida.backward()
                optimizador.step()

                # Guardamos la perdida
                perdidas.append(perdida.item())

        for nombre, tensor in modelo_local.state_dict().items():
            pesos[nombre] = tensor.detach().clone()

        if len(perdidas) > 0:
            metrics["train_loss"] = sum(perdidas)/len(perdidas)
        else:
            metrics["train_loss"] = 0

        return ClientUpdate(cid=self.cid, state_dict=pesos, n_samples=self.n_samples, metrics=metrics)





