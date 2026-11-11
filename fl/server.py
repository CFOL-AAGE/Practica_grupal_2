"""Servidor: selecciona clientes, agrega sus actualizaciones y evalúa el modelo global.

Los agregadores están en ``fl/aggregators/`` y los codecs en ``fl/codecs/``.
"""

from __future__ import annotations

import copy
import time

import numpy as np
from torch import nn
from torch.utils.data import Dataset

from fl.aggregators import Aggregator, FedAvg
from fl.clients import Client
from fl.codecs import Codec, state_dict_nbytes
from fl.models import evaluate
from fl.utils import RunLogger


def sample_clients(n_clients: int, fraction: float, rng: np.random.Generator) -> list[int]:
    """Elige al azar, sin repetición, ``max(1, round(fraction * n_clients))`` clientes.

    Devuelve sus identificadores (0..n_clients-1) ordenados de menor a mayor.
    """
    # min(): con fraction > 1 no se pueden elegir más clientes de los que hay.
    n_selected = min(n_clients, max(1, round(fraction * n_clients)))
    selected = rng.choice(n_clients, size=n_selected, replace=False)
    return sorted(int(cid) for cid in selected)  # int de Python, no np.int64

class Server:
    """Orquesta el entrenamiento federado: ``run`` (dado) repite ``run_round`` (vuestro)."""

    def __init__(
        self,
        model: nn.Module,
        clients: list[Client],
        test_set: Dataset,
        *,
        fraction: float = 1.0,
        aggregator: Aggregator | None = None,
        codec: Codec | None = None,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.clients = clients
        self.test_set = test_set
        self.fraction = fraction
        self.aggregator = aggregator or FedAvg()
        self.codec = codec or Codec()
        self.rng = np.random.default_rng(seed)
        self.device = device

    def run(self, rounds: int, logger: RunLogger | None = None) -> list[dict]:
        """Ejecuta ``rounds`` rondas y devuelve (y registra) una fila de métricas por ronda."""
        history, bytes_cum, start = [], 0, time.perf_counter()
        for rnd in range(1, rounds + 1):
            row = {"round": rnd, **self.run_round()}
            bytes_cum += row["bytes_up"] + row["bytes_down"]
            row.update(bytes_cum=bytes_cum, time_s=round(time.perf_counter() - start, 2))
            history.append(row)
            if logger:
                logger.log(row)
        return history


    def run_round(self) -> dict:
        """Una ronda de FL: el servidor elige clientes, estos entrenan a partir del modelo global y
        envían su actualización (a través de ``self.codec``), el servidor las agrega y evalúa el
        nuevo modelo global sobre ``self.test_set``.

        Devuelve un diccionario con ``test_acc``, ``test_loss``, ``train_loss`` (media de los
        clientes) y los bytes enviados en la ronda: ``bytes_up`` (clientes -> servidor) y
        ``bytes_down`` (servidor -> clientes).
        """
        # 1. El servidor elige los clientes de esta ronda.
        selected = sample_clients(len(self.clients), self.fraction, self.rng)

        # 2. Copia del modelo global: es la referencia que conocen cliente y servidor (codec).
        global_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        # 3. Cada cliente entrena y sube su modelo codificado; el servidor lo decodifica.
        updates, bytes_up = [], 0
        for cid in selected:
            update = self.clients[cid].local_train(self.model)
            payload = self.codec.encode(update.state_dict, global_state)
            bytes_up += state_dict_nbytes(payload)
            update.state_dict = self.codec.decode(payload, global_state)
            updates.append(update)

        # 4. Agregación y evaluación del nuevo modelo global.
        self.model.load_state_dict(self.aggregator.aggregate(updates))
        metrics = evaluate(self.model, self.test_set, self.device)

        return {
            "test_acc": metrics["acc"],
            "test_loss": metrics["loss"],
            "train_loss": sum(u.metrics["train_loss"] for u in updates) / len(updates),
            "bytes_up": bytes_up,
            "bytes_down": len(selected) * state_dict_nbytes(global_state),  # el global baja a cada uno
        }

class LocalOnly(Server):
    """Referencia «suelo»: cada cliente entrena solo con sus datos y nunca se comunica.

    Cada cliente conserva su propio modelo entre rondas (todos parten del mismo modelo inicial).
    En cada ronda entrenan todos y se registra la media de sus métricas sobre el test global.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_models = [copy.deepcopy(self.model) for _ in self.clients]

    def run_round(self) -> dict:
        # Cada cliente entrena SU modelo (el que conserva entre rondas) solo con sus datos.
        accs, losses, train_losses = [], [], []
        for client, local_model in zip(self.clients, self.local_models):
            update = client.local_train(local_model)
            local_model.load_state_dict(update.state_dict)  # local_train trabaja sobre una copia
            metrics = evaluate(local_model, self.test_set, self.device)
            accs.append(metrics["acc"])
            losses.append(metrics["loss"])
            train_losses.append(update.metrics["train_loss"])

        n = len(self.clients)
        return {
            "test_acc": sum(accs) / n,           # media de los modelos locales sobre el test global
            "test_loss": sum(losses) / n,
            "train_loss": sum(train_losses) / n,
            "bytes_up": 0,                       # sin comunicación
            "bytes_down": 0,
        }