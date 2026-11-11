"""FedAvg (McMahan et al., 2017): media ponderada por el nº de ejemplos de cada cliente."""

from __future__ import annotations

from fl.aggregators.base import Aggregator
from fl.clients import ClientUpdate
from fl.models import StateDict

import torch

class FedAvg(Aggregator):
    def aggregate(self, updates: list[ClientUpdate]) -> StateDict:
        """Devuelve, para cada tensor, ``sum_k (n_k / n) * w_k`` con ``n = sum_k n_k``.

        - Conserva las claves, su orden y el ``dtype`` de cada tensor.
        - Los tensores enteros (p. ej. ``num_batches_tracked`` de BatchNorm) no pueden quedar
          como float: redondea la media y vuelve al tipo original.
        """
        if not updates:
            raise ValueError("FedAvg necesita al menos una actualización")
        n_total = sum(u.n_samples for u in updates)
        if n_total <= 0:
            raise ValueError("FedAvg: los clientes de la ronda no tienen ejemplos")
        weights = [u.n_samples / n_total for u in updates]

        aggregated: StateDict = {}
        for key, reference in updates[0].state_dict.items():
            # Se acumula en float64 para no perder precisión al sumar muchos clientes.
            avg = torch.zeros_like(reference, dtype=torch.float64)
            for w, u in zip(weights, updates):
                avg += w * u.state_dict[key].to(torch.float64)
            if reference.dtype.is_floating_point:
                aggregated[key] = avg.to(reference.dtype)
            else:
                aggregated[key] = avg.round().to(reference.dtype)
        return aggregated