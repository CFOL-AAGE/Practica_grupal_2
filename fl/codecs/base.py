"""Comunicación: interfaz de los codecs y medida de lo que ocupa un modelo."""

from __future__ import annotations

import torch

from fl.models import StateDict


def state_dict_nbytes(state_dict: StateDict) -> int:
    """Bytes que ocupan todos los tensores de ``state_dict``."""
    total_size = 0

    for tensor in state_dict.values():
        # Tomar el tamaño en bytes de un elemento en el tensor y multiplicarlo
        # por el número de elementos del mismo.
        total_size += tensor.element_size() * tensor.numel()

    return total_size


class Codec:
    """Transforma lo que un cliente envía al servidor. Este, el de por defecto, no comprime nada.

    - ``encode(state, reference)``: en el cliente; devuelve el ``payload`` que viaja por la red.
    - ``decode(payload, reference)``: en el servidor; reconstruye los pesos a partir del ``payload``.

    ``reference`` es el modelo global de la ronda (lo conocen cliente y servidor).
    """

    def encode(self, state: StateDict, reference: StateDict) -> StateDict:
        return state

    def decode(self, payload: StateDict, reference: StateDict) -> StateDict:
        return payload
