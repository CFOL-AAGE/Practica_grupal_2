import math

import torch

from fl.codecs.base import Codec
from fl.models import StateDict


class topkCodeDecode(Codec):
    '''Funciones de encode y decode: solo se envían las componentes más grandes'''

    def __init__(self, fraccion: float = 0.1):

        if not 0 < fraccion <= 1:
            raise ValueError(f"topk_fraction debe estar en (0, 1], no {fraccion}")

        self.fraccion = fraccion

    def encode(self, state:StateDict, reference:StateDict):
        '''El cliente se queda con las componentes más grandes pa enviar'''

        comprimido = {}

        for nombre, tensor in state.items():

            # Los tensores que no son float (p. ej. contadores) se envían tal cual
            if not tensor.is_floating_point():
                comprimido[nombre] = tensor
                continue

            # Calculamos la dif con el global (en una sola dimensión)
            diff = (tensor - reference[nombre]).flatten()

            # Cuántas componentes enviamos (al menos una)
            k = max(1, math.ceil(self.fraccion * diff.numel()))

            # Nos quedamos con las k de mayor valor absoluto
            posiciones = diff.abs().topk(k).indices
            valores = diff[posiciones]

            # Guardamos posiciones (int32) y valores (float32): 8 bytes por componente
            comprimido[nombre + ".posiciones"] = posiciones.to(torch.int32)
            comprimido[nombre + ".valores"] = valores.to(torch.float32)

        return comprimido


    def decode(self, comprimido:StateDict, reference:StateDict):
        '''El servidor reconstruye los pesos: las componentes no enviadas valen 0'''

        pesos = {}

        for nombre, tensor_global in reference.items():

            if not tensor_global.is_floating_point():
                pesos[nombre] = comprimido[nombre]
                continue

            posiciones = comprimido[nombre + ".posiciones"].long()
            valores = comprimido[nombre + ".valores"]

            # Rellenamos con ceros y colocamos los valores recibidos en sus posiciones
            diff = torch.zeros(tensor_global.numel(), dtype=tensor_global.dtype, device=tensor_global.device)
            diff[posiciones] = valores.to(tensor_global.dtype)

            pesos[nombre] = tensor_global + diff.view_as(tensor_global)

        return pesos