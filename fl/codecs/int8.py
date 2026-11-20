import torch

from fl.codecs.base import Codec
from fl.models import StateDict


class int8CodeDecode(Codec):
    '''Funciones de encode y decode'''

    def encode(self, state:StateDict, reference:StateDict):
        '''El cliente comprime para enviar'''

        comprimido = {}

        for nombre, tensor in state.items():

            # Calculamos la dif con el global
            diff = tensor.float() - reference[nombre].float()

            # Escalamos
            v_max = diff.abs().max()

            if v_max == 0:
                escala = torch.tensor(1.0)
            else:
                escala = v_max/127

            # Cuantizamos
            estirado = diff / escala
            redondeo = torch.round(estirado)
            recortado = torch.clamp(redondeo, min = -127, max=127)

            v_int8 = recortado.to(torch.int8)

            # Guardamos
            comprimido[nombre] = v_int8
            comprimido[nombre + ".escala"] = escala

        return comprimido


    def decode(self, comprimido:StateDict, reference:StateDict):
        '''El servidor tiene que descomprimir lo que comprimió el encode'''

        pesos = {}

        for nombre, tensor_global in reference.items():

            v_int8 = comprimido[nombre]
            escala = comprimido[nombre + ".escala"]

            diff = v_int8 * escala

            pesos[nombre] = tensor_global + diff

        return pesos