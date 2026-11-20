"""Tests propios del codec top-k (fl/codecs/topk.py)."""

import pytest
import torch

from fl.codecs import build_codec, state_dict_nbytes
from fl.codecs.topk import topkCodeDecode
from fl.models import MLP
from fl.utils import load_config


def test_se_activa_desde_la_configuracion():
    codec = build_codec(load_config(overrides=["server.codec=topk", "server.topk_fraction=0.2"]).server)
    assert isinstance(codec, topkCodeDecode) and codec.fraction == 0.2


def test_subida_ocupa_el_doble_de_la_fraccion():
    reference = MLP().state_dict()
    state = {k: v + 0.01 * torch.randn_like(v) for k, v in reference.items()}
    payload = topkCodeDecode(0.1).encode(state, reference)
    # cada componente enviada ocupa 8 bytes (valor + posición) frente a 4: con 10 % se sube ~20 %
    assert state_dict_nbytes(payload) / state_dict_nbytes(state) == pytest.approx(0.2, abs=0.001)


def test_recupera_las_componentes_grandes_y_anula_el_resto():
    reference = {"w": torch.zeros(10)}
    state = {"w": torch.tensor([0.0, 5.0, 0.1, -7.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0])}
    decoded = topkCodeDecode(0.2).decode(topkCodeDecode(0.2).encode(state, reference), reference)
    assert torch.equal(decoded["w"], torch.tensor([0.0, 5.0, 0, -7.0, 0, 0, 0, 0, 0, 0]))


def test_con_fraccion_1_es_exacto_y_conserva_claves_forma_y_dtype():
    reference = MLP(hidden=8).state_dict()
    state = {k: v + torch.randn_like(v) for k, v in reference.items()}
    decoded = topkCodeDecode(1.0).decode(topkCodeDecode(1.0).encode(state, reference), reference)
    assert list(decoded) == list(state)
    for k in state:
        assert decoded[k].shape == state[k].shape and decoded[k].dtype == state[k].dtype
        assert torch.allclose(decoded[k], state[k], atol=1e-6)


def test_tensores_enteros_pasan_sin_tocar():
    reference = {"steps": torch.tensor(3), "w": torch.zeros(4)}
    state = {"steps": torch.tensor(8), "w": torch.ones(4)}
    decoded = topkCodeDecode(0.5).decode(topkCodeDecode(0.5).encode(state, reference), reference)
    assert decoded["steps"].dtype == torch.int64 and decoded["steps"].item() == 8


@pytest.mark.parametrize("fraction", [0, -0.1, 1.5])
def test_fraccion_invalida(fraction):
    with pytest.raises(ValueError):
        topkCodeDecode(fraction)