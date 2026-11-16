"""Tests propios de los modelos (fl/models.py)."""

import torch

import pytest

from fl.models import CNN, CNN_MAX_PARAMS, MLP, CNN3x3, CNNGroupNorm, build_model, count_params
from fl.utils import load_config


def test_mlp_tiene_los_parametros_de_las_diapositivas():
    assert count_params(MLP()) == 199_210  # 784·200+200 + 200·200+200 + 200·10+10


def test_mlp_respeta_hidden():
    assert count_params(MLP(hidden=16)) == 784 * 16 + 16 + 16 * 16 + 16 + 16 * 10 + 10


def test_salidas_con_un_solo_ejemplo():
    x = torch.randn(1, 1, 28, 28)
    assert MLP()(x).shape == (1, 10)
    assert CNN()(x).shape == (1, 10)


def test_cnn_dentro_del_limite():
    assert count_params(CNN()) == 83_466 <= CNN_MAX_PARAMS


def test_state_dict_solo_tiene_parametros_float():
    # Sin BatchNorm no hay buffers (running_mean, num_batches_tracked): FedAvg e int8 solo ven floats.
    for model in (MLP(), CNN()):
        state = model.state_dict()
        assert set(state) == {name for name, _ in model.named_parameters()}
        assert all(t.dtype == torch.float32 for t in state.values())


@pytest.mark.parametrize("name, cls, params", [
    ("cnn", CNN, 83_466), ("cnn3x3", CNN3x3, 50_186), ("cnn_gn", CNNGroupNorm, 83_658), ("mlp", MLP, 199_210),
])
def test_build_model(name, cls, params):
    model = build_model(load_config(overrides=[f"model.name={name}"]).model)
    assert isinstance(model, cls) and count_params(model) == params
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


def test_variantes_sin_buffers_enteros():
    # GroupNorm, a diferencia de BatchNorm, no guarda estadísticas: FedAvg solo promedia parámetros.
    for model in (CNN3x3(), CNNGroupNorm()):
        assert set(model.state_dict()) == {name for name, _ in model.named_parameters()}


def test_modelo_desconocido():
    with pytest.raises(ValueError):
        build_model(load_config(overrides=["model.name=resnet"]).model)
