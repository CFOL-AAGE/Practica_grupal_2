"""Tests propios del servidor (fl/server.py) y de state_dict_nbytes."""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset

from fl.clients import Client
from fl.codecs import Codec, state_dict_nbytes
from fl.server import LocalOnly, Server, sample_clients


def small_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 8), nn.ReLU(), nn.Linear(8, 10))


def small_data(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, 1, 28, 28, generator=g), torch.randint(0, 10, (n,), generator=g))


def make_server(server_cls=Server, fraction=0.5, codec=None):
    data = small_data(120)
    clients = [Client(i, Subset(data, range(i, 120, 4)), lr=0.05) for i in range(4)]
    return server_cls(small_model(), clients, small_data(50, seed=1), fraction=fraction, codec=codec, seed=0)


class HalfPrecision(Codec):
    """Codec de prueba: sube los pesos en float16 (la mitad de bytes)."""

    calls = 0

    def encode(self, state, reference):
        HalfPrecision.calls += 1
        return {k: v.half() for k, v in state.items()}

    def decode(self, payload, reference):
        return {k: v.float() for k, v in payload.items()}


# --- state_dict_nbytes -------------------------------------------------------------------------

def test_nbytes_de_un_modelo():
    assert state_dict_nbytes(small_model().state_dict()) == (784 * 8 + 8 + 8 * 10 + 10) * 4


def test_nbytes_vacio_y_escalar():
    assert state_dict_nbytes({}) == 0
    assert state_dict_nbytes({"x": torch.tensor(3, dtype=torch.int64)}) == 8


# --- sample_clients ----------------------------------------------------------------------------

def test_sample_clients_ordenados_y_de_tipo_int():
    selected = sample_clients(10, 0.5, np.random.default_rng(0))
    assert selected == sorted(selected)
    assert all(type(c) is int and 0 <= c < 10 for c in selected)


def test_sample_clients_reproducible():
    a = [sample_clients(10, 0.3, np.random.default_rng(4)) for _ in range(2)]
    assert a[0] == a[1]


def test_sample_clients_cambia_entre_rondas():
    rng = np.random.default_rng(0)
    rounds = {tuple(sample_clients(10, 0.5, rng)) for _ in range(20)}
    assert len(rounds) > 1, "cada ronda debe muestrear de nuevo"


def test_sample_clients_casos_limite():
    rng = np.random.default_rng(0)
    assert len(sample_clients(10, 0.0, rng)) == 1
    assert sample_clients(1, 0.5, rng) == [0]
    assert sample_clients(10, 1.0, rng) == list(range(10))


# --- Server.run_round --------------------------------------------------------------------------

def test_run_round_con_todos_los_clientes():
    server = make_server(fraction=1.0)
    row = server.run_round()
    model_bytes = state_dict_nbytes(server.model.state_dict())
    assert row["bytes_down"] == row["bytes_up"] == 4 * model_bytes
    assert np.isfinite(row["train_loss"]) and 0 <= row["test_acc"] <= 1


def test_run_round_devuelve_solo_las_columnas_de_metrics():
    row = make_server().run_round()
    assert set(row) == {"test_acc", "test_loss", "train_loss", "bytes_up", "bytes_down"}


def test_run_round_pasa_por_el_codec_y_mide_lo_que_se_sube():
    HalfPrecision.calls = 0
    server = make_server(fraction=0.5, codec=HalfPrecision())
    row = server.run_round()
    assert HalfPrecision.calls == 2, "cada cliente elegido debe codificar su subida"
    assert row["bytes_up"] * 2 == row["bytes_down"], "bytes_up se mide sobre el payload codificado"


def test_run_round_reproducible():
    rows = [make_server().run(rounds=2) for _ in range(2)]
    for a, b in zip(*rows):
        assert a["test_acc"] == b["test_acc"] and a["bytes_cum"] == b["bytes_cum"]


# --- LocalOnly ---------------------------------------------------------------------------------

def test_local_only_cada_cliente_conserva_su_modelo():
    server = make_server(LocalOnly)
    initial = copy.deepcopy(server.local_models[0].state_dict())
    server.run_round()
    after_one = copy.deepcopy(server.local_models[0].state_dict())
    server.run_round()
    after_two = server.local_models[0].state_dict()
    assert any(not torch.equal(initial[k], after_one[k]) for k in initial), "el modelo local debe entrenar"
    assert any(not torch.equal(after_one[k], after_two[k]) for k in initial), "y seguir desde donde lo dejó"


def test_local_only_no_toca_el_modelo_global():
    server = make_server(LocalOnly)
    before = copy.deepcopy(server.model.state_dict())
    server.run_round()
    assert all(torch.equal(before[k], v) for k, v in server.model.state_dict().items())
