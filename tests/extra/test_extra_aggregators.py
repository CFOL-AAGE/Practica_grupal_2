"""Tests propios de FedAvg (fl/aggregators/fedavg.py)."""

import pytest
import torch

from fl.aggregators import FedAvg
from fl.clients import ClientUpdate


def update(cid, n, **tensors):
    return ClientUpdate(cid=cid, state_dict=dict(tensors), n_samples=n)


def test_tres_clientes_ponderados():
    updates = [
        update(0, 1, w=torch.tensor([1.0])),
        update(1, 2, w=torch.tensor([4.0])),
        update(2, 7, w=torch.tensor([10.0])),
    ]
    result = FedAvg().aggregate(updates)
    assert torch.allclose(result["w"], torch.tensor([(1 * 1 + 2 * 4 + 7 * 10) / 10]))


def test_un_solo_cliente_devuelve_sus_pesos():
    w = torch.randn(3, 4)
    result = FedAvg().aggregate([update(0, 25, w=w)])
    assert torch.allclose(result["w"], w)


def test_conserva_claves_orden_forma_y_dtype():
    a = update(0, 1, z=torch.zeros(2, 3), a=torch.zeros(4, dtype=torch.float64), m=torch.zeros(1, dtype=torch.float16))
    b = update(1, 1, z=torch.ones(2, 3), a=torch.ones(4, dtype=torch.float64), m=torch.ones(1, dtype=torch.float16))
    result = FedAvg().aggregate([a, b])
    assert list(result) == ["z", "a", "m"]
    for key in result:
        assert result[key].shape == a.state_dict[key].shape
        assert result[key].dtype == a.state_dict[key].dtype


def test_tensores_enteros_se_redondean_y_siguen_siendo_enteros():
    a = update(0, 1, steps=torch.tensor(1))
    b = update(1, 2, steps=torch.tensor(4))  # media ponderada = 3
    result = FedAvg().aggregate([a, b])
    assert result["steps"].dtype == torch.int64
    assert result["steps"].item() == 3


def test_no_modifica_las_actualizaciones():
    a = update(0, 1, w=torch.tensor([1.0, 2.0]))
    b = update(1, 1, w=torch.tensor([3.0, 4.0]))
    before = [u.state_dict["w"].clone() for u in (a, b)]
    result = FedAvg().aggregate([a, b])
    result["w"] += 100  # tocar el resultado tampoco puede cambiar las entradas
    assert torch.equal(a.state_dict["w"], before[0]) and torch.equal(b.state_dict["w"], before[1])


def test_sin_ejemplos_lanza_error():
    with pytest.raises(ValueError):
        FedAvg().aggregate([update(0, 0, w=torch.zeros(1))])
