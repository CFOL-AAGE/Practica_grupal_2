"""Tests propios del particionado (fl/data.py): casos que los públicos no cubren."""

import numpy as np
import pytest

from fl.data import dirichlet, dirichlet_shards, iid

# Mismo tamaño por clase que Fashion-MNIST (6000), desordenado.
LABELS = np.random.default_rng(1).permutation(np.repeat(np.arange(10), 6000))


def assert_is_partition(partition, labels, n_clients):
    assert len(partition) == n_clients
    all_indices = np.concatenate(partition)
    assert np.issubdtype(all_indices.dtype, np.integer), "los índices deben ser enteros"
    assert len(all_indices) == len(np.unique(all_indices)) == len(labels)
    assert all_indices.min() == 0 and all_indices.max() == len(labels) - 1


def same_partition(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


# --- IID ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("n, n_clients", [(6001, 10), (1000, 7), (10, 10), (50, 1)])
def test_iid_tamanos_no_divisibles(n, n_clients):
    labels = np.arange(n) % 10
    partition = iid(labels, n_clients, seed=0)
    assert_is_partition(partition, labels, n_clients)
    sizes = [len(p) for p in partition]
    assert max(sizes) - min(sizes) <= 1


def test_iid_reproducible_y_depende_de_la_semilla():
    assert same_partition(iid(LABELS, 10, seed=3), iid(LABELS, 10, seed=3))
    assert not same_partition(iid(LABELS, 10, seed=3), iid(LABELS, 10, seed=4))


# --- Dirichlet ---------------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.05, 0.1, 1.0, 100.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dirichlet_es_particion_y_respeta_min_size(alpha, seed):
    partition = dirichlet(LABELS, 10, alpha=alpha, seed=seed)
    assert_is_partition(partition, LABELS, 10)
    assert min(len(p) for p in partition) >= 10


def test_dirichlet_reproducible():
    assert same_partition(dirichlet(LABELS, 10, 0.1, seed=5), dirichlet(LABELS, 10, 0.1, seed=5))


def test_dirichlet_alpha_grande_se_parece_a_iid():
    for idx in dirichlet(LABELS, 10, alpha=100.0, seed=0):
        freqs = np.bincount(LABELS[idx], minlength=10) / len(idx)
        assert freqs.max() < 0.2, "con alpha grande ninguna clase debería dominar"


# --- Shards (k clases por cliente) -------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("alpha", [0.1, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_shards_particion_sin_clientes_vacios(k, alpha, seed):
    partition = dirichlet_shards(LABELS, 10, classes_per_client=k, alpha=alpha, seed=seed)
    assert_is_partition(partition, LABELS, 10)
    for idx in partition:
        assert len(idx) >= 10, "ningún cliente puede quedar vacío o casi vacío"
        assert len(np.unique(LABELS[idx])) <= k


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_shards_k_clases_distintas(k):
    # Con alpha grande los pesos son casi iguales: cada cliente debe tener exactamente k clases.
    for idx in dirichlet_shards(LABELS, 10, classes_per_client=k, alpha=100.0, seed=0):
        assert len(np.unique(LABELS[idx])) == k


def test_shards_k1_cada_cliente_una_clase_entera():
    partition = dirichlet_shards(LABELS, 10, classes_per_client=1, alpha=0.1, seed=0)
    assert sorted(int(LABELS[idx][0]) for idx in partition) == list(range(10))
    assert all(len(idx) == 6000 for idx in partition)


def test_shards_reproducible():
    a = dirichlet_shards(LABELS, 10, 2, 0.1, seed=7)
    b = dirichlet_shards(LABELS, 10, 2, 0.1, seed=7)
    assert same_partition(a, b)


@pytest.mark.parametrize("n_clients, k", [(3, 2), (4, 2), (10, 0), (10, 11)])
def test_shards_configuraciones_imposibles(n_clients, k):
    with pytest.raises(ValueError):
        dirichlet_shards(LABELS, n_clients, classes_per_client=k, alpha=1.0, seed=0)
