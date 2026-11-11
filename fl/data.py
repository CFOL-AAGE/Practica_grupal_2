"""Datos: carga de Fashion-MNIST y reparto entre clientes.

Un *particionado* (``Partition``) es una lista con un array de índices por
cliente: ``partition[i]`` son las posiciones del conjunto de entrenamiento que
tiene el cliente ``i``. Todo particionado debe repartir cada índice a
exactamente un cliente (sin huecos ni solapes).
"""

from __future__ import annotations
import random

import numpy as np
from matplotlib.figure import Figure
from torch.utils.data import TensorDataset
from torchvision import datasets

from fl.utils import Config


Partition = list[np.ndarray]

FMNIST_MEAN, FMNIST_STD = 0.2860, 0.3530

# ---------------------------------------------------------------------------
# Carga (DADO)
# ---------------------------------------------------------------------------
def load_dataset(name: str, root: str) -> tuple[TensorDataset, TensorDataset]:
    """Descarga (la primera vez) el dataset y lo carga entero en memoria.

    Devuelve ``(train, test)``. Cada ejemplo es un par ``(x, y)``: ``x`` es
    una imagen normalizada de forma ``(1, 28, 28)`` e ``y`` su clase (0-9).
    Las etiquetas de todo el conjunto están en ``train.tensors[1]``.
    """
    if name != "fmnist":
        raise ValueError(f"Dataset desconocido: {name!r}")
    splits = (datasets.FashionMNIST(root, train=train, download=True)
              for train in (True, False))
    return tuple(_to_tensors(split) for split in splits)


def _to_tensors(split: datasets.FashionMNIST) -> TensorDataset:
    x = (
        split.data.
            unsqueeze(1).
            float().
            div(255).
            sub(FMNIST_MEAN).div(FMNIST_STD)
    )
    return TensorDataset(x, split.targets.long())


# ---------------------------------------------------------------------------
# Particionado
# ---------------------------------------------------------------------------
def make_partition(labels: np.ndarray, cfg: Config, seed: int) -> Partition:
    """
    Elige la estrategia de reparto según ``cfg.type``
    (sección ``partition`` de la config).
    """
    if cfg.type == "iid":
        return iid(labels, cfg.n_clients, seed)
    if cfg.type == "dirichlet":
        return dirichlet(labels, cfg.n_clients, cfg.alpha, seed)
    if cfg.type == "shards":
        return dirichlet_shards(labels, cfg.n_clients,
                                cfg.classes_per_client, cfg.alpha, seed)
    raise ValueError(f"Particionado desconocido: {cfg.type!r}")


def split_by_proportions(indices: np.ndarray, proportions: np.ndarray) -> list[np.ndarray]:
    """(DADO) Divide ``indices`` en ``len(proportions)`` trozos consecutivos
    de tamaño ≈ proporcional.

    >>> split_by_proportions(np.arange(10), np.array([0.5, 0.3, 0.2]))
    [array([0, 1, 2, 3, 4]), array([5, 6, 7]), array([8, 9])]
    """
    proportions = np.asarray(proportions, dtype=float) / np.sum(proportions)
    cuts = np.round(np.cumsum(proportions)[:-1] * len(indices)).astype(int)
    return np.split(indices, cuts)


def iid(labels: np.ndarray, n_clients: int, seed: int) -> Partition:
    """Reparto IID: cada cliente recibe una muestra aleatoria del mismo tamaño (±1).

    Args:
        labels: etiqueta de cada ejemplo de entrenamiento, forma ``(N,)``.
        n_clients: número de clientes.
        seed: semilla; la misma semilla debe dar el mismo reparto.
    """
    # TODO: baraja los índices 0..N-1 con un generador propio (np.random.default_rng(seed))
    #       y divídelos en n_clients trozos.
    N = len(labels)
    rng = np.random.default_rng(seed)
    indices = np.arange(N)
    # Realiza una permutación uniformemente aleatoria lo que aumenta la
    # probabilidad de obtener particiones IID.
    rng.shuffle(indices)

    # Distribución equitativa
    ratio, rest = N // n_clients, N % n_clients

    start = 0
    partitions = []
    # Recorrer el array de índices para repartir las particiones entre los clientes
    for i in range(n_clients):
        end = start + ratio + (i < rest)
        partitions.append(indices[start:end])
        start = end

    return partitions


def dirichlet(labels: np.ndarray, n_clients: int, alpha: float, seed: int, min_size: int = 10) -> Partition:
    """Reparto no-IID por Dirichlet (Hsu et al., 2019).

    Para cada clase se muestrean unas proporciones ``p ~ Dir(alpha, ..., alpha)`` (una por cliente)
    y los ejemplos de esa clase se reparten según ``p``. Con ``alpha`` pequeño cada cliente se queda
    con pocas clases; con ``alpha`` grande el reparto se parece al IID.
    Si algún cliente queda con menos de ``min_size`` ejemplos, se repite el muestreo.
    """
    # TODO: para cada clase, baraja sus índices, muestrea las proporciones con rng.dirichlet
    #       y repártelos con split_by_proportions. Comprueba min_size al final.
    N = len(labels)
    indexes = np.arange(N)
    repeat = True
    rng = np.random.default_rng(seed)

    while repeat:

        partitions = [np.array([], dtype=int) for _ in range(n_clients)]
        for clase in np.unique(labels):
            # Obtener los índices pertenecientes a la clase
            class_indexs = indexes[labels == clase]
            rng.shuffle(class_indexs) # Barajar los índices

            # Muestrear las proporciones
            props = rng.dirichlet([alpha for _ in range(n_clients)])

            class_partitions = split_by_proportions(indices=class_indexs, proportions=props)
            for i, split in enumerate(class_partitions):
                partitions[i] = np.concatenate((partitions[i], split))

        # Revisar que ningún cliente quedé con menos de `min_size` elementos
        repeat = any([len(p) < min_size for p in partitions])

    return partitions


def dirichlet_shards(labels: np.ndarray, n_clients: int, classes_per_client: int, alpha: float, seed: int) -> Partition:
    """Reparto por clases: cada cliente tiene solo ``classes_per_client`` clases (k).

    1. Se asignan k clases distintas a cada cliente, de modo que todas las clases tengan dueño
       (hace falta ``n_clients * k >= n_clases``).
    2. Dentro de cada cliente, el peso de cada una de sus k clases sigue ``Dir(alpha)``.
    3. Los ejemplos de cada clase se reparten entre sus dueños en proporción a esos pesos.
    """
    classes = np.unique(labels)
    n_classes, k = len(classes), classes_per_client
    if not 1 <= k <= n_classes:
        raise ValueError(f"classes_per_client debe estar entre 1 y {n_classes}, no {k}")
    if n_clients * k < n_classes:
        raise ValueError(
            f"Con {n_clients} clientes y {k} clases por cliente no se cubren las {n_classes} clases "
            f"(hace falta n_clients * k >= {n_classes})"
        )
    rng = np.random.default_rng(seed)

    # 1. Clases de cada cliente: k posiciones consecutivas de una permutación recorrida en círculo.
    #    Son distintas dentro de cada cliente (k <= n_clases), todas las clases tienen dueño
    #    (n_clients * k >= n_clases) y cada clase tiene el mismo nº de dueños (±1).
    perm = rng.permutation(classes)
    client_classes = [perm[(i * k + np.arange(k)) % n_classes] for i in range(n_clients)]

    # 2. Peso de cada una de sus k clases dentro de cada cliente: w_i ~ Dir(alpha, ..., alpha).
    weights = np.nan_to_num(rng.dirichlet(np.full(k, alpha), size=n_clients))
    weights[weights.sum(axis=1) == 0] = 1.0 / k  # guarda: con alpha muy pequeño puede salir 0/0

    # 3. Cada clase se reparte entre sus dueños en proporción a sus pesos para esa clase.
    #    Cada cliente tiene al menos una clase con peso >= 1/k, así que nunca queda vacío.
    chunks = [[] for _ in range(n_clients)]
    for c in classes:
        owners = [i for i in range(n_clients) if c in client_classes[i]]
        w = np.array([weights[i][np.flatnonzero(client_classes[i] == c)[0]] for i in owners])
        if w.sum() == 0:
            w = np.ones(len(owners))
        class_indices = rng.permutation(np.flatnonzero(labels == c))
        for i, part in zip(owners, split_by_proportions(class_indices, w)):
            chunks[i].append(part)

    return [np.concatenate(parts) for parts in chunks]

# def dirichlet_shards(labels: np.ndarray, n_clients: int, classes_per_client: int, alpha: float, seed: int) -> Partition:
#     """Reparto por clases: cada cliente tiene solo ``classes_per_client`` clases (k).

#     1. Se asignan k clases distintas a cada cliente, de modo que todas las clases tengan dueño
#        (hace falta ``n_clients * k >= n_clases``).
#     2. Dentro de cada cliente, el peso de cada una de sus k clases sigue ``Dir(alpha)``.
#     3. Los ejemplos de cada clase se reparten entre sus dueños en proporción a esos pesos.
#     """
#     # TODO: decide qué clases tiene cada cliente, muestrea sus pesos con rng.dirichlet y reparte
#     #       cada clase entre sus dueños (split_by_proportions). Ningún ejemplo debe quedar fuera.
#     clases = np.unique(labels)
#     indexes = np.arange( len(labels) )

#     # Tantos como k * n_clients.
#     rng = np.random.default_rng(seed)
#     slots = list(clases) # Se debe garantizar primero que todas las clases son cubiertas
#     remaning_slots = n_clients * classes_per_client - len(clases)
#     slots += list( rng.choice(clases, remaning_slots, replace=True) )
#     rng.shuffle(slots)

#     # Asignación
#     map_class_clients = {c: [] for c in clases} # {class_1: [client_0, client_1]}
#     cliente = 0
#     for start in range(0, len(slots), classes_per_client):
#         cls = slots[start: start + classes_per_client]
#         for c in cls:
#             map_class_clients[c].append(cliente)
#         cliente += 1

#     # Distribuir los ejemplos de las clases sobre los clientes
#     partitions = [np.array([], dtype=int) for _ in range(n_clients)]

#     for c in clases:
#         clientes = map_class_clients[c]
#         class_indexs = indexes[labels == c]
#         props = rng.dirichlet([alpha for _ in range(len(clientes))])

#         class_partitions = split_by_proportions(indices=class_indexs, proportions=props)
#         for i, cl in enumerate(clientes):
#             partitions[cl] = np.concatenate((partitions[cl], class_partitions[i]))

#     return partitions


def plot_partition(labels: np.ndarray, partition: Partition, path) -> None:
    """(DADO) Guarda un histograma apilado con el nº de ejemplos de cada clase por cliente."""
    n_classes = int(labels.max()) + 1
    counts = np.array([np.bincount(labels[idx], minlength=n_classes) for idx in partition])
    fig = Figure(figsize=(max(6, 0.5 * len(partition)), 4))
    ax = fig.subplots()
    bottom = np.zeros(len(partition))
    for c in range(n_classes):
        ax.bar(range(len(partition)), counts[:, c], bottom=bottom, label=str(c), color=f"C{c}")
        bottom += counts[:, c]
    ax.set(xlabel="cliente", ylabel="nº de ejemplos", xticks=range(len(partition)))
    ax.legend(title="clase", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
