"""Experimentos de la sección 4 del enunciado.

Cada pregunta es una lista de ejecuciones, y cada ejecución es la lista de --set que la
diferencia de configs/default.yaml (la semilla incluida). Dentro de una pregunta, todas las
configuraciones que se comparan usan las mismas semillas.

Una misma ejecución puede aparecer en varias preguntas (FedAvg con alpha=0.1, por ejemplo, sirve
para Q1, Q2, Q3, Q4 y Q5). sweep.py la reconoce por su nombre en results/ y solo la lanza una vez.

Los decimales se escriben siempre con punto y cifra decimal ("1.0", no "1"): el nombre de la
carpeta sale de los valores y "a1" y "a1.0" serían dos ejecuciones distintas.

"cnn" no es una pregunta del enunciado: es la comparación de variantes de la CNN de
decisiones/01-arquitectura-cnn.md (sección 6).
"""

SEEDS_3 = [0, 1, 2]
SEEDS_5 = [0, 1, 2, 3, 4]

# Repartos de los datos
IID = ["partition.type=iid"]
DIR_1 = ["partition.type=dirichlet", "partition.alpha=1.0"]
DIR_01 = ["partition.type=dirichlet", "partition.alpha=0.1"]
SHARDS_K2 = ["partition.type=shards", "partition.classes_per_client=2", "partition.alpha=0.1"]

# Referencias
CENTRALIZED = ["partition.type=iid", "partition.n_clients=1", "server.fraction=1.0"]
LOCAL_ONLY = ["mode=local_only", "partition.type=dirichlet", "partition.alpha=0.1"]

E5 = ["client.local_epochs=5"]
INT8 = ["server.codec=int8"]
TOPK = ["server.codec=topk"]
CNN = ["model.name=cnn"]
MUS = ["0.01", "0.1", "1.0"]


def with_seeds(overrides, seeds):
    return [overrides + [f"seed={s}"] for s in seeds]


def heterogeneity(model, seeds_iid, seeds_01):
    """Q1 (y Q6 con la CNN): los cuatro repartos y las dos referencias."""
    return (
        with_seeds(model + IID, seeds_iid)
        + with_seeds(model + DIR_1, SEEDS_3)
        + with_seeds(model + DIR_01, seeds_01)
        + with_seeds(model + SHARDS_K2, seeds_01)
        + with_seeds(model + CENTRALIZED, SEEDS_3)
        + with_seeds(model + LOCAL_ONLY, SEEDS_3)
    )


EXPERIMENTS = {
    # Q1. IID, alpha=1, alpha=0.1 y k=2, con centralizado y solo local.
    #     IID, alpha=0.1 y k=2 llevan 5 semillas porque se reutilizan en Q4 y Q5.
    "q1": heterogeneity([], SEEDS_5, SEEDS_5),
    # Q2. E=1 frente a E=5, con datos IID y no-IID.
    "q2": (
        with_seeds(IID, SEEDS_3)
        + with_seeds(IID + E5, SEEDS_3)
        + with_seeds(DIR_01, SEEDS_5)
        + with_seeds(DIR_01 + E5, SEEDS_5)
    ),
    # Q3. Fracción de clientes por ronda C = 0.3, 0.5, 1.0.
    "q3": [
        run
        for c in ["0.3", "0.5", "1.0"]
        for run in with_seeds(DIR_01 + [f"server.fraction={c}"], SEEDS_3)
    ],
    # Q4. FedAvg (mu=0) frente a FedProx con tres valores de mu, en tres repartos con E=1
    #     y en alpha=0.1 con E=5, donde la deriva de los clientes es mayor.
    "q4": [
        run
        for data in [IID, DIR_01, SHARDS_K2, DIR_01 + E5]
        for mu in ["0.0"] + MUS
        for run in with_seeds(data + ([f"client.fedprox_mu={mu}"] if mu != "0.0" else []), SEEDS_5)
    ],
    # Q5. Con y sin cuantización int8 de la subida.
    "q5": [run for data in [DIR_01, IID] for codec in [[], INT8] for run in with_seeds(data + codec, SEEDS_5)],
    # Q6. El experimento 1 con la CNN, y las mismas ejecuciones del MLP para comparar.
    "q6": heterogeneity(CNN, SEEDS_3, SEEDS_3) + heterogeneity([], SEEDS_3, SEEDS_3),
    # Variantes de la CNN: la elegida (5x5), filtros 3x3 y con GroupNorm. Las de "cnn" se comparten con Q6.
    "cnn": [
        run
        for data in [IID, DIR_01]
        for name in ["cnn", "cnn3x3", "cnn_gn"]
        for run in with_seeds(data + [f"model.name={name}"], SEEDS_3)
    ],

    # Extensión: top-k frente a int8 y sin comprimir. Las de float32 e int8 se comparten con Q5.
    "topk": [run for data in [DIR_01, IID] for codec in [[], INT8, TOPK] for run in with_seeds(data + codec, SEEDS_5)],
    # Extensión: más clientes, siempre con 5 participantes por ronda (C·K = 5), para que solo
    # cambie cuántos datos tiene cada cliente y no cuánto se comunica por ronda.
    "clientes": [
        run
        for n, c in [("10", "0.5"), ("50", "0.1"), ("100", "0.05")]
        for run in with_seeds(DIR_01 + [f"partition.n_clients={n}", f"server.fraction={c}"], SEEDS_3)
    ],
}
