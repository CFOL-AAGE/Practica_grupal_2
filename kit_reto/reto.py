"""Reto de aprendizaje federado: simula el escenario del reto con el código de la práctica y prepara el envío.

    python reto.py                                  # simula con configs/reto.yaml (1 semilla)
    python reto.py --seeds 0 1 2 --set client.lr=0.05 server.aggregator=mi_agregador
    python reto.py --pack                           # comprueba el código y crea envio.zip

Usa las piezas de la práctica tal cual: el cliente de ``build_clients``, el agregador de ``build_aggregator``,
el codec de ``build_codec`` y el modelo de ``fl/reto_models.py``. El escenario (clientes, rondas,
participación, reparto, abandonos y presupuesto) lo fija este archivo y no se puede cambiar desde la
configuración. El evaluador de la web ejecuta este mismo código (y añade clientes maliciosos).
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

HERE = Path(__file__).resolve().parent


@dataclass
class Scenario:
    n_clients: int = 100
    fraction: float = 0.1        # clientes elegidos por ronda: 10
    rounds: int = 20
    alpha: float = 0.05          # Dirichlet del reparto (muy no-IID)
    dropout: float = 0.2         # probabilidad de que un cliente elegido no responda
    budget_bytes: int = 163_840  # máximo que sube un cliente en una ronda (160 KiB)
    seed: int = 0


class StrategyError(Exception):
    """Error atribuible al código del grupo (excepción, formato o presupuesto)."""


# ---------------------------------------------------------------------------
# Reglas del código (las mismas que se comprueban al enviar)
# ---------------------------------------------------------------------------
REPLACED = {"fl/utils.py", "fl/data.py"}  # el evaluador usa la versión original de estos dos
MAX_ZIP_BYTES = 300_000
ALLOWED_IMPORTS = {
    "fl", "torch", "numpy", "math", "copy", "collections", "itertools", "functools", "random",
    "dataclasses", "typing", "heapq", "statistics", "operator", "enum", "abc", "__future__", "time",
    "warnings",
}
FORBIDDEN_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "breakpoint",
    "input", "getattr", "setattr", "delattr", "__builtins__",
}
FORBIDDEN_ATTRS = {
    "__globals__", "__dict__", "__subclasses__", "__bases__", "__mro__", "__code__", "__closure__",
    "__builtins__", "__loader__", "__spec__", "__import__", "f_back", "f_globals", "f_locals",
    "gi_frame", "cr_frame", "tb_frame", "_getframe",
    "load", "save", "loadtxt", "savetxt", "fromfile", "tofile", "genfromtxt", "memmap", "hub",
    "cpp_extension", "ops", "from_file", "jit", "compile", "_C", "ctypeslib", "distributed",
    "multiprocessing", "open_memmap", "DataSource", "npyio", "system", "popen",
}


def check_source(source: str, name: str = "") -> list[str]:
    """Problemas de un archivo .py (lista vacía si cumple las reglas)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"{name}:{e.lineno}: error de sintaxis: {e.msg}"]
    problems = []
    for node in ast.walk(tree):
        where = f"{name}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] not in ALLOWED_IMPORTS or FORBIDDEN_ATTRS & set(parts):
                    problems.append(f"{where}: import {a.name} no permitido")
        elif isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".")
            if node.level or parts[0] not in ALLOWED_IMPORTS or FORBIDDEN_ATTRS & (set(parts) | {a.name for a in node.names}):
                problems.append(f"{where}: from {node.module or '.'} import ... no permitido")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"{where}: no se permite usar {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            problems.append(f"{where}: no se permite acceder a .{node.attr}")
    return problems


def submission_files(repo: Path) -> dict[str, Path]:
    """Lo que entra en el envío: fl/**/*.py y los dos YAML de configs/."""
    files = {p.relative_to(repo).as_posix(): p for p in sorted((repo / "fl").rglob("*.py"))
             if "__pycache__" not in p.parts}
    for name in ("configs/default.yaml", "configs/reto.yaml"):
        files[name] = repo / name
    return files


def check_files(files: dict[str, bytes]) -> list[str]:
    """Comprueba el contenido de un envío ({ruta: bytes})."""
    problems = []
    for need in ("fl/reto_models.py", "configs/default.yaml", "configs/reto.yaml"):
        if need not in files:
            problems.append(f"falta {need}")
    for name, raw in files.items():
        ok_path = (name.startswith("fl/") and name.endswith(".py")) or name in ("configs/default.yaml", "configs/reto.yaml")
        if not ok_path or ".." in name.split("/"):
            problems.append(f"{name}: solo se aceptan fl/**/*.py, configs/default.yaml y configs/reto.yaml")
        elif name.endswith(".py") and name not in REPLACED:
            problems += check_source(raw.decode("utf-8", errors="replace"), name)
    return problems


# ---------------------------------------------------------------------------
# Comunicación
# ---------------------------------------------------------------------------
ALLOWED_DTYPES = {
    torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int8, torch.uint8,
    torch.int16, torch.int32, torch.int64, torch.bool,
}


def payload_nbytes(payload: dict) -> int:
    return sum(t.numel() * t.element_size() for t in payload.values())


def check_payload(payload, budget: int | None = None) -> dict:
    """Lo que sube un cliente (la salida de codec.encode) debe ser dict[str, Tensor] y caber."""
    if not isinstance(payload, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in payload.items()):
        raise StrategyError("codec.encode debe devolver un dict[str, Tensor]")
    for k, v in payload.items():
        if v.dtype not in ALLOWED_DTYPES:
            raise StrategyError(f"codec.encode: tipo {v.dtype} no permitido en {k!r}")
    n = payload_nbytes(payload)
    if budget is not None and n > budget:
        raise StrategyError(f"un cliente intenta subir {n:,} bytes y el presupuesto es {budget:,} bytes por ronda. "
                            "Hace falta un modelo más pequeño o comprimir lo que se sube (codec).")
    return payload


def _clone(sd: dict) -> dict:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Código del alumno, envuelto
# ---------------------------------------------------------------------------
class Session:
    """Carga el paquete ``fl`` del repositorio ``repo`` y ejecuta sus piezas."""

    def __init__(self, repo: Path, config: Path | None = None, overrides=(), threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        repo = Path(repo).resolve()
        for name in [m for m in sys.modules if m == "fl" or m.startswith("fl.")]:
            del sys.modules[name]
        sys.path.insert(0, str(repo))
        from fl.aggregators import build_aggregator
        from fl.clients import build_clients
        from fl.codecs import build_codec
        from fl.reto_models import build_reto_model
        from fl.utils import load_config, set_seed

        self.cfg = load_config(str(config or repo / "configs" / "reto.yaml"), list(overrides))
        self._build_clients, self._build_aggregator, self._build_codec = build_clients, build_aggregator, build_codec
        self._build_model, self._set_seed = build_reto_model, set_seed
        self.pending = {}

    def build_model(self):
        return self._build_model(self.cfg.model)

    def setup(self, client_data: dict, n_clients: int, rounds: int, fraction: float, seed: int):
        # el escenario lo fija el reto, no la configuración
        self.cfg.partition.n_clients, self.cfg.server.rounds, self.cfg.server.fraction = n_clients, rounds, fraction
        self.cfg.seed = seed
        self._set_seed(seed)
        self.model = self.build_model()
        self.aggregator = self._build_aggregator(self.cfg.server)
        self.codec = self._build_codec(self.cfg.server)
        self.clients = {}
        for cid, (x, y) in client_data.items():
            client = self._build_clients(TensorDataset(x, y), [np.arange(len(x))], self.cfg, "cpu")[0]
            client.cid = cid
            self.clients[cid] = client

    def message(self, rnd: int) -> dict:
        """El modelo global de la ronda (lo que baja a los clientes)."""
        self.global_state = _clone(self.model.state_dict())
        self.pending = {}
        return self.global_state

    def update(self, cid: int, rnd: int) -> dict:
        """Entrena el cliente y devuelve lo que sube (codec.encode)."""
        update = self.clients[cid].local_train(copy.deepcopy(self.model))
        if not hasattr(update, "state_dict") or not hasattr(update, "n_samples"):
            raise StrategyError("local_train debe devolver un ClientUpdate")
        payload = check_payload(self.codec.encode(update.state_dict, _clone(self.global_state)))
        self.pending[cid] = update
        return _clone(payload)

    def aggregate(self, payloads: dict, rnd: int):
        """Decodifica lo recibido y agrega. El agregador tiene el modelo global en ``self.global_state``."""
        updates = []
        for cid, payload in payloads.items():
            update = self.pending.pop(cid)
            update.state_dict = self.codec.decode(payload, _clone(self.global_state))
            updates.append(update)
        self.aggregator.global_state = _clone(self.global_state)
        self.model.load_state_dict(self.aggregator.aggregate(updates))

    def final_state(self) -> dict:
        return _clone(self.model.state_dict())

    @torch.no_grad()
    def predict(self, state: dict, x: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
        model = self.build_model()
        model.load_state_dict(state)
        model.eval()
        return torch.cat([model(x[i:i + batch_size]).argmax(1) for i in range(0, len(x), batch_size)])

    def close(self):
        pass


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float, rng, min_size: int = 10):
    for _ in range(100):
        shares = [[] for _ in range(n_clients)]
        for c in np.unique(labels):
            idx = rng.permutation(np.flatnonzero(labels == c))
            cuts = (np.cumsum(rng.dirichlet(np.full(n_clients, alpha))) * len(idx)).astype(int)[:-1]
            for s, chunk in zip(shares, np.split(idx, cuts)):
                s.append(chunk)
        parts = [np.sort(np.concatenate(s)) for s in shares]
        if min(len(p) for p in parts) >= min_size:
            return parts
    raise RuntimeError("no se consiguió un reparto sin clientes vacíos")


@dataclass
class Stats:
    bytes_up: int = 0
    bytes_down: int = 0
    uploads: int = 0
    time_s: float = 0.0
    curve: list = field(default_factory=list)


def run(session, sc: Scenario, x: torch.Tensor, y: torch.Tensor, *, poison=None, attack=None,
        eval_fn=None, log=None) -> tuple[dict, Stats]:
    """Simula el reto y devuelve el modelo global final (state_dict) y las estadísticas.

    ``poison`` y ``attack`` son ganchos del evaluador (clientes maliciosos); aquí no se usan.
    """
    rng = np.random.default_rng(sc.seed)
    parts = dirichlet_partition(y.numpy(), sc.n_clients, sc.alpha, rng)
    data = {}
    for cid, idx in enumerate(parts):
        yc = y[idx].clone()
        data[cid] = (x[idx].clone(), poison(cid, yc) if poison else yc)

    stats, t0 = Stats(), time.time()
    session.setup(data, sc.n_clients, sc.rounds, sc.fraction, sc.seed)
    m = max(1, round(sc.fraction * sc.n_clients))
    for rnd in range(sc.rounds):
        global_state = session.message(rnd)
        chosen = sorted(int(c) for c in rng.choice(sc.n_clients, m, replace=False))
        active = [c for c in chosen if rng.random() >= sc.dropout]
        payloads = {}
        for cid in active:
            payload = check_payload(session.update(cid, rnd), sc.budget_bytes)
            stats.bytes_down += payload_nbytes(global_state)
            stats.bytes_up += payload_nbytes(payload)
            payloads[cid] = attack(cid, payload, global_state) if attack else payload
        stats.uploads += len(payloads)
        if payloads:
            session.aggregate(payloads, rnd)
        if eval_fn:
            stats.curve.append(eval_fn(session))
        if log:
            acc = f" | acc val {stats.curve[-1]:.4f}" if eval_fn else ""
            log(f"ronda {rnd + 1:2d}/{sc.rounds} | {len(active):2d}/{m} clientes{acc} | "
                f"subida {stats.bytes_up / 1e6:6.1f} MB | {time.time() - t0:5.1f} s")
    state = session.final_state()
    stats.time_s = time.time() - t0
    return state, stats


# ---------------------------------------------------------------------------
# Línea de comandos
# ---------------------------------------------------------------------------
def pack(repo: Path, out: Path) -> None:
    files = {name: p.read_bytes() for name, p in submission_files(repo).items() if p.exists()}
    problems = check_files(files)
    if problems:
        print("El código no cumple las reglas del reto:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, raw in files.items():
            z.writestr(name, raw)
    size = out.stat().st_size
    if size > MAX_ZIP_BYTES:
        print(f"{out} ocupa {size / 1000:.0f} KB; el máximo es {MAX_ZIP_BYTES / 1000:.0f} KB")
        raise SystemExit(1)
    print(f"{out} listo ({len(files)} archivos, {size / 1000:.0f} KB). Se sube en la web del reto.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/reto.yaml")
    ap.add_argument("--set", nargs="+", action="extend", default=[], metavar="CLAVE=VALOR")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--curve", action="store_true", help="evalúa en validación en cada ronda")
    ap.add_argument("--data", default="data/reto")
    ap.add_argument("--pack", action="store_true", help="crea envio.zip para subirlo")
    args = ap.parse_args()
    repo = Path.cwd()

    if args.pack:
        pack(repo, repo / "envio.zip")
        return

    x = torch.from_numpy(np.load(Path(args.data) / "train_x.npy"))
    y = torch.from_numpy(np.load(Path(args.data) / "train_y.npy")).long()
    perm = np.random.default_rng(12345).permutation(len(x))  # 10 % para validación
    n_val = len(x) // 10
    xv, yv, xt, yt = x[perm[:n_val]], y[perm[:n_val]], x[perm[n_val:]], y[perm[n_val:]]

    def val_acc(session, state=None):
        return (session.predict(state or session.final_state(), xv) == yv).float().mean().item()

    accs = []
    for seed in args.seeds:
        session = Session(repo, args.config, args.set)
        try:
            state, stats = run(session, Scenario(seed=seed), xt, yt, eval_fn=val_acc if args.curve else None, log=print)
        except StrategyError as e:
            raise SystemExit(f"Error: {e}")
        accs.append(val_acc(session, state))
        print(f"semilla {seed}: accuracy en validación {accs[-1]:.4f} | subida {stats.bytes_up / 1e6:.1f} MB "
              f"| {stats.time_s:.0f} s\n")
    print(json.dumps({"media": round(float(np.mean(accs)), 4), "desv": round(float(np.std(accs)), 4), "accs": accs}))


if __name__ == "__main__":
    main()
