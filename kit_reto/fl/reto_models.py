"""Modelos del reto: reciben un embedding de 384 números y devuelven 100 logits.

Aquí se añaden los modelos, que se eligen con ``model.name`` en ``configs/reto.yaml`` (igual que
``build_model`` en ``fl/models.py``). El evaluador reconstruye el modelo final con esta función.
"""

from __future__ import annotations

from torch import nn

from fl.utils import Config

IN_FEATURES, N_CLASSES = 384, 100


def build_reto_model(cfg: Config) -> nn.Module:
    """Crea el modelo indicado en la sección ``model`` de la configuración."""
    if cfg.name == "linear":
        return nn.Linear(IN_FEATURES, N_CLASSES)
    if cfg.name == "mlp":
        # ~97 000 parámetros: en float32 no cabe en el presupuesto de subida (hace falta comprimir)
        return nn.Sequential(nn.Linear(IN_FEATURES, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, N_CLASSES))
    raise ValueError(f"Modelo desconocido: {cfg.name!r}")
