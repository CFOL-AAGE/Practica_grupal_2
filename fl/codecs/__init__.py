"""Codecs de comunicación: qué viaja por la red. Para añadir uno, registradlo en ``build_codec``."""

from fl.codecs.base import Codec, state_dict_nbytes
from fl.codecs.int8 import int8CodeDecode
from fl.utils import Config


__all__ = ["Codec", "build_codec", "state_dict_nbytes"]


def build_codec(cfg: Config) -> Codec:
    """Crea el codec indicado en ``cfg.codec`` (sección ``server`` de la configuración)."""
    if cfg.codec == "none":
        return Codec()
    if cfg.codec == "int8":
        return int8CodeDecode()
    raise ValueError(f"Codec desconocido: {cfg.codec!r}")
