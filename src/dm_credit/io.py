"""Punto único de lectura de los datos del proyecto.

Todo notebook o script debe cargar los datos a través de este módulo para que
las rutas y la forma de lectura sean idénticas en todo el equipo.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

DATA_DIR_ENV = "DM_DATA_DIR"
SPLITS = ("train", "test")


def data_dir() -> Path:
    """Raíz de datos: ``$DM_DATA_DIR`` si está definida; si no, ``<repo>/data``."""
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data"


def base_path(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return data_dir() / "parquet_files" / split / f"{split}_base.parquet"


def load_parquet(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    """Lee un Parquet de forma inmediata. Si falta el archivo, lanza un error con instrucciones."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Data lives outside git; place it under <repo>/data "
            f"or point {DATA_DIR_ENV} to your local copy."
        )
    return pl.read_parquet(path, columns=columns)


def load_base(split: str = "train") -> pl.DataFrame:
    return load_parquet(base_path(split))


def load_feature_definitions() -> pl.DataFrame:
    return _read_csv(data_dir() / "feature_definitions.csv")


def _read_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Check {DATA_DIR_ENV}.")
    return pl.read_csv(path)
