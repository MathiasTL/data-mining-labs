"""Punto único de lectura de los datos del proyecto.

Todo notebook o script debe cargar los datos a través de este módulo para que
las rutas y la forma de lectura sean idénticas en todo el equipo.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

import polars as pl
import pyarrow.parquet as pq

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


def shard_paths(split: str, source: str) -> list[Path]:
    """Shards de una fuente física, ordenados por índice numérico (``_2`` antes que ``_10``).

    Acepta ``{split}_{source}.parquet`` y ``{split}_{source}_{n}.parquet``. ``split`` es el de los
    archivos (``train`` o ``test``); la partición OOT se separa después con la tabla base.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    folder = data_dir() / "parquet_files" / split
    pattern = re.compile(rf"^{re.escape(split)}_{re.escape(source)}(?:_(\d+))?\.parquet$")
    found = []
    if folder.is_dir():
        for path in folder.iterdir():
            m = pattern.match(path.name)
            if m:
                found.append((int(m.group(1)) if m.group(1) is not None else -1, path))
    if not found:
        raise FileNotFoundError(
            f"no shards for source {source!r} in {folder}. Data lives outside git; "
            f"check {DATA_DIR_ENV} or place the files under <repo>/data."
        )
    return [p for _, p in sorted(found, key=lambda item: item[0])]


def scan_parquet_shards(split: str, source: str, columns: list[str] | None = None) -> pl.LazyFrame:
    """Escaneo perezoso de todos los shards de una fuente, con proyección de columnas."""
    paths = shard_paths(split, source)
    lf = pl.concat([pl.scan_parquet(p) for p in paths], how="diagonal_relaxed")
    if columns is not None:
        missing = [c for c in columns if c not in lf.collect_schema().names()]
        if missing:
            raise ValueError(f"source {source!r} has no columns {missing}")
        lf = lf.select(columns)
    return lf


def iter_row_groups(path: str | Path, columns: list[str] | None = None) -> Iterator[pl.DataFrame]:
    """Un grupo de filas por iteración, con proyección de columnas (para fuentes grandes)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Check {DATA_DIR_ENV}.")
    pf = pq.ParquetFile(path)
    for i in range(pf.num_row_groups):
        yield pl.from_arrow(pf.read_row_group(i, columns=columns))
