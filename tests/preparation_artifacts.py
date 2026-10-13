"""E3 — Datos y artefactos pequeños de ``applprev`` y ``person`` para probar el ensamblador sin Parquet reales.

``write_sample_data`` escribe, bajo una carpeta de datos temporal, las fuentes sintéticas con la estructura
de archivos real (shards incluidos). Los casos de test usan ``case_id + 100`` para no solaparse con train.
``build_small_artifacts`` publica los agregados de train, OOT y test y devuelve los artefactos.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import preparation_fixtures as fx
from dm_credit.preparation.families import applications_people as ap

CUTOFF = 11  # con las semanas de los fixtures (10, 10, 11, 11, 12, 12): train = casos 1 y 2, OOT = 3 a 6
SMALL_APP = ap.ApplicationsConfig(
    numeric={"credamount_590A": ("count", "sum")},
    categorical_distinct=("credtype_587L",),
    category_shares={},
    bool_cols=("isbidproduct_390L",),
    date_cols=("approvaldate_319D",),
    children_category_shares={},
)
SMALL_PEOPLE = ap.PeopleConfig(
    main_numeric=("mainoccupationinc_384A",),
    main_categorical=(),
    birth_col="birth_259D",
    employment_col="empl_employedfrom_271D",
    contact_category_shares={},
    child_category_shares={},
)


def shifted(df: pl.DataFrame, offset: int) -> pl.DataFrame:
    return df.with_columns(pl.col("case_id") + offset)


def write_sample_data(root: Path) -> None:
    """Fuentes sintéticas con la estructura real: train en dos shards y test con ids desplazados."""
    for split in ("train", "test"):
        (root / "parquet_files" / split).mkdir(parents=True, exist_ok=True)
    tr, te = root / "parquet_files" / "train", root / "parquet_files" / "test"
    fx.base().write_parquet(tr / "train_base.parquet")
    a1 = fx.applprev_1()
    a1.head(4).write_parquet(tr / "train_applprev_1_0.parquet")
    a1.tail(4).write_parquet(tr / "train_applprev_1_1.parquet")
    fx.applprev_2().write_parquet(tr / "train_applprev_2.parquet")
    fx.person_1().write_parquet(tr / "train_person_1.parquet")
    fx.person_2().write_parquet(tr / "train_person_2.parquet")
    shifted(fx.base().drop("target"), 100).write_parquet(te / "test_base.parquet")
    shifted(fx.applprev_1(), 100).write_parquet(te / "test_applprev_1_0.parquet")
    shifted(fx.applprev_2(), 100).write_parquet(te / "test_applprev_2.parquet")
    shifted(fx.person_1(), 100).write_parquet(te / "test_person_1.parquet")
    shifted(fx.person_2(), 100).write_parquet(te / "test_person_2.parquet")


def build_small_artifacts(output_dir: Path):
    """Artefactos de ``applprev`` y ``person`` para train, OOT y test (requiere ``DM_DATA_DIR`` apuntando a los datos)."""
    from dm_credit import io
    from dm_credit.preparation.split import partition_base

    parts = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=CUTOFF)
    bases = {"train": parts.train, "oot": parts.oot, "test": parts.test}
    out = []
    for split, base in bases.items():
        out.append(ap.build_applprev_features(base, split, cutoff_week=CUTOFF, output_dir=output_dir / "applprev" / split, config=SMALL_APP))
        out.append(ap.build_person_features(base, split, cutoff_week=CUTOFF, output_dir=output_dir / "person" / split, config=SMALL_PEOPLE))
    return out
