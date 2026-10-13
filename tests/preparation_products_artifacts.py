"""E3 (products and tax registries part) — Small synthetic C6/C7 data and artifacts for the assembler.

The data uses the base of ``preparation_fixtures`` (cases 1 to 6, ``CUTOFF = 11``: train = cases 1 and 2,
OOT = cases 3 to 6) so these artifacts can be assembled together with ``preparation_artifacts.py`` and
``preparation_bureau_artifacts.py``. Test files reuse the rows with ``case_id + 100``, except
``test_tax_registry_c_1.parquet``, which is empty with physical ``Null`` columns as in the real data.
Case 99 exists in ``debitcard_1`` but not in the base and must be ignored.

The fixture G2 decision approves no column, as the real one: every artifact is coverage only.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import preparation_fixtures as fx
from dm_credit.preparation.families import products_registry as pr

CUTOFF = 11
TEST_OFFSET = 100

# (case_id, num_group1) rows per source
ROWS = {
    "debitcard_1": [(1, 0), (1, 1), (1, 2), (2, 0), (4, 0), (99, 0)],
    "deposit_1": [(1, 0), (1, 1), (3, 0)],
    "other_1": [(2, 0), (5, 0)],  # 1:0..1
    "tax_registry_a_1": [(1, 0), (1, 1), (1, 2), (2, 0), (6, 0)],
    "tax_registry_b_1": [(2, 0), (3, 0)],
    "tax_registry_c_1": [(1, 0), (3, 0), (4, 0)],
}
# sources whose test file is empty with physical Null columns
EMPTY_NULL_TEST_SOURCES = ("tax_registry_c_1",)
NULL_TEST_COLUMNS = ("case_id", "employername_160M", "num_group1", "pmtamount_36A", "processingdate_168D")

# hand-computed n_records, one value per base case (1 to 6); source_present is n_records > 0
EXPECTED_N_RECORDS = {
    "debitcard_1": [3, 1, 0, 1, 0, 0],
    "deposit_1": [2, 0, 1, 0, 0, 0],
    "other_1": [0, 1, 0, 0, 1, 0],
    "tax_registry_a_1": [3, 1, 0, 0, 0, 1],
    "tax_registry_b_1": [0, 1, 1, 0, 0, 0],
    "tax_registry_c_1": [1, 0, 1, 1, 0, 0],
}


def shifted(df: pl.DataFrame, offset: int) -> pl.DataFrame:
    return df.with_columns(pl.col("case_id") + offset)


def source(name: str) -> pl.DataFrame:
    return pl.DataFrame(ROWS[name], schema={"case_id": pl.Int64, "num_group1": pl.Int64}, orient="row")


def write_selection(path: Path) -> Path:
    """Small B6-like selection: one blocked column and the excluded key per source."""
    rows = []
    for name in pr.SOURCES:
        rows.append((name, "amount_1A", "Float64", "bloquear_hasta_G2", "r", ""))
        rows.append((name, "case_id", "Int64", "excluir", "clave", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=["family", "column", "dtype", "action", "reason", "note"], orient="row").write_csv(path)
    return path


def write_decisions(path: Path) -> Path:
    """G2-like decision for ``write_selection``: every column stays ``bloqueada``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "family": list(pr.SOURCES),
            "column": ["amount_1A"] * len(pr.SOURCES),
            "g2_decision": ["bloqueada"] * len(pr.SOURCES),
        }
    ).write_csv(path)
    return path


def write_sample_data(root: Path, *, write_base: bool = True) -> None:
    """Products and tax registry sources with the real layout; ``write_base`` also writes ``{split}_base.parquet``."""
    tr, te = root / "parquet_files" / "train", root / "parquet_files" / "test"
    for folder in (tr, te):
        folder.mkdir(parents=True, exist_ok=True)
    if write_base:
        fx.base().write_parquet(tr / "train_base.parquet")
        shifted(fx.base().drop("target"), TEST_OFFSET).write_parquet(te / "test_base.parquet")
    for name in pr.SOURCES:
        source(name).write_parquet(tr / f"train_{name}.parquet")
        if name in EMPTY_NULL_TEST_SOURCES:
            pl.DataFrame(
                {column: [] for column in NULL_TEST_COLUMNS}, schema={column: pl.Null for column in NULL_TEST_COLUMNS}
            ).write_parquet(te / f"test_{name}.parquet")
        else:
            shifted(source(name), TEST_OFFSET).write_parquet(te / f"test_{name}.parquet")


def build_small_artifacts(output_dir: Path, selection_path: Path, g2_path: Path):
    """Products and tax registry artifacts for train, OOT and test (requires ``DM_DATA_DIR`` pointing to the sample data)."""
    from dm_credit import io
    from dm_credit.preparation.split import partition_base

    parts = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=CUTOFF)
    bases = {"train": parts.train, "oot": parts.oot, "test": parts.test}
    out = []
    for family, build in pr.BUILDERS.items():
        for split, base in bases.items():
            out.append(
                build(
                    base, split, cutoff_week=CUTOFF, output_dir=output_dir / family / split,
                    selection_path=selection_path, g2_path=g2_path,
                )
            )
    return out
