"""E3 (depth-zero part) — Small synthetic ``static_0``/``static_cb_0`` data and artifacts for the assembler.

The data uses the base of ``preparation_fixtures`` (cases 1 to 6, ``CUTOFF = 11``: train = cases 1 and 2,
OOT = cases 3 to 6) so these artifacts can be assembled together with ``preparation_artifacts.py`` and
``preparation_bureau_artifacts.py``. Test cases use ``case_id + 100``. ``static_0`` covers every case and is
written in two train shards; ``static_cb_0`` only has cases 1, 3 and 5, so the others keep zero coverage.
Case 99 exists in the sources but not in the base and must be ignored.

The fixture G2 decision approves one column per source so the assembler also receives predictors; the real
G2 file approves none (coverage only).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import preparation_fixtures as fx
from dm_credit.preparation.families import depth0

CUTOFF = 11
TEST_OFFSET = 100

# hand-computed expectations, one value per base case (1 to 6)
EXPECTED = {
    "static_0__n_records": [1, 1, 1, 1, 1, 1],
    "static_0__source_present": [True, True, True, True, True, True],
    "static_0__static_value__first__all": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    "static_cb_0__n_records": [1, 0, 1, 0, 1, 0],
    "static_cb_0__source_present": [True, False, True, False, True, False],
    "static_cb_0__static_cb_value__first__all": [5, None, 7, None, 9, None],
}


def shifted(df: pl.DataFrame, offset: int) -> pl.DataFrame:
    return df.with_columns(pl.col("case_id") + offset)


def static_0() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "case_id": [1, 2, 3, 4, 5, 6, 99],
            "static_value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 990.0],
            "static_flag": [True, False, None, True, False, True, True],
        }
    )


def static_cb_0() -> pl.DataFrame:
    return pl.DataFrame({"case_id": [1, 3, 5, 99], "static_cb_value": [5, 7, 9, 99]})


def write_selection(path: Path) -> Path:
    """Small B3-like selection: one approved candidate per source, one blocked candidate and one manual review."""
    rows = [
        ("static_0", "static_value", "provisional_candidate"),
        ("static_0", "static_flag", "provisional_candidate"),
        ("static_0", "not_used", "manual_review"),
        ("static_cb_0", "static_cb_value", "provisional_candidate"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [(family, column, action, "fixture", "", "all", "fixture") for family, column, action in rows],
        schema=["family", "column", "proposed_action", "description", "reason", "temporal_rule", "evidence"],
        orient="row",
    ).write_csv(path)
    return path


def write_decisions(path: Path) -> Path:
    """G2-like decision for ``write_selection``: ``static_flag`` stays ``bloqueada``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "family": ["static_0", "static_0", "static_cb_0"],
            "column": ["static_value", "static_flag", "static_cb_value"],
            "g2_decision": ["aprobada", "bloqueada", "aprobada"],
        }
    ).write_csv(path)
    return path


def write_sample_data(root: Path, *, write_base: bool = True) -> None:
    """Depth-zero sources with the real layout; ``write_base`` also writes ``{split}_base.parquet``."""
    tr, te = root / "parquet_files" / "train", root / "parquet_files" / "test"
    for folder in (tr, te):
        folder.mkdir(parents=True, exist_ok=True)
    if write_base:
        fx.base().write_parquet(tr / "train_base.parquet")
        shifted(fx.base().drop("target"), TEST_OFFSET).write_parquet(te / "test_base.parquet")
    s0 = static_0()
    s0.head(3).write_parquet(tr / "train_static_0_0.parquet")
    s0.tail(4).write_parquet(tr / "train_static_0_1.parquet")
    static_cb_0().write_parquet(tr / "train_static_cb_0.parquet")
    shifted(s0, TEST_OFFSET).write_parquet(te / "test_static_0.parquet")
    shifted(static_cb_0(), TEST_OFFSET).write_parquet(te / "test_static_cb_0.parquet")


def build_small_artifacts(output_dir: Path, selection_path: Path, g2_path: Path):
    """Depth-zero artifacts for train, OOT and test (requires ``DM_DATA_DIR`` pointing to the sample data)."""
    from dm_credit import io
    from dm_credit.preparation.split import partition_base

    parts = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=CUTOFF)
    bases = {"train": parts.train, "oot": parts.oot, "test": parts.test}
    out = []
    for family, build in depth0.BUILDERS.items():
        for split, base in bases.items():
            out.append(
                build(
                    base, split, cutoff_week=CUTOFF, output_dir=output_dir / family / split,
                    selection_path=selection_path, g2_path=g2_path,
                )
            )
    return out
