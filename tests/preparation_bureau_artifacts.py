"""E3 (bureau part) — Small synthetic bureau data and artifacts for the assembler, without real Parquet files.

``write_sample_data`` writes the bureau sources with the real file layout under a temporary data folder:
train ``credit_bureau_a_1`` in two shards (case 1 split across them, row groups of two rows) and test in three
shards, so shard counts differ between splits as in the real data. Test cases use ``case_id + 100``.
``build_small_artifacts`` publishes both bureau families for train, OOT and test. Complements
``preparation_artifacts.py`` (``applprev``/``person``), which is left untouched.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import preparation_bureau_fixtures as bfx
from dm_credit.preparation.families import bureau


def shifted(df: pl.DataFrame, offset: int) -> pl.DataFrame:
    return df.with_columns(pl.col("case_id") + offset)


def _write_shards(frame: pl.DataFrame, folder: Path, stem: str, cuts: list[int], row_group_size: int = 2) -> None:
    if not cuts:
        frame.write_parquet(folder / f"{stem}.parquet", row_group_size=row_group_size)
        return
    bounds = [0, *cuts, frame.height]
    for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        frame.slice(lo, hi - lo).write_parquet(folder / f"{stem}_{i}.parquet", row_group_size=row_group_size)


def write_sample_data(root: Path, *, write_base: bool = True) -> None:
    """Bureau sources with the real layout; ``write_base`` also writes ``{split}_base.parquet``."""
    tr, te = root / "parquet_files" / "train", root / "parquet_files" / "test"
    for folder in (tr, te):
        folder.mkdir(parents=True, exist_ok=True)
    if write_base:
        bfx.base().write_parquet(tr / "train_base.parquet")
        shifted(bfx.base().drop("target"), bfx.TEST_OFFSET).write_parquet(te / "test_base.parquet")
    _write_shards(bfx.credit_bureau_a_1(), tr, "train_credit_bureau_a_1", [2])  # case 1 spans both shards
    _write_shards(bfx.credit_bureau_a_2(), tr, "train_credit_bureau_a_2", [3])
    _write_shards(bfx.credit_bureau_b_1(), tr, "train_credit_bureau_b_1", [])
    _write_shards(bfx.credit_bureau_b_2(), tr, "train_credit_bureau_b_2", [])
    _write_shards(shifted(bfx.credit_bureau_a_1(), bfx.TEST_OFFSET), te, "test_credit_bureau_a_1", [1, 4])
    _write_shards(shifted(bfx.credit_bureau_a_2(), bfx.TEST_OFFSET), te, "test_credit_bureau_a_2", [1, 2, 4])
    _write_shards(shifted(bfx.credit_bureau_b_1(), bfx.TEST_OFFSET), te, "test_credit_bureau_b_1", [])
    _write_shards(shifted(bfx.credit_bureau_b_2(), bfx.TEST_OFFSET), te, "test_credit_bureau_b_2", [])


def build_small_artifacts(output_dir: Path, selection_path: Path):
    """Bureau artifacts for train, OOT and test (requires ``DM_DATA_DIR`` pointing to the sample data)."""
    from dm_credit import io
    from dm_credit.preparation.split import partition_base

    parts = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=bfx.CUTOFF)
    bases = {"train": parts.train, "oot": parts.oot, "test": parts.test}
    out = []
    for family, build in bureau.BUILDERS.items():
        for split, base in bases.items():
            out.append(
                build(
                    base, split, cutoff_week=bfx.CUTOFF, output_dir=output_dir / family / split,
                    selection_path=selection_path, category_shares=bfx.CATEGORY_SHARES if family == "credit_bureau_a" else None,
                )
            )
    return out
