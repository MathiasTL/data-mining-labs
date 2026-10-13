"""Evidencia de A4: cifras que sustentan las reglas temporales de solicitudes previas y personas.

Todo es de solo lectura y acepta DataFrames o LazyFrames, de modo que sobre los datos reales
se consulta con ``scan_parquet`` sin materializar joins entre familias.

Uso:
    POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.applications_people_evidence
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from dm_credit import io

Frame = pl.DataFrame | pl.LazyFrame
DATE = "%Y-%m-%d"


def _scalar(lf: pl.LazyFrame, expr: pl.Expr) -> int:
    return int(lf.select(expr).collect().item() or 0)


def compute_evidence(base: Frame, applprev_1: Frame, applprev_2: Frame, person_1: Frame, person_2: Frame) -> dict[str, int]:
    decision = base.lazy().select("case_id", pl.col("date_decision").str.to_date(DATE).alias("decision"))
    a1 = applprev_1.lazy().join(decision, on="case_id").with_columns(pl.col("creationdate_885D").str.to_date(DATE).alias("created"))
    valid_a1 = a1.filter(pl.col("created") <= pl.col("decision"))
    a2 = applprev_2.lazy()
    p1 = person_1.lazy().join(decision, on="case_id")
    p2 = person_2.lazy()

    parents = a1.select("case_id", "num_group1").unique()
    valid_parents = valid_a1.select("case_id", "num_group1").unique()
    p1_keys = p1.select("case_id", "num_group1").unique()
    keys = ["case_id", "num_group1"]

    return {
        "applprev_1_rows": _scalar(a1, pl.len()),
        "applprev_1_cases": _scalar(a1, pl.col("case_id").n_unique()),
        "applprev_1_rows_after_decision": _scalar(a1, (pl.col("created") > pl.col("decision")).sum()),
        "applprev_1_rows_null_creation": _scalar(a1, pl.col("created").is_null().sum()),
        "applprev_1_cases_after_filter": _scalar(valid_a1, pl.col("case_id").n_unique()),
        "applprev_2_rows": _scalar(a2, pl.len()),
        "applprev_2_orphans": _scalar(a2.join(parents, on=keys, how="anti"), pl.len()),
        "applprev_2_rows_valid_parent": _scalar(a2.join(valid_parents, on=keys), pl.len()),
        "person_1_rows": _scalar(p1, pl.len()),
        "person_1_cases": _scalar(p1, pl.col("case_id").n_unique()),
        "person_1_cases_with_main": _scalar(p1.filter(pl.col("num_group1") == 0), pl.col("case_id").n_unique()),
        "person_1_employment_after_decision": _scalar(
            p1, (pl.col("empl_employedfrom_271D").str.to_date(DATE) > pl.col("decision")).sum()
        ),
        "person_1_birth_after_decision": _scalar(p1, (pl.col("birth_259D").str.to_date(DATE) > pl.col("decision")).sum()),
        "person_2_rows": _scalar(p2, pl.len()),
        "person_2_orphans": _scalar(p2.join(p1_keys, on=keys, how="anti"), pl.len()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/preparation/A4_evidence.csv"))
    args = ap.parse_args()
    train = io.data_dir() / "parquet_files" / "train"
    metrics = compute_evidence(
        pl.scan_parquet(train / "train_base.parquet"),
        pl.scan_parquet(train / "train_applprev_1_*.parquet"),
        pl.scan_parquet(train / "train_applprev_2.parquet"),
        pl.scan_parquet(train / "train_person_1.parquet"),
        pl.scan_parquet(train / "train_person_2.parquet"),
    )
    out = pl.DataFrame({"metric": list(metrics), "value": list(metrics.values())})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out)
    with pl.Config(tbl_rows=-1):
        print(out)
    print(f"escrito en {args.out}")


if __name__ == "__main__":
    main()
