"""Evidencia empírica de disponibilidad temporal para fuentes sin fecha propia.

Lee los parquet crudos de train y escribe únicamente CSV en esta carpeta.
No modifica datos, artefactos ni código productivo, y no entrena modelos.

Uso desde la raíz del repositorio:
    POLARS_MAX_THREADS=4 .venv/bin/python context/auditoria-variables-predictoras/auditar_disponibilidad.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "parquet_files" / "train"
PERIOD_BREAKS = [30, 60, 73]
PERIOD_LABELS = ["w00-30", "w31-60", "w61-73", "oot"]


def scan(table: str) -> pl.LazyFrame:
    files = sorted(glob.glob(str(RAW / f"train_{table}*.parquet")))
    return pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")


def period() -> pl.Expr:
    return pl.col("WEEK_NUM").cut(PERIOD_BREAKS, labels=PERIOD_LABELS).cast(pl.String)


def base() -> pl.LazyFrame:
    return scan("base").select(
        "case_id", "WEEK_NUM", "target", pl.col("date_decision").str.to_date()
    )


def valid_parents(b: pl.LazyFrame) -> pl.LazyFrame:
    """Padres de applprev_1 con creación anterior o igual a la decisión."""
    parents = scan("applprev_1").select(
        "case_id", "num_group1", pl.col("creationdate_885D").str.to_date()
    )
    children = scan("applprev_2").group_by("case_id", "num_group1").agg(pl.len().alias("kids"))
    return (
        parents.join(b, on="case_id")
        .filter(pl.col("creationdate_885D") <= pl.col("date_decision"))
        .with_columns(
            parent_age_days=(pl.col("date_decision") - pl.col("creationdate_885D")).dt.total_days()
        )
        .join(children, on=["case_id", "num_group1"], how="left")
        .with_columns(pl.col("kids").fill_null(0))
    )


def applprev_children(b: pl.LazyFrame) -> pl.DataFrame:
    """Hijos por padre según antigüedad del padre y periodo de decisión.

    Si applprev_2 fuera una foto tomada al extraer el dataset, una misma cohorte
    de padres tendría igual número de hijos con independencia de cuándo se
    decidió la solicitud actual. Un aumento con decisiones posteriores es
    compatible con un conteo cortado en date_decision.
    """
    par = valid_parents(b).with_columns(
        period=period(),
        parent_age=pl.col("parent_age_days").cut([30, 365, 1095]).cast(pl.String),
        parent_year=pl.col("creationdate_885D").dt.year().cast(pl.String),
    )
    by_age = par.group_by(pl.lit("parent_age").alias("cut"), pl.col("parent_age").alias("bucket"), "period")
    by_year = par.filter(pl.col("parent_year").is_in(["2017", "2018", "2019", "2020"])).group_by(
        pl.lit("parent_year").alias("cut"), pl.col("parent_year").alias("bucket"), "period"
    )
    aggs = [
        pl.len().alias("parents"),
        pl.col("kids").mean().round(4).alias("kids_mean"),
        (pl.col("kids") == 0).mean().round(4).alias("zero_kids_share"),
    ]
    return pl.concat([by_age.agg(aggs).collect(), by_year.agg(aggs).collect()]).sort("cut", "bucket", "period")


def person_roles() -> pl.DataFrame:
    """Qué representan las filas de person_1 según num_group1 y rol."""
    p1 = scan("person_1").with_columns(
        group=pl.when(pl.col("num_group1") == 0).then(pl.lit("num_group1=0")).otherwise(pl.lit("num_group1>=1"))
    )
    return (
        p1.group_by("group", "role_1084L")
        .agg(
            pl.len().alias("rows"),
            pl.col("type_25L").mode().first().alias("type_25L_mode"),
            pl.col("relationshiptoclient_415T").is_not_null().mean().round(4).alias("relationship_non_null"),
            pl.col("birth_259D").is_not_null().mean().round(4).alias("birth_non_null"),
            pl.col("mainoccupationinc_384A").is_not_null().mean().round(4).alias("income_non_null"),
        )
        .sort("group", "rows", descending=[False, True])
        .collect()
    )


def coverage_by_period(b: pl.LazyFrame) -> pl.DataFrame:
    """Presencia de cada fuente por periodo y tasa de target según presencia."""
    sources = {
        "applprev_1 (padre válido)": valid_parents(b).select("case_id").unique(),
        "person_1": scan("person_1").select("case_id").unique(),
        "person_2": scan("person_2").select("case_id").unique(),
        "static_0": scan("static_0").select("case_id").unique(),
        "static_cb_0": scan("static_cb_0").select("case_id").unique(),
        "other_1 (bruto)": scan("other_1").select("case_id").unique(),
    }
    frames = []
    for name, ids in sources.items():
        frames.append(
            b.join(ids.with_columns(pl.lit(True).alias("present")), on="case_id", how="left")
            .with_columns(pl.col("present").fill_null(False), period=period())
            .group_by("period")
            .agg(
                pl.len().alias("cases"),
                pl.col("present").mean().round(4).alias("present_rate"),
                (~pl.col("present")).sum().alias("absent_cases"),
                pl.col("target").filter(pl.col("present")).mean().round(4).alias("target_rate_present"),
                pl.col("target").filter(~pl.col("present")).mean().round(4).alias("target_rate_absent"),
            )
            .with_columns(pl.lit(name).alias("source"))
            .collect()
        )
    return pl.concat(frames).select("source", pl.exclude("source")).sort("source", "period")


def contacts_by_period(b: pl.LazyFrame) -> pl.DataFrame:
    """Media de filas num_group1>=1 de person_1 por periodo, total y por rol."""
    rows = scan("person_1").filter(pl.col("num_group1") >= 1).select("case_id", "role_1084L")
    per_case = rows.group_by("case_id").agg(
        pl.len().alias("n_contacts"),
        (pl.col("role_1084L") == "PE").sum().alias("n_role_PE"),
        (pl.col("role_1084L") == "CL").sum().alias("n_role_CL"),
        (pl.col("role_1084L") == "EM").sum().alias("n_role_EM"),
    )
    cols = ["n_contacts", "n_role_PE", "n_role_CL", "n_role_EM"]
    return (
        b.join(per_case, on="case_id", how="left")
        .with_columns(pl.col(cols).fill_null(0), period=period())
        .group_by("period")
        .agg(pl.len().alias("cases"), *[pl.col(c).mean().round(4) for c in cols])
        .sort("period")
        .collect()
    )


def main() -> None:
    b = base()
    outputs = {
        "disponibilidad-applprev2-hijos.csv": applprev_children(b),
        "disponibilidad-person1-roles.csv": person_roles(),
        "disponibilidad-person1-contactos-periodo.csv": contacts_by_period(b),
        "disponibilidad-cobertura-periodo.csv": coverage_by_period(b),
    }
    for name, frame in outputs.items():
        frame.write_csv(OUT / name)
        print(f"{name}: {frame.height} filas")


if __name__ == "__main__":
    main()
