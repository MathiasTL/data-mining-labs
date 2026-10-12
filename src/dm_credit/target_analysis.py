"""Analiza la distribución de ``target`` y su comportamiento por semana.

Uso:
    python -m dm_credit.target_analysis
    python -m dm_credit.target_analysis --out results/target_analysis
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from dm_credit import io

TARGET, WEEK = "target", "WEEK_NUM"


@dataclass
class TargetAnalysisReport:
    summary: pl.DataFrame  # metric, value
    target_distribution: pl.DataFrame  # target, cases, pct
    weekly: pl.DataFrame  # WEEK_NUM, cases, positives, target_missing, target_unexpected, default_rate_pct
    cutoffs: pl.DataFrame  # candidate cutoff and train/validation sizes and event rates


def analyze_target_timing(df: pl.DataFrame) -> TargetAnalysisReport:
    # Este reporte necesita la etiqueta y la semana para describir el objetivo
    # y simular cortes temporales. No requiere predictores.
    missing = [column for column in (TARGET, WEEK) if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if not df.schema[TARGET].is_numeric():
        raise ValueError(f"{TARGET} must be numeric")
    if not df.schema[WEEK].is_numeric():
        raise ValueError(f"{WEEK} must be numeric")

    n = df.height
    # Cuenta cada clase y expresa su peso sobre el total de casos.
    # Se conserva una fila para etiquetas nulas para que el problema sea visible.
    target_distribution = (
        df.group_by(TARGET)
        .agg(pl.len().alias("cases"))
        .with_columns(
            pl.col(TARGET).cast(pl.String).fill_null("(null)").alias("target"),
            (pl.col("cases") * 100.0 / n if n else pl.lit(0.0)).round(4).alias("pct"),
        )
        .select("target", "cases", "pct")
        .sort("target", nulls_last=True)
    )

    # Agrupa por semana: casos totales, positivos y etiquetas problemáticas.
    # La tasa de incumplimiento es positivos / casos de esa semana.
    weekly = (
        df.group_by(WEEK)
        .agg(
            pl.len().alias("cases"),
            pl.col(TARGET).eq(1).sum().alias("positives"),
            pl.col(TARGET).is_null().sum().alias("target_missing"),
            (pl.col(TARGET).is_not_null() & ~pl.col(TARGET).is_in([0, 1]))
            .sum()
            .alias("target_unexpected"),
        )
        .with_columns(
            (pl.col("positives") * 100.0 / pl.col("cases")).round(4).alias("default_rate_pct")
        )
        .sort(WEEK, nulls_last=True)
    )

    # Cada semana observada (salvo la primera) puede ser un corte:
    # semanas anteriores entrenan; desde el corte inclusive se valida.
    week_values = (
        df[WEEK]
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    cutoff_values = week_values[1:]
    # Para comparar cortes, resumimos tamaño y cantidad de positivos en ambas
    # particiones. La tabla aporta evidencia; no escoge el corte por el equipo.
    cutoff_rows = []
    for cutoff in cutoff_values:
        train = df.filter(pl.col(WEEK) < cutoff)
        validation = df.filter(pl.col(WEEK) >= cutoff)
        cutoff_rows.append(
            {
                "cutoff_week": cutoff,
                "train_cases": train.height,
                "train_positives": int(train[TARGET].eq(1).sum()),
                "train_rate_pct": round(
                    100.0 * int(train[TARGET].eq(1).sum()) / train.height, 4
                ),
                "validation_cases": validation.height,
                "validation_positives": int(validation[TARGET].eq(1).sum()),
                "validation_rate_pct": round(
                    100.0 * int(validation[TARGET].eq(1).sum()) / validation.height, 4
                ),
                "validation_weeks": validation[WEEK].n_unique(),
            }
        )
    cutoffs = pl.DataFrame(cutoff_rows) if cutoff_rows else pl.DataFrame(
        schema={
            "cutoff_week": df.schema[WEEK],
            "train_cases": pl.Int64,
            "train_positives": pl.Int64,
            "train_rate_pct": pl.Float64,
            "validation_cases": pl.Int64,
            "validation_positives": pl.Int64,
            "validation_rate_pct": pl.Float64,
            "validation_weeks": pl.UInt32,
        }
    )

    # Controles generales para detectar rápidamente faltantes, valores de
    # target distintos de 0/1 y la cobertura temporal disponible.
    summary = pl.DataFrame(
        {
            "metric": [
                "cases",
                "target_missing",
                "target_unexpected",
                "week_missing",
                "week_min",
                "week_max",
                "week_count",
            ],
            "value": [
                n,
                df[TARGET].null_count(),
                int((df[TARGET].is_not_null() & ~df[TARGET].is_in([0, 1])).sum()),
                df[WEEK].null_count(),
                df[WEEK].min(),
                df[WEEK].max(),
                len(week_values),
            ],
        }
    )
    return TargetAnalysisReport(summary, target_distribution, weekly, cutoffs)


def save_report(report: TargetAnalysisReport, out_dir: Path) -> None:
    # Cada CSV representa una tabla del análisis
    out_dir.mkdir(parents=True, exist_ok=True)
    report.summary.write_csv(out_dir / "summary.csv")
    report.target_distribution.write_csv(out_dir / "target_distribution.csv")
    report.weekly.write_csv(out_dir / "weekly.csv")
    report.cutoffs.write_csv(out_dir / "cutoff_candidates.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="Parquet file (default: train_base)")
    ap.add_argument("--out", type=Path, default=Path("results/target_analysis"))
    args = ap.parse_args()

    # Si no se indica archivo, usa train_base mediante el lector compartido.
    df = io.load_parquet(args.path) if args.path else io.load_base("train")
    report = analyze_target_timing(df)
    save_report(report, args.out)
    print(f"report written to {args.out}")
    print(f"summary: {report.summary.to_dicts()}")
    print(f"target distribution:\n{report.target_distribution}")
    print(f"weekly detail: {report.weekly.height} rows")
    print(f"cutoff candidates: {report.cutoffs.height} rows")


if __name__ == "__main__":
    main()
