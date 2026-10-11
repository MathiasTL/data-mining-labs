"""Integridad estructural de la tabla base (Etapa A, paso 2).

Verifica tamaño y estructura, integridad de la clave y validación estructural
mínima. No analiza nulos, distribuciones ni cardinalidad: eso corresponde al
perfilador (``dm_credit.profiling``) y a otros pasos.

Uso:
    python -m dm_credit.integrity                       # train_base
    python -m dm_credit.integrity path/to/file.parquet --out results/integrity_x
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from dm_credit import io

KEY, TARGET, WEEK = "case_id", "target", "WEEK_NUM"
DATE_COL, MONTH = "date_decision", "MONTH"
DATE_FORMAT = "%Y-%m-%d"
SAMPLE_KEYS = 20  # claves repetidas incluidas en la muestra

BLOCKING, OBSERVATION, OK = "blocking", "observation", "ok"


@dataclass
class IntegrityReport:
    n_rows: int
    n_cols: int
    memory_mb: float
    structure: pl.DataFrame  # column, dtype, apparent_kind, role, type_issue
    key_integrity: pl.DataFrame  # metric, value
    key_frequency: pl.DataFrame  # times_repeated, n_keys (vacío si no hay repetidas)
    repeated_sample: pl.DataFrame  # filas de las primeras claves repetidas (vacío si no hay)
    findings: pl.DataFrame  # check, status, count, detail
    blocks_baseline: bool
    date_range: tuple[str, str] | None = None  # rango de date_decision si es 100 % parseable


def _parse_dates(s: pl.Series) -> pl.Series | None:
    """Devuelve la serie como ``Date`` si es temporal o un String 100 % parseable; si no, ``None``."""
    if s.dtype == pl.Date:
        return s
    if s.dtype == pl.Datetime:
        return s.dt.date()
    if s.dtype != pl.String:
        return None
    values = s.drop_nulls()
    if values.len() == 0:
        return None
    parsed = s.str.strptime(pl.Date, DATE_FORMAT, strict=False)
    return parsed if parsed.null_count() == s.null_count() else None


def _type_issue(s: pl.Series) -> str:
    """Describe una incompatibilidad entre el dtype y el contenido aparente; vacío si no hay."""
    if s.dtype != pl.String:
        return ""
    if _parse_dates(s) is not None:
        return "date stored as String"
    values = s.drop_nulls()
    if values.len() and values.cast(pl.Float64, strict=False).null_count() == 0:
        return "number stored as String"
    return ""


def _apparent_kind(s: pl.Series) -> str:
    if s.dtype.is_numeric():
        return "numeric"
    if s.dtype.is_temporal() or _parse_dates(s) is not None:
        return "temporal"
    return "categorical"


def _name_issues(columns: list[str]) -> list[str]:
    issues = [f"{c!r} has leading/trailing whitespace" for c in columns if c != c.strip()]
    groups: dict[str, list[str]] = {}
    for c in columns:
        groups.setdefault(c.strip().lower(), []).append(c)
    issues += [f"names collide after normalizing: {g}" for g in groups.values() if len(g) > 1]
    return issues


def check_integrity(df: pl.DataFrame) -> IntegrityReport:
    n = df.height
    role = {KEY: "key", TARGET: "target", WEEK: "week"}

    structure = pl.DataFrame(
        [
            {
                "column": c,
                "dtype": str(df.schema[c]),
                "apparent_kind": _apparent_kind(df[c]),
                "role": role.get(c, ""),
                "type_issue": _type_issue(df[c]),
            }
            for c in df.columns
        ],
        schema={"column": pl.String, "dtype": pl.String, "apparent_kind": pl.String, "role": pl.String, "type_issue": pl.String},
    )

    findings: list[dict] = []

    def add(check: str, bad: int, severity: str, detail: str = "") -> None:
        findings.append({"check": check, "status": severity if bad else OK, "count": bad, "detail": detail})

    # 2.1 columnas requeridas
    for c in (KEY, TARGET, WEEK):
        add(f"missing_column:{c}", int(c not in df.columns), BLOCKING, f"required column {c} is absent")

    # 2.2 integridad de la clave
    key_rows: list[dict] = []
    key_frequency = pl.DataFrame(schema={"times_repeated": pl.Int64, "n_keys": pl.Int64})
    repeated_sample = df.clear()
    n_dup_rows = int(df.is_duplicated().sum())
    key_rows.append({"metric": "duplicate_rows", "value": n_dup_rows})
    add("duplicate_rows", n_dup_rows, OBSERVATION, "rows that are exact copies of another row")

    if KEY in df.columns:
        null_keys = df[KEY].null_count()
        counts = df.get_column(KEY).drop_nulls().value_counts().rename({"count": "n"}) if n else None
        repeated = counts.filter(pl.col("n") > 1) if counts is not None else pl.DataFrame({"n": []})
        n_rep_keys = repeated.height
        n_rep_rows = int(repeated["n"].sum()) if n_rep_keys else 0
        key_rows += [
            {"metric": "null_keys", "value": null_keys},
            {"metric": "repeated_keys", "value": n_rep_keys},
            {"metric": "rows_in_repeated_keys", "value": n_rep_rows},
        ]
        add("null_keys", null_keys, BLOCKING, f"nulls in {KEY}")
        add("repeated_keys", n_rep_keys, BLOCKING, f"{n_rep_rows} rows share a {KEY} with another row")
        if n_rep_keys:
            key_frequency = (
                repeated.group_by("n").len().rename({"n": "times_repeated", "len": "n_keys"}).sort("times_repeated")
            )
            sample_keys = repeated.sort(KEY).head(SAMPLE_KEYS)[KEY]
            repeated_sample = df.filter(pl.col(KEY).is_in(sample_keys.implode())).sort(KEY)

    # 2.3 validación estructural
    empty = [c for c in df.columns if n and df[c].null_count() == n]
    add("empty_columns", len(empty), OBSERVATION, ", ".join(empty))
    name_issues = _name_issues(df.columns)
    add("column_name_issues", len(name_issues), OBSERVATION, "; ".join(name_issues))
    typed = structure.filter(pl.col("type_issue") != "")
    add("dtype_content_mismatch", typed.height, OBSERVATION, "; ".join(f"{c}: {i}" for c, i in zip(typed["column"], typed["type_issue"])))

    date_range = None
    if DATE_COL in df.columns:
        dates = _parse_dates(df[DATE_COL])
        if dates is not None:
            date_range = (str(dates.min()), str(dates.max()))
            if MONTH in df.columns and df[MONTH].dtype.is_numeric():
                expected = dates.dt.year().cast(pl.Int64) * 100 + dates.dt.month().cast(pl.Int64)
                add("month_mismatch", int((df[MONTH].cast(pl.Int64) != expected).sum()), OBSERVATION, f"{MONTH} != yyyymm({DATE_COL})")
            if WEEK in df.columns and df[WEEK].dtype.is_numeric():
                expected = (dates - dates.min()).dt.total_days() // 7
                add(
                    "week_num_mismatch",
                    int((df[WEEK].cast(pl.Int64) != expected.cast(pl.Int64)).sum()),
                    OBSERVATION,
                    f"{WEEK} != floor(({DATE_COL} - min({DATE_COL})) / 7 days)",
                )

    if TARGET in df.columns:
        values = df[TARGET].drop_nulls()
        outside = int((~values.cast(pl.Float64).is_in([0.0, 1.0])).sum()) if values.dtype.is_numeric() else values.len()
        add("target_outside_0_1", outside, BLOCKING, f"{TARGET} values not in {{0, 1}}")

    findings_df = pl.DataFrame(
        findings, schema={"check": pl.String, "status": pl.String, "count": pl.Int64, "detail": pl.String}
    )
    return IntegrityReport(
        n_rows=n,
        n_cols=df.width,
        memory_mb=df.estimated_size("mb"),
        structure=structure,
        key_integrity=pl.DataFrame(key_rows, schema={"metric": pl.String, "value": pl.Int64}),
        key_frequency=key_frequency,
        repeated_sample=repeated_sample,
        findings=findings_df,
        blocks_baseline=bool((findings_df["status"] == BLOCKING).any()),
        date_range=date_range,
    )


def save_report(report: IntegrityReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pl.DataFrame(
        {
            "metric": ["rows", "columns", "memory_mb", "date_min", "date_max", "blocks_baseline"],
            "value": [
                str(report.n_rows),
                str(report.n_cols),
                f"{report.memory_mb:.2f}",
                report.date_range[0] if report.date_range else "",
                report.date_range[1] if report.date_range else "",
                str(report.blocks_baseline).lower(),
            ],
        }
    )
    summary.write_csv(out_dir / "summary.csv")
    report.structure.write_csv(out_dir / "structure.csv")
    report.key_integrity.write_csv(out_dir / "key_integrity.csv")
    report.findings.write_csv(out_dir / "findings.csv")
    if report.repeated_sample.height:
        report.repeated_sample.write_csv(out_dir / "repeated_keys_sample.csv")
        report.key_frequency.write_csv(out_dir / "repeated_keys_frequency.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="Parquet file (default: train_base)")
    ap.add_argument("--out", type=Path, help="Directory where CSV reports are written")
    args = ap.parse_args()

    df = io.load_parquet(args.path) if args.path else io.load_base("train")
    r = check_integrity(df)
    print(f"rows={r.n_rows:,} cols={r.n_cols} memory={r.memory_mb:.2f}MB")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80):
        print(r.structure)
        print(r.key_integrity)
        print(r.findings)
        if r.repeated_sample.height:
            print(r.key_frequency)
            print(r.repeated_sample)
    if r.date_range:
        print(f"{DATE_COL} range: {r.date_range[0]} .. {r.date_range[1]}")
    print(f"blocks_baseline={r.blocks_baseline}")
    if args.out:
        save_report(r, args.out)
        print(f"reports written to {args.out}")


if __name__ == "__main__":
    main()
