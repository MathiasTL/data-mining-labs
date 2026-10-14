"""C6/C7 — Cobertura de productos y registros fiscales.

Primer corte autorizado por G2: solo métricas de cobertura (`n_records`,
`source_present`) y **cero predictores** (`G2_products_registry_columns.csv`
no aprueba ninguna columna). Las funciones puras trabajan sobre tablas ya
leídas; los constructores leen por el seam compartido ``dm_credit.io``.

Semántica de cobertura (declarada también en el manifiesto): filas de la fuente
**disponibles a ``date_decision``** para los casos de la base. Cada fuente tiene
un ancla de disponibilidad (``ANCHORS``); una fila cuenta si su ancla no es nula
y es anterior o igual a la decisión, como en bureau. ``other_1`` no tiene fecha:
no se puede comprobar su disponibilidad y todas sus filas se descartan.

Motivo (F7, 13/10/2026): en ``products-v1`` la cobertura era física, sin filtro,
y el 100 % de las filas de ``tax_registry_a_1`` tenía ``recorddate`` posterior a
la decisión (mediana +14 días); ``tax_registry_c_1`` tenía el 21,7 %.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from dm_credit import io
from dm_credit.preparation import contracts
from dm_credit.preparation.coercion import coerce_to_logical
from dm_credit.preparation.split import partition_base

KEY = contracts.KEY
RULES_VERSION = "products-v2"
SOURCES = (
    "debitcard_1",
    "deposit_1",
    "other_1",
    "tax_registry_a_1",
    "tax_registry_b_1",
    "tax_registry_c_1",
)
# debitcard_1 y deposit_1 son 1:N (clave candidata case_id+num_group1);
# other_1 es 1:0..1 (como máximo una fila por caso).
SINGLE_ROW_SOURCES = ("other_1",)
# ancla de disponibilidad de cada fuente; None = sin fecha, no se puede comprobar
ANCHORS = {
    "debitcard_1": "openingdate_857D",
    "deposit_1": "openingdate_313D",
    "other_1": None,
    "tax_registry_a_1": "recorddate_4527225D",
    "tax_registry_b_1": "deductiondate_4917603D",
    "tax_registry_c_1": "processingdate_168D",
}


def temporal_rule(source: str) -> str:
    """Regla de disponibilidad declarada en el manifiesto de cada fuente."""
    anchor = ANCHORS[source]
    if anchor is None:
        return (
            "Sin fecha de disponibilidad: no se puede comprobar que alguna fila exista a "
            "date_decision, así que todas se descartan y la cobertura es 0 en todos los casos."
        )
    return (
        f"{anchor} <= date_decision; fila con {anchor} nula descartada. n_records cuenta "
        "las filas válidas del caso y source_present indica si hay al menos una."
    )
DEFAULT_SELECTION_PATH = (
    Path(__file__).resolve().parents[4]
    / "results"
    / "preparation"
    / "B6_products_registry_columns.csv"
)
DEFAULT_G2_PATH = DEFAULT_SELECTION_PATH.with_name("G2_products_registry_columns.csv")


@dataclass(frozen=True)
class CoverageAggregate:
    features: pl.DataFrame  # una fila por case_id de la base
    exclusions: dict[str, int]  # totales para el manifiesto


def _require(df: pl.DataFrame, columns: Sequence[str], table: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing required columns {missing}")


def _selected_columns(
    source: str, selection_path: Path, g2_path: Path
) -> tuple[list[str], dict[str, int]]:
    """Columnas aprobadas por G2 para una fuente (hoy: ninguna) y conteos B6.

    Falla si falta un manifiesto, si G2 aprueba una columna excluida en B6 o
    si aprueba una clave reservada.
    """
    selection_path, g2_path = Path(selection_path), Path(g2_path)
    if not selection_path.is_file():
        raise FileNotFoundError(f"B6 selection manifest not found: {selection_path}")
    selection = pl.read_csv(selection_path)
    required_selection = {"family", "column", "action"}
    if required_selection - set(selection.columns):
        raise ValueError(
            f"B6 selection manifest is missing columns {sorted(required_selection - set(selection.columns))}"
        )
    family_rows = selection.filter(pl.col("family") == source)
    if family_rows.is_empty():
        raise ValueError(f"B6 selection manifest contains no rows for {source!r}")
    if not g2_path.is_file():
        raise FileNotFoundError(f"G2 products decision not found: {g2_path}")
    decisions = pl.read_csv(g2_path)
    required_g2 = {"family", "column", "g2_decision"}
    if required_g2 - set(decisions.columns):
        raise ValueError(
            f"G2 products decision is missing columns {sorted(required_g2 - set(decisions.columns))}"
        )
    family_decisions = decisions.filter(pl.col("family") == source)
    if family_decisions.is_empty():
        # sin filas no hay decisión: no se interpreta como "cero aprobadas" (F6)
        raise ValueError(f"G2 products decision contains no rows for {source!r}")
    names = family_decisions["column"].to_list()
    if len(names) != len(set(names)) or any(name is None for name in names):
        raise ValueError(
            f"G2 products decision has duplicate or invalid columns for {source!r}"
        )
    excluded = set(
        family_rows.filter(pl.col("action") == "excluir")["column"].to_list()
    )
    approved = family_decisions.filter(pl.col("g2_decision") == "aprobada")[
        "column"
    ].to_list()
    if KEY in approved or "num_group1" in approved or "num_group2" in approved:
        raise ValueError(
            f"G2 products decision approves a reserved key for {source!r}"
        )
    if set(approved) - set(family_rows["column"].to_list()):
        raise ValueError(
            f"G2 products decision approves unknown columns for {source!r}"
        )
    if set(approved) & excluded:
        raise ValueError(
            f"G2 products decision approves B6-excluded columns for {source!r}: {sorted(set(approved) & excluded)}"
        )
    counts = {
        "b6_excluded": int((family_rows["action"] == "excluir").sum()),
        "b6_blocked": int((family_rows["action"] == "bloquear_hasta_G2").sum()),
        "b6_candidates": int((family_rows["action"] == "candidato_G2").sum()),
        "b6_transform_as_of": int((family_rows["action"] == "transformar_as_of").sum()),
        "g2_approved": len(approved),
    }
    return sorted(approved), counts


def _validate_source_keys(source_rows: pl.DataFrame, source: str, *, one_row_per_case: bool = True) -> None:
    """Claves de la fuente: sin nulos y sin duplicados en su granularidad.

    ``one_row_per_case=False`` valida solo la clave de la tabla (``case_id`` y
    ``num_group1``): se usa sobre el archivo completo, donde ``other_1`` puede
    traer más de una fila para casos ajenos a la base (muestra real de test).
    """
    _require(source_rows, [KEY], source)
    keys = source_rows[KEY]
    if keys.null_count() or (keys.dtype.is_float() and keys.is_nan().any()):
        raise ValueError(f"source {source!r}: {KEY} contains nulls")
    if source in SINGLE_ROW_SOURCES and one_row_per_case:
        if keys.n_unique() != source_rows.height:
            raise ValueError(
                f"source {source!r}: {KEY} is not unique (expected 1:0..1)"
            )
    elif "num_group1" in source_rows.columns:
        pairs = source_rows.select(KEY, "num_group1")
        if pairs.is_duplicated().any():
            raise ValueError(
                f"source {source!r}: (case_id, num_group1) is not unique"
            )


def filter_as_of(
    source: str, base: pl.DataFrame, source_rows: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Filas de la fuente disponibles a ``date_decision`` y conteo de las descartadas por regla."""
    if source not in SOURCES:
        raise ValueError(f"unknown C6 source {source!r}; expected one of {list(SOURCES)}")
    _require(base, [KEY, "date_decision"], "base")
    anchor = ANCHORS[source]
    if anchor is None:
        return source_rows.clear(), {f"{source}_rows_no_anchor": source_rows.height}
    _require(source_rows, [KEY, anchor], source)
    decisions = coerce_to_logical(base.select(KEY, "date_decision"), {"date_decision": "date"})
    joined = coerce_to_logical(source_rows, {anchor: "date"}).join(decisions, on=KEY, how="left")
    null_anchor = joined[anchor].is_null()
    after = ~null_anchor & (joined[anchor] > joined["date_decision"])
    valid = joined.filter(~null_anchor & ~after).drop("date_decision")
    return valid, {
        f"{source}_rows_after_decision": int(after.sum()),
        f"{source}_rows_null_anchor": int(null_anchor.sum()),
    }


def aggregate_coverage(
    source: str,
    base_cases: pl.DataFrame,
    source_rows: pl.DataFrame,
) -> CoverageAggregate:
    """Cobertura física por caso: una fila por ``case_id`` de la base.

    Las filas de la fuente de casos fuera de la base se ignoran (semi-join).
    No filtra por fecha: recibe filas ya filtradas por ``filter_as_of``.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown C6 source {source!r}; expected one of {list(SOURCES)}")
    _require(base_cases, [KEY], "base")
    if base_cases[KEY].null_count() or base_cases[KEY].n_unique() != base_cases.height:
        raise ValueError("base: case_id must be unique and non-null")
    _validate_source_keys(source_rows, source)

    cases = source_rows.join(base_cases.select(KEY), on=KEY, how="semi")
    counts = cases.group_by(KEY).agg(pl.len().alias(f"{source}__n_records"))
    features = (
        base_cases.select(KEY)
        .join(counts, on=KEY, how="left")
        .with_columns(
            pl.col(f"{source}__n_records").fill_null(0).cast(pl.Int64),
            (pl.col(f"{source}__n_records").fill_null(0) > 0).alias(
                f"{source}__source_present"
            ),
        )
        .sort(KEY)
    )
    exclusions = {
        "source_rows_outside_base": source_rows.height - cases.height,
    }
    return CoverageAggregate(features=features, exclusions=exclusions)


def _base_fingerprint(base: pl.DataFrame) -> str:
    ordered = base.select(KEY, "WEEK_NUM", "date_decision").sort(KEY)
    return hashlib.sha256(ordered.write_csv().encode()).hexdigest()


def _shard_fingerprint(file_split: str, source: str) -> str:
    return io.shards_fingerprint(file_split, [source])  # contenido, no fecha: portable entre computadoras


def _read_source(file_split: str, source: str) -> pl.DataFrame:
    """Lee la fuente completa con proyección mínima (clave, ``num_group1`` y ancla).

    Se lee entera para validar las claves de todas las filas, también las de casos
    que no están en la base del split (F6); el filtro por casos lo hace el constructor.

    Un shard vacío con columnas físicas ``Null`` (caso ``test_tax_registry_c_1``)
    se convierte a ``Int64`` antes del join: la conversión es exacta porque una
    columna ``Null`` no tiene valores. Otro tipo no entero en la clave es error.
    """
    available = set(
        io.scan_parquet_shards(file_split, source).collect_schema().names()
    )
    if KEY not in available:
        raise ValueError(f"source {source!r}: missing required column {KEY!r}")
    anchor = ANCHORS[source]
    if anchor is not None and anchor not in available:
        raise ValueError(f"source {source!r}: missing availability date {anchor!r}")
    columns = [KEY] + (["num_group1"] if "num_group1" in available else []) + ([anchor] if anchor else [])
    scan = io.scan_parquet_shards(file_split, source, columns=columns)
    key_schema = scan.collect_schema()
    if key_schema[KEY] == pl.Null:
        scan = scan.with_columns(
            pl.col(c).cast(pl.Int64) for c in columns if key_schema[c] == pl.Null
        )
    elif not key_schema[KEY].is_integer():
        raise ValueError(
            f"source {source!r}: {KEY} has non-integer dtype {key_schema[KEY]}"
        )
    frame = scan.collect()
    schema = {KEY: "int"}
    if "num_group1" in frame.columns:
        schema["num_group1"] = "int"
    return coerce_to_logical(frame, schema)


def _build_source_features(
    base: pl.DataFrame,
    split: str,
    source: str,
    *,
    cutoff_week: int,
    output_dir: Path,
    selection_path: Path,
    g2_path: Path,
) -> contracts.FamilyFeatureArtifact:
    contracts.validate_base(base, split, cutoff_week)
    approved, counts = _selected_columns(source, selection_path, g2_path)
    if approved:
        raise ValueError(
            f"{source}: G2 approves predictors {approved} but this coverage-only "
            "cut (products-v2) emits no predictors; a follow-up implementation must "
            "define their aggregation first"
        )
    file_split = "test" if split == "test" else "train"
    cases = base.select(KEY)
    raw = _read_source(file_split, source)
    # todas las filas: una clave inválida es error aunque su caso no esté en la base
    _validate_source_keys(raw, source, one_row_per_case=False)
    source_rows = raw.join(cases, on=KEY, how="semi")
    _validate_source_keys(source_rows, source)  # 1:0..1 de other_1 sobre los casos de la base
    valid, as_of = filter_as_of(source, base, source_rows)
    result = aggregate_coverage(source, cases, valid)
    result.exclusions["source_rows_outside_base"] = raw.height - source_rows.height
    definitions = contracts.describe_features(
        result.features, temporal_rule=temporal_rule(source), source_columns={}
    )
    exclusions = {**counts, **as_of, **result.exclusions}
    return contracts.write_family_features(
        result.features,
        definitions,
        family=source,
        split=split,
        output_dir=output_dir,
        exclusions=exclusions,
        rules_version=RULES_VERSION,
        inputs={
            "cutoff_week": str(cutoff_week),
            "base_sha256": _base_fingerprint(base),
            "shards": _shard_fingerprint(file_split, source),
            "selection_manifest": str(Path(selection_path)),
            "g2_decision": str(Path(g2_path)),
        },
    )


def build_debitcard_1_features(
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    g2_path: Path = DEFAULT_G2_PATH,
) -> contracts.FamilyFeatureArtifact:
    """Cobertura de ``debitcard_1`` (1:N): conteo físico y presencia por caso."""
    return _build_source_features(
        base, split, "debitcard_1", cutoff_week=cutoff_week,
        output_dir=output_dir, selection_path=selection_path, g2_path=g2_path,
    )


def build_deposit_1_features(
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    g2_path: Path = DEFAULT_G2_PATH,
) -> contracts.FamilyFeatureArtifact:
    """Cobertura de ``deposit_1`` (1:N): conteo físico y presencia por caso."""
    return _build_source_features(
        base, split, "deposit_1", cutoff_week=cutoff_week,
        output_dir=output_dir, selection_path=selection_path, g2_path=g2_path,
    )


def build_other_1_features(
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    g2_path: Path = DEFAULT_G2_PATH,
) -> contracts.FamilyFeatureArtifact:
    """Cobertura de ``other_1`` (1:0..1): presencia física por caso, sin predictores."""
    return _build_source_features(
        base, split, "other_1", cutoff_week=cutoff_week,
        output_dir=output_dir, selection_path=selection_path, g2_path=g2_path,
    )


def build_tax_registry_a_1_features(
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    g2_path: Path = DEFAULT_G2_PATH,
) -> contracts.FamilyFeatureArtifact:
    """Cobertura de ``tax_registry_a_1`` (1:N): conteo físico y presencia por caso."""
    return _build_source_features(
        base, split, "tax_registry_a_1", cutoff_week=cutoff_week,
        output_dir=output_dir, selection_path=selection_path, g2_path=g2_path,
    )


def build_tax_registry_b_1_features(
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    g2_path: Path = DEFAULT_G2_PATH,
) -> contracts.FamilyFeatureArtifact:
    """Cobertura de ``tax_registry_b_1`` (1:N): conteo físico y presencia por caso."""
    return _build_source_features(
        base, split, "tax_registry_b_1", cutoff_week=cutoff_week,
        output_dir=output_dir, selection_path=selection_path, g2_path=g2_path,
    )


def build_tax_registry_c_1_features(
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    g2_path: Path = DEFAULT_G2_PATH,
) -> contracts.FamilyFeatureArtifact:
    """Cobertura de ``tax_registry_c_1`` (1:N en train; vacía con columnas ``Null`` en test)."""
    return _build_source_features(
        base, split, "tax_registry_c_1", cutoff_week=cutoff_week,
        output_dir=output_dir, selection_path=selection_path, g2_path=g2_path,
    )


BUILDERS = {
    "debitcard_1": build_debitcard_1_features,
    "deposit_1": build_deposit_1_features,
    "other_1": build_other_1_features,
    "tax_registry_a_1": build_tax_registry_a_1_features,
    "tax_registry_b_1": build_tax_registry_b_1_features,
    "tax_registry_c_1": build_tax_registry_c_1_features,
}


def run_real_data(
    splits: Sequence[str],
    *,
    output_dir: Path,
    catalog_path: Path,
    run_report_path: Path,
    f7_report_path: Path | None = None,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    g2_path: Path = DEFAULT_G2_PATH,
    cutoff_week: int = 74,
) -> list[contracts.FamilyFeatureArtifact]:
    """Construye artefactos C6 desde los Parquet reales (ejecución pesada [R])."""
    train_base = io.load_base("train")
    test_base = io.load_base("test")
    partitions = partition_base(train_base, test_base, cutoff_week=cutoff_week)
    bases = {"train": partitions.train, "oot": partitions.oot, "test": partitions.test}
    artifacts = []
    for split in splits:
        if split not in bases:
            raise ValueError(f"split must be one of {tuple(bases)}, got {split!r}")
        for source, builder in (
            ("debitcard_1", build_debitcard_1_features),
            ("deposit_1", build_deposit_1_features),
            ("other_1", build_other_1_features),
            ("tax_registry_a_1", build_tax_registry_a_1_features),
            ("tax_registry_b_1", build_tax_registry_b_1_features),
            ("tax_registry_c_1", build_tax_registry_c_1_features),
        ):
            artifacts.append(
                builder(
                    bases[split],
                    split,
                    cutoff_week=cutoff_week,
                    output_dir=Path(output_dir) / source / split,
                    selection_path=selection_path,
                    g2_path=g2_path,
                )
            )
    contracts.write_feature_catalog(artifacts, Path(catalog_path))
    summary_rows = []
    for artifact in artifacts:
        coverage = (
            pl.scan_parquet(artifact.data_path)
            .select(
                pl.col(f"{artifact.family}__n_records").sum().alias("source_rows_for_base"),
                pl.col(f"{artifact.family}__source_present").sum().alias("cases_with_source"),
            )
            .collect()
        )
        predictor_count = sum(
            feature.role == "predictor" for feature in artifact.feature_schema
        )
        summary_rows.append(
            {
                "split": artifact.split,
                "family": artifact.family,
                "base_cases": artifact.case_count,
                "source_rows_for_base": int(coverage["source_rows_for_base"][0] or 0),
                "cases_with_source": int(coverage["cases_with_source"][0] or 0),
                "cases_without_source": artifact.case_count
                - int(coverage["cases_with_source"][0] or 0),
                "predictor_features": predictor_count,
                "coverage_features": len(artifact.feature_schema) - predictor_count,
            }
        )
    report_path = Path(run_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(summary_rows).sort(["split", "family"]).write_csv(report_path)
    if f7_report_path is not None:
        write_f7_exclusions(artifacts, Path(f7_report_path))
    return artifacts


F7_SCHEMA = {"split": pl.String, "family": pl.String, "metric": pl.String, "value": pl.Int64}


def collect_f7_exclusions(
    artifacts: Sequence[contracts.FamilyFeatureArtifact],
) -> pl.DataFrame:
    """Aporte F7: exclusiones por split/familia/métrica, mismo esquema que el de applprev/person.

    El detalle por columna (excluidas, motivos, evidencia) vive en los CSV B6 y
    G2; aquí van los conteos auditables de cada manifiesto.
    """
    rows = [
        {"split": art.split, "family": art.family, "metric": metric, "value": int(value)}
        for art in artifacts
        for metric, value in json.loads(Path(art.manifest_path).read_text())["exclusions"].items()
    ]
    return pl.DataFrame(rows, schema=F7_SCHEMA).sort(["split", "family", "metric"])


def write_f7_exclusions(
    artifacts: Sequence[contracts.FamilyFeatureArtifact], path: Path
) -> Path:
    """Escribe el aporte F7 de productos/registros; vacío es un error."""
    table = collect_f7_exclusions(artifacts)
    if table.is_empty():
        raise ValueError("cannot write empty F7 exclusions")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "oot", "test"), default=("train", "oot")
    )
    parser.add_argument("--cutoff-week", type=int, default=74)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=io.data_dir() / "artifacts" / "preparation" / "C6-v1",
    )
    parser.add_argument(
        "--catalog", type=Path, default=Path("results/preparation/products_feature_catalog.csv")
    )
    parser.add_argument(
        "--run-report", type=Path, default=Path("results/preparation/C6_products_run.csv")
    )
    parser.add_argument(
        "--f7-report", type=Path, default=Path("results/preparation/F7_products_registry_exclusions.csv")
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--g2-decision", type=Path, default=DEFAULT_G2_PATH)
    args = parser.parse_args()
    artifacts = run_real_data(
        args.splits,
        output_dir=args.output_dir,
        catalog_path=args.catalog,
        run_report_path=args.run_report,
        f7_report_path=args.f7_report,
        selection_path=args.selection,
        g2_path=args.g2_decision,
        cutoff_week=args.cutoff_week,
    )
    for artifact in artifacts:
        print(
            f"{artifact.family}/{artifact.split}: "
            f"{artifact.case_count} cases, {len(artifact.feature_schema)} features"
        )
    print(f"wrote C8 products catalog to {args.catalog}")
    print(f"wrote C6 execution summary to {args.run_report}")


if __name__ == "__main__":
    main()
