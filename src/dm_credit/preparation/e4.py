"""E4: auditoría serial, de solo lectura sobre los 36 artefactos existentes.

CLI: POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e4
No construye familias, no ensambla matrices ni usa checkpoints. Solo publica E4_*.
Los helpers públicos no escriben; run_audit persiste informes incluso con fallos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import resource
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import polars as pl

from dm_credit import io
from dm_credit.preparation import contracts
from dm_credit.preparation.columns import validate_predictor_columns
from dm_credit.preparation.families import applications_people as ap
from dm_credit.preparation.families import applications_people_g2 as g2
from dm_credit.preparation.families import bureau, depth0
from dm_credit.preparation.families import products_registry as products
from dm_credit.preparation.split import partition_base

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_ROWS = {"train": 1332001, "oot": 194658, "test": 10}
A2 = "credit_bureau_a_2"
LIMITATIONS = [
    "Los manifiestos no capturan commit ni hash de configuración del build; los hashes E4 son actuales, no históricos.",
    "No hay tiempos ni recursos históricos de build verificables: se informan null, sin inferirlos de mtimes.",
    "ru_maxrss es el máximo acumulado del proceso auditor (KiB en Linux), no memoria por familia ni del build.",
    "La conciliación usa family_cardinality histórico; no recalcula filtros temporales ni demuestra equivalencia de valores con un rebuild.",
    "Las exclusiones de valores y columnas no se suman como filas; A2 no se lee ni se incluyen sus shards en fingerprints.",
    "Cobertura con current_inputs_valid=false describe un artefacto obsoleto: su balance no autoriza su reutilización.",
]


@dataclass(frozen=True)
class FamilySpec:
    family: str
    directory: str
    rules_version: str
    tables: tuple[str, ...]
    catalog: str
    decisions: str


def family_specs() -> tuple[FamilySpec, ...]:
    """Inventario cerrado: 12 familias, versiones tomadas de los builders actuales."""
    groups = (
        (depth0.SOURCES, "C1-v1", depth0.RULES_VERSION, "depth0", "depth0"),
        (
            ap.TABLES,
            "C2-C3-v1",
            ap.RULES_VERSION,
            "applications_people",
            "applications_people",
        ),
        (bureau.FAMILIES, "C4-C5-v2", bureau.RULES_VERSION, "bureau", "bureau"),
        (
            products.SOURCES,
            "C6-v1",
            products.RULES_VERSION,
            "products",
            "products_registry",
        ),
    )
    return tuple(
        FamilySpec(
            f,
            directory,
            version,
            tuple(families[f]) if isinstance(families, dict) else (f,),
            f"{catalog}_feature_catalog.csv",
            f"G2_{decision}_columns.csv",
        )
        for families, directory, version, catalog, decision in groups
        for f in families
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_fingerprint(base: pl.DataFrame) -> str:
    """Misma representación canónica que los cuatro builders, sin target."""
    return hashlib.sha256(
        base.select("case_id", "WEEK_NUM", "date_decision")
        .sort("case_id")
        .write_csv()
        .encode()
    ).hexdigest()


def shard_fingerprint(
    split: str, tables: tuple[str, ...], *, ordered: bool = False
) -> str:
    """Stat actual nombre:size:mtime_ns; bureau conserva orden numérico de io."""
    if A2 in tables:
        raise ValueError("E4 no debe consultar shards de credit_bureau_a_2")
    parts = []
    for table in tables:
        for path in io.shard_paths("test" if split == "test" else "train", table):
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return ";".join(parts if ordered else sorted(parts))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_equal(actual, expected, label: str) -> None:
    require(actual == expected, f"{label}: actual={actual!r}; esperado={expected!r}")


class Checks:
    """Frontera de errores: un fallo no impide auditar las otras familias/checks."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def run(self, scope: str, name: str, operation: Callable):
        try:
            value = operation()
        except Exception as exc:
            self.rows.append(
                {
                    "scope": scope,
                    "check": name,
                    "status": "invalid",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return None
        self.rows.append({"scope": scope, "check": name, "status": "ok", "error": None})
        return value


def load_decisions(path: Path, tables: tuple[str, ...], *, zero: bool) -> pl.DataFrame:
    frame = pl.read_csv(path, infer_schema=False)
    require(
        {"family", "column", "g2_decision"} <= set(frame.columns), "G2: faltan campos"
    )
    require(
        not frame.select("family", "column").is_duplicated().any(),
        "G2: claves duplicadas",
    )
    require(
        not frame.select(
            pl.any_horizontal(pl.col("family", "column", "g2_decision").is_null()).any()
        ).item(),
        "G2: claves/decisiones nulas",
    )
    require(
        set(frame["g2_decision"]) <= set(bureau.G2_DECISIONS),
        "G2: decisiones desconocidas",
    )
    require(set(tables) <= set(frame["family"]), "G2: faltan tablas")
    if zero:
        require(
            frame.filter(pl.col("g2_decision") == "aprobada").is_empty(),
            "G2: se requieren cero aprobaciones",
        )
    return frame


def expected_ap_features(family: str, config) -> dict[str, tuple[str, ...]]:
    """Contrato de nombres y linaje con config/vocab vigente, sin agregar datos."""
    out: dict[str, tuple[str, ...]] = {}

    def add(table, column, aggregation):
        norm = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
        name = f"{table}__{norm(column)}__{aggregation}__all"
        require(name not in out, f"config/vocab produce nombre duplicado: {name}")
        out[name] = (column,)

    def shares(table, vocabulary):
        for column, categories in vocabulary.items():
            for category in categories:
                token = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
                add(table, column, f"share_{token}")

    for table in ap.TABLES[family]:
        out.update({f"{table}__n_records": (), f"{table}__source_present": ()})
    if family == "applprev":
        for agg in ("days_since_last", "days_since_first"):
            add("applprev_1", ap.CREATION, agg)
        for column, aggregations in config.numeric.items():
            for agg in aggregations:
                add("applprev_1", column, agg)
        for columns, agg in (
            (config.categorical_distinct, "n_distinct"),
            (config.bool_cols, "share_true"),
            (config.date_cols, "days_since_last"),
        ):
            for column in columns:
                add("applprev_1", column, agg)
        shares("applprev_1", config.category_shares)
        shares("applprev_2", config.children_category_shares)
        for agg in ("mean", "max"):
            out[f"applprev_2__children_per_parent_{agg}__all"] = ()
    else:
        for column in (*config.main_numeric, *config.main_categorical):
            add("person_1", column, "main")
        for column, agg in (
            (config.birth_col, "age_years"),
            (config.employment_col, "days_since_last"),
        ):
            if column:
                add("person_1", column, agg)
        out["person_1__n_contacts__all"] = ()
        shares("person_1", config.contact_category_shares)
        shares("person_2", config.child_category_shares)
    return out


def validate_definitions(
    artifact, schema, spec: FamilySpec, tables, decisions, expected
) -> None:
    definitions = artifact.feature_schema
    names = [d.name for d in definitions]
    validate_predictor_columns(names)
    check_equal(schema.names(), ["case_id", *names], "orden/esquema físico")
    require(schema["case_id"].is_integer(), "case_id debe ser entero")
    if expected is None:
        expected = {
            f"{table}__{metric}": ()
            for table in tables
            for metric in contracts.COVERAGE
        }
    check_equal(set(names), set(expected), "inventario config/vocab")
    approved = set(
        decisions.filter(pl.col("g2_decision") == "aprobada")
        .select("family", "column")
        .iter_rows()
    )
    for d in definitions:
        check_equal(d.source_columns, expected[d.name], f"{d.name}: source_columns")
        if d.role == "predictor":
            require(spec.family in ap.TABLES, f"predictor prohibido: {d.name}")
            require(
                all((d.source_table, c) in approved for c in d.source_columns),
                f"predictor con columna no aprobada: {d.name}",
            )
        if spec.family in ap.TABLES:
            rule = (
                f"{ap.CREATION} <= date_decision; hijos solo con padre válido"
                if spec.family == "applprev"
                else "atributos de la solicitud actual; fechas internas <= date_decision"
            )
        elif spec.family in bureau.FAMILIES:
            rule = bureau.TEMPORAL_RULES[d.source_table]
        else:
            rule = (
                depth0.TEMPORAL_RULE
                if spec.family in depth0.SOURCES
                else products.TEMPORAL_RULE
            )
        template = contracts.describe_features(
            pl.DataFrame(schema={d.name: schema[d.name]}),
            temporal_rule=rule,
            source_columns={c.lower(): c for c in d.source_columns},
        )[0]
        check_equal(d, template, f"FeatureDefinition {d.name}")


def validate_keys(
    path: Path, base: pl.DataFrame, expected_rows: int, manifest: dict
) -> None:
    keys = pl.read_parquet(path, columns=["case_id"])
    check_equal(keys.height, expected_rows, "filas físicas")
    check_equal(manifest["row_count"], expected_rows, "row_count")
    check_equal(manifest["case_count"], expected_rows, "case_count")
    contracts.validate_base_keys(keys)
    require(keys["case_id"].is_sorted(), "case_id no ordenado")
    require(
        keys.equals(base.select("case_id").sort("case_id")),
        "claves/tipo no coinciden exactamente con base",
    )


def coverage_metrics(path: Path, table: str, base_cases: int) -> dict:
    """Lee únicamente el par de columnas de cobertura de una tabla."""
    n, present = f"{table}__n_records", f"{table}__source_present"
    frame = pl.read_parquet(path, columns=[n, present])
    require(
        frame.schema[n].is_integer() and frame.schema[present] == pl.Boolean,
        "tipos de cobertura inválidos",
    )
    require(
        not frame[n].null_count() and not frame[present].null_count(), "cobertura nula"
    )
    require(not (frame[n] < 0).any(), "n_records negativo")
    require(
        ((frame[n] > 0) == frame[present]).all(), "source_present != (n_records > 0)"
    )
    cases = int(frame[present].sum())
    return {
        "rows": int(frame[n].cast(pl.Int64).sum()),
        "cases": cases,
        "base_cases": base_cases,
        "cases_without_source": base_cases - cases,
        "coverage_pct": 100 * cases / base_cases if base_cases else 0.0,
    }


def exclusion_unit(metric: str) -> str:
    if "_values_" in metric:
        return "values"
    if "_rows_" in metric or metric.startswith("source_rows_"):
        return "rows"
    if metric.startswith("b6_") or metric in {
        "g2_approved",
        "manual_review",
        "exclude_predictor",
        "not_selected_from_source",
    }:
        return "columns"
    return "unknown"


def reconcile_cardinality(
    coverage: list[dict], exclusions: list[dict], cardinality: pl.DataFrame
) -> list[dict]:
    """Conciliación train+OOT; solo suma contadores de filas disjuntos conocidos."""
    require(
        not cardinality["family"].is_duplicated().any(),
        "cardinalidad: tablas duplicadas",
    )
    raw = {r["family"]: r for r in cardinality.iter_rows(named=True)}
    metrics = {(r["split"], r["family"], r["metric"]): r["value"] for r in exclusions}
    results = []
    tables = {r["table"] for r in coverage}
    for table in sorted(tables):
        rows = [
            r
            for r in coverage
            if r["table"] == table and r["split"] in ("train", "oot")
        ]
        item = {"table": table, "status": "invalid", "error": None}
        try:
            require(
                len(rows) == 2 and {r["split"] for r in rows} == {"train", "oot"},
                "faltan coberturas train/OOT",
            )
            source = raw[table]
            raw_rows, raw_cases = int(source["rows"]), int(source["covered_base_cases"])
            item.update(raw_rows=raw_rows, raw_cases=raw_cases)
            if table == A2:
                require(
                    all(r["status"] == "skipped" for r in rows),
                    "A2: plan skipped no validado",
                )
                item.update(
                    status="skipped",
                    rows=0,
                    cases=0,
                    lost_cases=raw_cases,
                    reason="sin ancla; no leída; cifras brutas solo del informe histórico",
                )
                results.append(item)
                continue
            require(all(r["status"] == "ok" for r in rows), "cobertura no comprobada")
            kept, cases = sum(r["rows"] for r in rows), sum(r["cases"] for r in rows)
            temporal = table.startswith(("applprev_", "credit_bureau_"))
            suffixes = (
                ("null_creation", "after_decision")
                if table == "applprev_1"
                else (
                    ("invalid_parent",)
                    if table == "applprev_2"
                    else ("null_anchor", "after_decision", "invalid_parent")
                    if table == "credit_bureau_b_2"
                    else ("null_anchor", "after_decision")
                    if temporal
                    else ()
                )
            )
            removed = 0
            for r in rows:
                for suffix in suffixes:
                    removed += metrics[
                        (r["split"], r["family"], f"{table}_rows_{suffix}")
                    ]
                if table in ("applprev_1", "person_1"):
                    check_equal(
                        metrics[
                            (r["split"], r["family"], f"{table}_rows_null_decision")
                        ],
                        0,
                        "null_decision: necesario para disjunción",
                    )
            item.update(
                rows=kept,
                cases=cases,
                row_exclusions_disjoint=removed,
                lost_cases=raw_cases - cases,
                coverage_pct=100 * cases / sum(r["base_cases"] for r in rows),
            )
            check_equal(int(source["orphan_rows"]), 0, "cardinalidad fuera de base")
            check_equal(
                kept + removed, raw_rows, "filas válidas + exclusiones disjuntas"
            )
            require(cases <= raw_cases, "cobertura supera bruto")
            if not temporal:
                check_equal(cases, raw_cases, "cobertura física")
            item["status"] = "ok"
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)
    return results


def compare_catalog(artifacts: list, path: Path) -> None:
    """Comparación completa, sin deduplicar silenciosamente esquemas divergentes."""
    seen = {}
    owners = {}
    for artifact in artifacts:
        for definition in artifact.feature_schema:
            if definition.name in seen:
                check_equal(
                    definition, seen[definition.name], "definición entre splits"
                )
                check_equal(
                    artifact.family,
                    owners[definition.name],
                    "nombre compartido entre familias",
                )
            seen[definition.name] = definition
            owners[definition.name] = artifact.family
    expected = contracts.feature_catalog(artifacts)
    current = pl.read_csv(
        path,
        schema_overrides=contracts.CATALOG_SCHEMA,
    ).fill_null("")
    check_equal(current.columns, list(contracts.CATALOG_SCHEMA), "columnas catálogo")
    require(not current["name"].is_duplicated().any(), "catálogo duplicado")
    require(current.sort("name").equals(expected), f"catálogo completo difiere: {path}")


def compare_reports(
    results_dir: Path,
    artifacts: list,
    coverage: list[dict],
    exclusions: list[dict],
    checks: Checks,
) -> None:
    index = {
        (r["family"], r["split"], r["table"]): r
        for r in coverage
        if r["status"] == "ok"
    }
    for filename, families in (
        ("C1_depth0_run.csv", depth0.SOURCES),
        ("C2_C3_applications_people_run.csv", ap.TABLES),
        ("C6_products_run.csv", products.SOURCES),
    ):

        def compare(filename=filename, families=families):
            old = pl.read_csv(results_dir / filename)
            require(
                not old.select("family", "split").is_duplicated().any(),
                "reporte duplicado",
            )
            actual = []
            for a in artifacts:
                if a.family not in families:
                    continue
                tables = ap.TABLES.get(a.family, (a.family,))
                c = index[(a.family, a.split, tables[0])]
                predictors = sum(d.role == "predictor" for d in a.feature_schema)
                row = {
                    "family": a.family,
                    "split": a.split,
                    "base_cases": a.case_count,
                    "cases_with_source": c["cases"],
                    "cases_without_source": c["cases_without_source"],
                    "predictor_features": predictors,
                    "coverage_features": len(a.feature_schema) - predictors,
                }
                if a.family in ap.TABLES:
                    row.update(
                        valid_rows_table_1=c["rows"],
                        valid_rows_table_2=index[(a.family, a.split, tables[1])][
                            "rows"
                        ],
                    )
                else:
                    row["source_rows_for_base"] = c["rows"]
                actual.append(row)
            observed = pl.DataFrame(actual).select(old.columns).sort("family", "split")
            require(
                old.sort("family", "split").equals(observed),
                f"{filename}: métricas/inventario difieren",
            )

        checks.run("global", filename, compare)

    def compare_f7():
        old = pl.read_csv(results_dir / "F7_applications_people_exclusions.csv")
        rows = [
            {k: r[k] for k in ("split", "family", "metric", "value")}
            for r in exclusions
            if r["family"] in ap.TABLES
        ]
        current = pl.DataFrame(rows).select(old.columns)
        require(
            old.sort("family", "split", "metric").equals(
                current.sort("family", "split", "metric")
            ),
            "F7 difiere",
        )

    checks.run("global", "F7_exclusions", compare_f7)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in rows)


def run_audit(
    *,
    artifact_root: Path | None = None,
    results_dir: Path | None = None,
    cardinality_path: Path | None = None,
) -> dict:
    """Audita y persiste los cuatro reportes; devuelve el JSON (ok=False si falla).

    Un artefacto inválido no detiene los demás. blocked indica una dependencia
    ausente/no validada. reused significa validado para reutilizar, no reconstruido.
    """
    started = time.perf_counter()
    results_dir = Path(results_dir or ROOT / "results/preparation")
    artifact_root = Path(artifact_root or io.data_dir() / "artifacts/preparation")
    cardinality_path = Path(
        cardinality_path or ROOT / "results/integration_analysis/family_cardinality.csv"
    )
    checks = Checks()
    build_records_path = results_dir / "E4_builds.json"
    build_records = (
        json.loads(build_records_path.read_text())
        if build_records_path.exists()
        else []
    )
    runs, coverage, exclusions, artifacts = [], [], [], []
    reconciliation, hashes = [], {}
    specs = family_specs()
    report = {
        "audit": "E4",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff_week": 74,
        "expected_rows": EXPECTED_ROWS,
        "limitations": LIMITATIONS,
        "hashes_scope": "current_at_audit_not_historical_build",
        "current_sha256": hashes,
        "artifact_root": str(artifact_root),
        "data_root": str(io.data_dir()),
        "polars_version": pl.__version__,
        "checks": checks.rows,
    }
    serial = checks.run(
        "global",
        "serial_runtime",
        lambda: (
            require(
                os.environ.get("POLARS_MAX_THREADS") == "1",
                "POLARS_MAX_THREADS=1 obligatorio antes de iniciar Python",
            ),
            check_equal(pl.thread_pool_size(), 1, "pool efectivo Polars"),
            True,
        )[-1],
    )
    bases = {}
    if serial:

        def load_bases():
            parts = partition_base(
                io.load_base("train"), io.load_base("test"), cutoff_week=74
            )
            out = {s: getattr(parts, s) for s in EXPECTED_ROWS}
            for split, base in out.items():
                check_equal(base.height, EXPECTED_ROWS[split], f"base {split}")
            return out

        bases = checks.run("global", "base_partitions", load_bases) or {}
        paths = {
            ROOT / "uv.lock",
            ROOT / "pyproject.toml",
            ROOT / "src/dm_credit/io.py",
            ROOT / "src/dm_credit/baseline.py",
            cardinality_path,
        }
        paths.update((ROOT / "src/dm_credit/preparation").rglob("*.py"))
        paths.update(results_dir.glob("B*.csv"))
        paths.update(results_dir.glob("G2*.csv"))
        paths.update(results_dir.glob("*feature_catalog*.csv"))
        paths.update(
            results_dir / name
            for name in (
                "C1_depth0_run.csv",
                "C2_C3_applications_people_run.csv",
                "C6_products_run.csv",
                "F7_applications_people_exclusions.csv",
            )
        )
        for path in sorted(paths):
            value = checks.run(
                "global",
                f"current_hash:{path.name}",
                lambda path=path: sha256_file(path),
            )
            hashes[str(path)] = value
    base_hashes = {
        s: checks.run("global", f"base_hash:{s}", lambda b=b: base_fingerprint(b))
        for s, b in bases.items()
    }
    report["current_base_sha256"] = base_hashes
    report["physical_schemas"] = {}
    for spec in specs:
        reference_schema = reference_definitions = None
        decisions = plan = expected = None
        tables = tuple(t for t in spec.tables if t != A2)
        if serial:
            decisions = checks.run(
                spec.family,
                "G2",
                lambda: load_decisions(
                    results_dir / spec.decisions,
                    spec.tables,
                    zero=spec.family not in ap.TABLES,
                ),
            )
            if spec.family in ap.TABLES:

                def config_names():
                    configs = g2.load_g2_configs(
                        results_dir / g2.G2_COLUMNS.name,
                        results_dir / g2.G2_VOCABULARY.name,
                    )
                    return expected_ap_features(
                        spec.family, configs[0 if spec.family == "applprev" else 1]
                    )

                expected = checks.run(spec.family, "config_vocabulary", config_names)
            if spec.family in bureau.FAMILIES:

                def current_plan():
                    resolved = bureau.resolve_columns(
                        bureau.load_selection(
                            results_dir / bureau.DEFAULT_SELECTION_PATH.name
                        ),
                        bureau.load_g2_decisions(
                            results_dir / bureau.DEFAULT_G2_PATH.name
                        ),
                    )
                    p = bureau.plan_from_decisions(spec.family, resolved)
                    check_equal(
                        set(p.skipped),
                        {A2} if spec.family == "credit_bureau_a" else set(),
                        "tablas skipped",
                    )
                    return p

                plan = checks.run(spec.family, "bureau_public_plan", current_plan)
                if plan is not None:
                    tables = tuple(t.table for t in plan.tables)
        for split in EXPECTED_ROWS:
            scope = f"{spec.family}/{split}"
            tick = time.perf_counter()
            path = (
                artifact_root
                / spec.directory
                / spec.family
                / split
                / f"{spec.family}__{split}.manifest.json"
            )
            replacement = artifact_root / "E4-v1" / spec.family / split / path.name
            if replacement.exists():
                path = replacement
            builds = [
                r
                for r in build_records
                if (r["family"], r["split"]) == (spec.family, split)
            ]
            build_record = builds[-1] if builds else None
            run = {
                "family": spec.family,
                "split": split,
                "manifest": str(path),
                "status": "blocked",
                "build_seconds": None,
                "build_ru_maxrss_kib": None,
                "origin": "prior_artifact",
                "build_blocker": build_record.get("error")
                if build_record and build_record["status"] == "blocked"
                else None,
            }
            if (
                replacement.exists()
                and build_record
                and build_record["status"] == "generated_pending_audit"
            ):
                run.update(
                    origin="generated_E4",
                    build_seconds=build_record["external_resources"]["seconds"],
                    build_ru_maxrss_kib=build_record["external_resources"][
                        "peak_rss_kib"
                    ],
                )
                checks.run(
                    scope,
                    "E4_build_manifest_hash",
                    lambda: check_equal(
                        sha256_file(path),
                        build_record["manifest_sha256"],
                        "manifest generado",
                    ),
                )
            runs.append(run)
            dependency_ok = (
                bool(bases)
                and base_hashes.get(split) is not None
                and decisions is not None
                and (spec.family not in ap.TABLES or expected is not None)
                and (spec.family not in bureau.FAMILIES or plan is not None)
            )
            manifest = artifact = None
            if serial:
                manifest = checks.run(
                    scope, "manifest_json", lambda: json.loads(path.read_text())
                )
                artifact = checks.run(
                    scope,
                    "load_family_artifact_checksum",
                    lambda: contracts.load_family_artifact(path),
                )
            if artifact is not None:
                artifacts.append(artifact)
                run.update(
                    rows=artifact.row_count,
                    cases=artifact.case_count,
                    columns=1 + len(artifact.feature_schema),
                    predictors=sum(
                        d.role == "predictor" for d in artifact.feature_schema
                    ),
                    coverage_features=sum(
                        d.role == "cobertura" for d in artifact.feature_schema
                    ),
                    manifest_sha256=sha256_file(path),
                    data_sha256=sha256_file(artifact.data_path),
                )
            if isinstance(manifest, dict):

                def read_exclusions():
                    for metric, value in manifest["exclusions"].items():
                        require(
                            type(value) is int and value >= 0,
                            f"exclusión no entera/negativa: {metric}",
                        )
                        if metric.endswith("_rows_null_decision"):
                            check_equal(
                                value,
                                0,
                                f"{metric}: base validada sin decisiones nulas",
                            )
                        exclusions.append(
                            {
                                "family": spec.family,
                                "split": split,
                                "metric": metric,
                                "value": value,
                                "unit": exclusion_unit(metric),
                                "origin": "manifest",
                            }
                        )

                checks.run(scope, "exclusions", read_exclusions)
            if artifact is not None and manifest is not None and dependency_ok:
                run["status"] = "reused"

                def metadata():
                    check_equal(
                        manifest["format_version"],
                        contracts.FORMAT_VERSION,
                        "format_version",
                    )
                    check_equal(
                        (artifact.family, artifact.split),
                        (spec.family, split),
                        "identidad",
                    )
                    check_equal(
                        manifest["data_path"],
                        f"{spec.family}__{split}.parquet",
                        "data_path",
                    )
                    check_equal(
                        manifest["rules_version"], spec.rules_version, "rules_version"
                    )
                    inputs = manifest["inputs"]
                    check_equal(inputs["cutoff_week"], "74", "cutoff_week")
                    check_equal(
                        inputs["base_sha256"], base_hashes[split], "base_sha256"
                    )
                    check_equal(
                        inputs["shards"],
                        shard_fingerprint(split, tables, ordered=plan is not None),
                        "shards actuales",
                    )
                    if spec.family not in ap.TABLES:
                        selection = (
                            bureau.DEFAULT_SELECTION_PATH
                            if plan is not None
                            else depth0.DEFAULT_SELECTION_PATH
                            if spec.family in depth0.SOURCES
                            else products.DEFAULT_SELECTION_PATH
                        )
                        check_equal(
                            Path(inputs["selection_manifest"]).name,
                            selection.name,
                            "selección vigente",
                        )
                        check_equal(
                            Path(inputs["g2_decision"]).name,
                            spec.decisions,
                            "decisión vigente",
                        )
                    if plan is not None:
                        check_equal(inputs["g2_mode"], "approved", "g2_mode")
                        check_equal(
                            inputs["plan_sha256"], plan.fingerprint(), "plan_sha256"
                        )
                        check_equal(
                            {
                                k.removeprefix("skipped_table:"): v
                                for k, v in inputs.items()
                                if k.startswith("skipped_table:")
                            },
                            dict(plan.skipped),
                            "skipped A2 obligatorio",
                        )

                checks.run(scope, "metadata_current_inputs", metadata)
                checks.run(
                    scope,
                    "exact_base_keys",
                    lambda: validate_keys(
                        artifact.data_path, bases[split], EXPECTED_ROWS[split], manifest
                    ),
                )
                schema = checks.run(
                    scope,
                    "physical_schema",
                    lambda: pl.scan_parquet(artifact.data_path).collect_schema(),
                )
                if schema is not None:
                    report["physical_schemas"][scope] = {
                        n: str(t) for n, t in schema.items()
                    }
                    if reference_schema is None:
                        reference_schema = schema
                        reference_definitions = artifact.feature_schema
                    checks.run(
                        scope,
                        "physical_schema_across_splits",
                        lambda: check_equal(
                            schema, reference_schema, "tipos físicos entre splits"
                        ),
                    )
                    checks.run(
                        scope,
                        "definitions_across_splits",
                        lambda: check_equal(
                            artifact.feature_schema,
                            reference_definitions,
                            "definiciones ordenadas entre splits",
                        ),
                    )
                    checks.run(
                        scope,
                        "feature_definitions",
                        lambda: validate_definitions(
                            artifact, schema, spec, tables, decisions, expected
                        ),
                    )
                    for name, dtype in schema.items():
                        if dtype.is_float():
                            checks.run(
                                scope,
                                f"finite:{name}",
                                lambda name=name: require(
                                    not pl.scan_parquet(artifact.data_path)
                                    .select(pl.col(name).is_infinite().any())
                                    .collect()
                                    .item(),
                                    f"infinitos en {name}",
                                ),
                            )
            for table in spec.tables:
                row = {
                    "family": spec.family,
                    "split": split,
                    "table": table,
                    "status": "blocked",
                }
                if table == A2 and plan is not None:
                    row.update(
                        status="skipped",
                        rows=0,
                        cases=0,
                        base_cases=EXPECTED_ROWS[split],
                        cases_without_source=EXPECTED_ROWS[split],
                        coverage_pct=0.0,
                    )
                elif artifact is not None and dependency_ok:
                    metrics = checks.run(
                        scope,
                        f"coverage:{table}",
                        lambda: coverage_metrics(
                            artifact.data_path, table, EXPECTED_ROWS[split]
                        ),
                    )
                    if metrics is not None:
                        row.update(status="ok", **metrics)
                coverage.append(row)
            run.update(
                audit_seconds=time.perf_counter() - tick,
                audit_process_ru_maxrss_kib=resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
            )
    if serial:
        for filename in sorted({s.catalog for s in specs}):
            families = {s.family for s in specs if s.catalog == filename}
            checks.run(
                "global",
                filename,
                lambda filename=filename, families=families: compare_catalog(
                    [a for a in artifacts if a.family in families],
                    results_dir / filename,
                ),
            )
        checks.run(
            "global",
            "C8_complete_catalog",
            lambda: compare_catalog(artifacts, results_dir / "C8_feature_catalog.csv"),
        )
        checks.run(
            "global",
            "artifact_inventory",
            lambda: check_equal(
                sorted((a.family, a.split) for a in artifacts),
                sorted((s.family, split) for s in specs for split in EXPECTED_ROWS),
                "36 artefactos",
            ),
        )
        cardinality = checks.run(
            "global", "cardinality_read", lambda: pl.read_csv(cardinality_path)
        )
        if cardinality is not None:
            reconciliation = (
                checks.run(
                    "global",
                    "cardinality_reconciliation",
                    lambda: reconcile_cardinality(coverage, exclusions, cardinality),
                )
                or []
            )
            for item in reconciliation:
                if item["status"] == "invalid":
                    family = next(s.family for s in specs if item["table"] in s.tables)
                    checks.rows.append(
                        {
                            "scope": family,
                            "check": f"cardinality:{item['table']}",
                            "status": "invalid",
                            "error": item["error"],
                        }
                    )
        compare_reports(results_dir, artifacts, coverage, exclusions, checks)
        # El bruto solo existe para train físico: no atribuir pérdidas a cada split.
        for item in reconciliation:
            total = EXPECTED_ROWS["train"] + EXPECTED_ROWS["oot"]
            coverage.append(
                {
                    **item,
                    "family": next(
                        s.family for s in specs if item["table"] in s.tables
                    ),
                    "split": "train+oot",
                    "base_cases": total,
                    "cases_without_source": total - item["cases"]
                    if "cases" in item
                    else None,
                    "coverage_pct": 100 * item["cases"] / total
                    if "cases" in item
                    else None,
                }
            )
    failures = [r for r in checks.rows if r["status"] != "ok"]
    for run in runs:
        relevant = [
            r
            for r in failures
            if r["scope"]
            in ("global", run["family"], f"{run['family']}/{run['split']}")
        ]
        if relevant and run["status"] != "blocked":
            run["status"] = "invalid"
        # A malformed/missing artifact is invalid, not just a blocked dependency.
        if any(r["scope"] == f"{run['family']}/{run['split']}" for r in relevant):
            run["status"] = "invalid"
        run["errors"] = json.dumps(relevant, ensure_ascii=False) if relevant else None
        if run["status"] == "reused" and run["origin"] == "generated_E4":
            run["status"] = "generated_validated"
    statuses = {(r["family"], r["split"]): r["status"] for r in runs}
    for row in coverage:
        splits = ("train", "oot") if row["split"] == "train+oot" else (row["split"],)
        row["artifact_provenance"] = ";".join(
            f"{s}:{statuses[row['family'], s]}" for s in splits
        )
        row["current_inputs_valid"] = all(
            statuses[row["family"], s] in ("reused", "generated_validated")
            for s in splits
        )
    for item in reconciliation:
        family = next(s.family for s in specs if item["table"] in s.tables)
        item["current_inputs_valid"] = all(
            statuses[family, s] in ("reused", "generated_validated")
            for s in ("train", "oot")
        )
    report.update(
        ok=not failures
        and all(r["status"] in ("reused", "generated_validated") for r in runs),
        runs=runs,
        coverage=coverage,
        exclusions=exclusions,
        cardinality=reconciliation,
        audit_seconds=time.perf_counter() - started,
        audit_process_ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource_scope="cumulative auditor process high-water mark, Linux KiB; not build resources",
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        results_dir / "E4_runs.csv",
        runs,
        [
            "family",
            "split",
            "manifest",
            "status",
            "origin",
            "rows",
            "cases",
            "columns",
            "predictors",
            "coverage_features",
            "manifest_sha256",
            "data_sha256",
            "build_blocker",
            "build_seconds",
            "build_ru_maxrss_kib",
            "audit_seconds",
            "audit_process_ru_maxrss_kib",
            "errors",
        ],
    )
    _write_csv(
        results_dir / "E4_coverage.csv",
        coverage,
        [
            "family",
            "split",
            "table",
            "status",
            "artifact_provenance",
            "current_inputs_valid",
            "rows",
            "cases",
            "base_cases",
            "cases_without_source",
            "coverage_pct",
            "raw_rows",
            "raw_cases",
            "row_exclusions_disjoint",
            "lost_cases",
            "reason",
            "error",
        ],
    )
    _write_csv(
        results_dir / "E4_exclusions.csv",
        exclusions,
        ["family", "split", "metric", "value", "unit", "origin"],
    )
    (results_dir / "E4_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--cardinality", type=Path)
    args = parser.parse_args(argv)
    report = run_audit(
        artifact_root=args.artifact_root,
        results_dir=args.results_dir,
        cardinality_path=args.cardinality,
    )
    print(
        f"E4: {'ok' if report['ok'] else 'invalid/blocked'}; {len(report['runs'])} artefactos auditados"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
