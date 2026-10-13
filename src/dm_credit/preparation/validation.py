"""F3, F4 y F7 — Validación de los artefactos publicados, resultados por familia e informe de fuga.

Trabaja sobre los artefactos de familia ya publicados (sus manifiestos), no sobre las fuentes:

    - ``validate_pipeline``: controles de F3 por split y entre splits (conteos, base común, cobertura de
      casos, ensamblado, finitud, catálogo, reservadas, partición, esquema, contrato del baseline y vistas
      ajustadas solo con train). Un control fallido se informa como ``fail``; nunca se omite en silencio.
    - ``family_results``: casos, cobertura y roles por familia y split (insumo de F4).
    - ``leakage_report``: filas y valores excluidos por regla temporal, tomados de cada manifiesto (F7).

Uso con los artefactos reales:
    POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.validation
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from dm_credit import baseline, io
from dm_credit.baseline import RESERVED
from dm_credit.preparation import contracts
from dm_credit.preparation.assembly import assemble_feature_matrix
from dm_credit.preparation.preprocessing import build_spec_from_catalog
from dm_credit.preparation.schema import validate_matrix_schemas
from dm_credit.preparation.split import partition_base
from dm_credit.preparation.views import fit_matrix_views

KEY = "case_id"
SPLITS = ("train", "oot", "test")
CONTROLS = ("case_id", "WEEK_NUM", "date_decision", "MONTH", "target")
# Cifras documentadas del capítulo 02 y del backlog (§9): casos y positivos con el corte 74.
EXPECTED_REAL = {"train": (1_332_001, 43_734), "oot": (194_658, 4_260)}
RESULTS = Path(__file__).resolve().parents[3] / "results" / "preparation"
ARTIFACT_SETS = ("C1-v1", "C2-C3-v1", "C4-C5-v2", "C6-v1")


def _row(check: str, split: str, ok: bool, detail: str) -> dict:
    return {"check": check, "split": split, "status": "ok" if ok else "fail", "detail": detail}


def _manifest(art: contracts.FamilyFeatureArtifact) -> dict:
    return json.loads(Path(art.manifest_path).read_text())


def leakage_report(artifacts: Sequence[contracts.FamilyFeatureArtifact]) -> pl.DataFrame:
    """F7: una fila por (split, familia, regla) con lo que cada familia excluyó o anuló por temporalidad."""
    rows = []
    for art in artifacts:
        m = _manifest(art)
        for metric, value in m.get("exclusions", {}).items():
            rows.append({"split": art.split, "family": art.family, "rules_version": m.get("rules_version", ""), "metric": metric, "value": int(value)})
    schema = {"split": pl.String, "family": pl.String, "rules_version": pl.String, "metric": pl.String, "value": pl.Int64}
    return pl.DataFrame(rows, schema=schema).unique(["split", "family", "metric"], keep="first").sort(["split", "family", "metric"])


def family_results(artifacts: Sequence[contracts.FamilyFeatureArtifact], base: pl.DataFrame) -> pl.DataFrame:
    """F4: casos, casos con fuente, roles y versión de reglas de cada familia en un split."""
    rows = []
    for art in artifacts:
        frame = pl.read_parquet(art.data_path)
        present = [c for c in frame.columns if c.endswith("__source_present")]
        covered = int(frame.select(pl.any_horizontal(present).sum()).item()) if present else 0
        predictors = sum(d.role == "predictor" for d in art.feature_schema)
        rows.append(
            {
                "split": art.split,
                "family": art.family,
                "rules_version": _manifest(art).get("rules_version", ""),
                "cases": frame.height,
                "cases_with_source": covered,
                "coverage_pct": round(100.0 * covered / base.height, 4) if base.height else 0.0,
                "predictor_features": predictors,
                "coverage_features": len(art.feature_schema) - predictors,
            }
        )
    return pl.DataFrame(rows).sort(["split", "family"])


def _split_checks(split: str, base: pl.DataFrame, arts, out_dir: Path, expected) -> tuple[list[dict], pl.DataFrame | None]:
    rows = []
    n_pos = int(base["target"].sum()) if "target" in base.columns else None
    if expected and split in expected:
        cases, positives = expected[split]
        rows.append(_row("base_cases", split, (base.height, n_pos) == (cases, positives), f"{base.height} casos, {n_pos} positivos; esperado {cases} y {positives}"))
    else:
        rows.append(_row("base_cases", split, base[KEY].n_unique() == base.height, f"{base.height} casos" + (f", {n_pos} positivos" if n_pos is not None else "")))

    shas = {a.family: _manifest(a).get("inputs", {}).get("base_sha256") for a in arts}
    distinct = set(shas.values())
    odd = sorted(f for f, s in shas.items() if list(shas.values()).count(s) == 1) if len(distinct) > 1 else []
    rows.append(_row("same_base_fingerprint", split, len(distinct) == 1, f"{len(arts)} familias con una sola huella de base" if len(distinct) == 1 else f"huellas distintas en: {', '.join(odd)}"))

    base_ids = set(base[KEY].to_list())
    uncovered = [a.family for a in arts if set(pl.read_parquet(a.data_path, columns=[KEY])[KEY].to_list()) != base_ids]
    rows.append(_row("artifacts_cover_base", split, not uncovered, "cada familia trae una fila por caso de la base" if not uncovered else f"no coinciden con la base: {', '.join(uncovered)}"))

    try:
        matrix = pl.read_parquet(assemble_feature_matrix(base, arts, split, output_dir=out_dir / split).data_path)
    except (ValueError, OSError) as exc:
        rows.append(_row("assembled_matrix", split, False, f"no se pudo ensamblar: {exc}"))
        return rows, None
    ok = matrix.height == base.height and matrix[KEY].n_unique() == matrix.height and matrix[KEY].to_list() == base[KEY].to_list()
    rows.append(_row("assembled_matrix", split, ok, f"{matrix.height} filas x {matrix.width} columnas, case_id único y en el orden de la base"))

    features = [c for c in matrix.columns if c not in CONTROLS]
    numeric = [c for c in features if matrix.schema[c].is_float()]
    bad = [c for c in numeric if matrix[c].is_infinite().any()]
    rows.append(_row("finite_values", split, not bad, f"{len(numeric)} columnas decimales sin infinitos" if not bad else f"infinitos en {bad}"))

    catalog = set(contracts.feature_catalog(arts)["name"].to_list())
    missing = sorted(set(features) - catalog)
    rows.append(_row("features_in_catalog", split, not missing, f"{len(features)} características, todas en el catálogo" if not missing else f"fuera del catálogo: {missing[:5]}"))

    reserved = sorted(set(features) & set(RESERVED))
    rows.append(_row("no_reserved_predictors", split, not reserved, "ninguna columna reservada es característica" if not reserved else f"reservadas como característica: {reserved}"))
    return rows, matrix


def validate_pipeline(
    bases: Mapping[str, pl.DataFrame],
    artifacts: Mapping[str, Sequence[contracts.FamilyFeatureArtifact]],
    *,
    output_dir: Path,
    cutoff_week: int = 74,
    expected: Mapping[str, tuple[int, int]] | None = None,
) -> pl.DataFrame:
    """F3: controles sobre los artefactos publicados. Devuelve una tabla ``check, split, status, detail``."""
    rows, matrices = [], {}
    families = {s: sorted(a.family for a in artifacts[s]) for s in SPLITS}
    same = families["train"] == families["oot"] == families["test"]
    rows.append(_row("family_set", "all", same, f"{len(families['train'])} familias en cada split" if same else f"familias distintas por split: {families}"))

    for split in SPLITS:
        split_rows, matrix = _split_checks(split, bases[split], artifacts[split], Path(output_dir), expected)
        rows += split_rows
        matrices[split] = matrix

    tr, oo = set(bases["train"][KEY].to_list()), set(bases["oot"][KEY].to_list())
    expected_total = sum(expected[s][0] for s in ("train", "oot")) if expected and {"train", "oot"} <= set(expected) else len(tr | oo)
    weeks_ok = bases["train"]["WEEK_NUM"].max() < cutoff_week <= bases["oot"]["WEEK_NUM"].min()
    rows.append(_row("train_oot_partition", "all", not (tr & oo) and len(tr | oo) == expected_total and weeks_ok,
                     f"train {len(tr)} + OOT {len(oo)} = {len(tr | oo)} casos, sin solapamiento, corte {cutoff_week}"))

    if any(m is None for m in matrices.values()):
        for name in ("schema_parity", "baseline_contract", "views_fit_on_train_only"):
            rows.append(_row(name, "all", False, "no se evaluó: falló el ensamblado de algún split"))
        return pl.DataFrame(rows)

    try:
        validate_matrix_schemas(matrices["train"], matrices["oot"], matrices["test"], cutoff_week=cutoff_week)
        rows.append(_row("schema_parity", "all", True, "mismos controles (salvo target) y mismas características, orden y tipos en train, OOT y test"))
    except ValueError as exc:
        rows.append(_row("schema_parity", "all", False, str(exc)))

    try:
        labeled = pl.concat([matrices["train"], matrices["oot"]], how="vertical_relaxed")
        baseline.validate_contract(labeled)
        res = baseline.run_baseline(labeled.select(KEY, "WEEK_NUM", "target"), cutoff_week, model="prevalence", features=[])
        ok = res.metrics["n_train"] == matrices["train"].height and res.metrics["n_val"] == matrices["oot"].height
        rows.append(_row("baseline_contract", "all", ok, f"la matriz entra sin cambios a dm_credit.baseline: {res.metrics['n_train']} de train y {res.metrics['n_val']} de OOT"))
    except ValueError as exc:
        rows.append(_row("baseline_contract", "all", False, str(exc)))

    try:
        spec = build_spec_from_catalog(contracts.feature_catalog(artifacts["train"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            views = fit_matrix_views(matrices["train"], spec, cutoff_week=cutoff_week)
            state = views.to_dict()
            nulls = 0
            for split in ("oot", "test"):
                lin = views.linear(matrices[split], split)
                pred = lin.drop([c for c in lin.columns if c in CONTROLS])
                nulls += sum(pred[c].null_count() for c in pred.columns)
                views.tree(matrices[split], split)
        ok = views.to_dict() == state and nulls == 0
        rows.append(_row("views_fit_on_train_only", "all", ok, f"vistas ajustadas con train; OOT y test no cambian el estado; {len(views.predictor_columns('linear'))} predictores lineales sin nulos"))
    except ValueError as exc:
        rows.append(_row("views_fit_on_train_only", "all", False, str(exc)))
    return pl.DataFrame(rows)


def write_reports(bases, artifacts, *, out_dir: Path, work_dir: Path, cutoff_week: int = 74, expected=None) -> pl.DataFrame:
    """Escribe ``integration_checks.csv``, ``temporal_leakage_report.csv`` y ``family_results.csv``."""
    checks = validate_pipeline(bases, artifacts, output_dir=work_dir, cutoff_week=cutoff_week, expected=expected)
    all_arts = [a for s in SPLITS for a in artifacts[s]]
    results = pl.concat([family_results(artifacts[s], bases[s]) for s in SPLITS])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks.write_csv(out_dir / "integration_checks.csv")
    leakage_report(all_arts).write_csv(out_dir / "temporal_leakage_report.csv")
    results.write_csv(out_dir / "family_results.csv")
    return checks


def load_published(roots: Sequence[Path]) -> dict[str, list[contracts.FamilyFeatureArtifact]]:
    """Artefactos publicados por split, descubiertos por sus manifiestos."""
    out = {s: [] for s in SPLITS}
    for root in roots:
        for split in SPLITS:
            out[split] += [contracts.load_family_artifact(m) for m in sorted(Path(root).rglob(f"*__{split}.manifest.json"))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", nargs="+", type=Path, default=[io.data_dir() / "artifacts" / "preparation" / s for s in ARTIFACT_SETS])
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--work-dir", type=Path, default=io.data_dir() / "artifacts" / "preparation" / "F3-validation")
    ap.add_argument("--cutoff-week", type=int, default=74)
    args = ap.parse_args()
    parts = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=args.cutoff_week)
    bases = {"train": parts.train, "oot": parts.oot, "test": parts.test}
    checks = write_reports(bases, load_published(args.artifacts), out_dir=args.out, work_dir=args.work_dir, cutoff_week=args.cutoff_week, expected=EXPECTED_REAL)
    with pl.Config(tbl_rows=-1, fmt_str_lengths=120, tbl_width_chars=200):
        print(checks)
    failed = checks.filter(pl.col("status") != "ok").height
    print(f"{checks.height - failed} controles ok, {failed} fallidos. Informes en {args.out}")


if __name__ == "__main__":
    main()
