"""Reconciliación de catálogo, especificación y matrices locales; sin ajustar modelos."""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONTROLS = {"case_id", "date_decision", "WEEK_NUM", "MONTH", "target"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    checks = []

    def check(scope, name, valid, detail):
        checks.append(
            dict(
                scope=scope,
                check=name,
                status="ok" if valid else "revisar",
                detail=detail,
            )
        )

    catalog_path = ROOT / "results/preparation/C8_feature_catalog.csv"
    spec_path = ROOT / "data/artifacts/preparation/E6-v1/matrix_views_spec.csv"
    catalog = read_csv(catalog_path)
    names = {row["name"] for row in catalog}
    spec = read_csv(spec_path)
    for view in ["linear", "tree"]:
        entries = [
            r
            for r in spec
            if r["view"] == view
            or (
                r["role"] == "dropped"
                and view in json.loads(r["parameters"]).get("views", [])
            )
        ]
        outputs = [r["output_column"] for r in entries if r["role"] != "dropped"]
        origins = {r["source_feature"] for r in entries if r["role"] != "control"}
        check(
            view,
            "unique_active_outputs",
            len(outputs) == len(set(outputs)),
            str(len(outputs)),
        )
        check(
            view,
            "catalog_origins",
            origins == names,
            json.dumps(
                dict(missing=sorted(names - origins), unknown=sorted(origins - names))
            ),
        )
        predictors = [r for r in entries if r["role"] == "predictor"]
        check(
            view,
            "no_reserved_predictors",
            not any(
                r["output_column"] in CONTROLS or r["source_feature"] in CONTROLS
                for r in predictors
            ),
            str(len(predictors)),
        )
        check(
            view,
            "valid_parameters_json",
            all(isinstance(json.loads(r["parameters"]), dict) for r in entries),
            "Parámetros JSON interpretables; no valida su ajuste estadístico",
        )
        check(
            view,
            "control_set",
            {r["output_column"] for r in entries if r["role"] == "control"} == CONTROLS,
            "Controles declarados",
        )

    report_rows = []
    for report in ["E5_master_tables.csv", "E6_views.csv"]:
        for row in read_csv(ROOT / "results/preparation" / report):
            relative = row["data_path"].split("/data-mining-labs/", 1)[-1]
            path = ROOT / relative
            scope = row.get("view", "master") + "/" + row["split"]
            if not path.is_file():
                checks.append(
                    dict(
                        scope=scope,
                        check="matrix_content",
                        status="no_ejecutado",
                        detail="Archivo ausente: " + relative,
                    )
                )
                continue
            frame = pl.read_parquet(path)
            controls = CONTROLS - ({"target"} if row["split"] == "test" else set())
            expected = (
                names
                if report.startswith("E5")
                else {
                    r["output_column"]
                    for r in spec
                    if r["view"] == row["view"] and r["role"] == "predictor"
                }
            ) | controls
            check(
                scope,
                "schema_columns",
                set(frame.columns) == expected,
                json.dumps(
                    dict(
                        missing=sorted(expected - set(frame.columns)),
                        unexpected=sorted(set(frame.columns) - expected),
                    )
                ),
            )
            check(
                scope,
                "shape",
                frame.height == int(row["rows"]) and frame.width == int(row["columns"]),
                f"{frame.height} filas, {frame.width} columnas",
            )
            check(
                scope,
                "unique_cases",
                frame["case_id"].null_count() == 0
                and frame["case_id"].n_unique() == frame.height,
                "Unicidad y no nulos de case_id",
            )
            if row["split"] == "oot":
                check(
                    scope,
                    "oot_weeks",
                    frame["WEEK_NUM"].null_count() == 0
                    and frame["WEEK_NUM"].min() >= 74,
                    "WEEK_NUM >= 74",
                )
                check(
                    scope,
                    "target_binary",
                    frame["target"].null_count() == 0
                    and set(frame["target"].unique().to_list()) <= {0, 1},
                    "Etiqueta binaria no nula",
                )
                master = (
                    ROOT
                    / "data/artifacts/preparation/E5-v1/feature_matrix__oot.parquet"
                )
                reference = pl.read_parquet(master, columns=list(CONTROLS))
                check(
                    scope,
                    "controls_match_master",
                    frame.select(sorted(CONTROLS)).equals(
                        reference.select(sorted(CONTROLS))
                    ),
                    "Mismos casos, orden, etiquetas y controles",
                )
            if row.get("view") == "linear":
                predictors = sorted(expected - controls)
                check(
                    scope,
                    "no_null_predictors",
                    sum(frame.select(predictors).null_count().row(0)) == 0,
                    "Ausencia de nulos en predictores lineales",
                )
            report_rows.append(
                dict(scope=scope, rows=frame.height, columns=frame.width)
            )

    with (OUT / "controles-vistas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["scope", "check", "status", "detail"]
        )
        writer.writeheader()
        writer.writerows(checks)
    result = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        statuses=dict(Counter(r["status"] for r in checks)),
        matrices=report_rows,
        spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        limitations=[
            "Sin entrenamiento local: no se verifican parámetros contra sus datos de ajuste.",
            "Sin validación de semántica temporal ni prueba de ausencia de fuga.",
            "Los controles ejecutados no equivalen a repetir F3 completo.",
        ],
    )
    (OUT / "resumen-vistas.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for row in checks:
        if row["status"] == "revisar":
            print(row)


if __name__ == "__main__":
    main()
