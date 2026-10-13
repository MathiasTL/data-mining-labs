"""F7 — Informe de fuga temporal consolidado por familia.

Une, en una fila por familia, la regla de disponibilidad declarada en los manifiestos de E4, las filas
excluidas y los valores anulados por split (``exclusions`` de cada manifiesto), las columnas de la
compuerta G2 por decisión y la evidencia que respalda cada número.

Uso:
    uv run -m dm_credit.preparation.f7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from dm_credit import io
from dm_credit.preparation import contracts
from dm_credit.preparation.e4 import exclusion_unit
from dm_credit.preparation.e4_rebuild import FAMILY_ORDER
from dm_credit.preparation.validation import load_published

SPLITS = ("train", "oot", "test")
RESULTS = Path(__file__).resolve().parents[3] / "results" / "preparation"
G2_FILES = ("G2_applications_people_columns.csv", "G2_depth0_columns.csv", "G2_bureau_columns.csv", "G2_products_registry_columns.csv")
G2_DECISIONS = {"aprobada": "columns_approved", "bloqueada": "columns_blocked", "excluida": "columns_excluded", "aplazada": "columns_deferred"}
# filas que no pertenecen al split: no son exclusiones temporales
NOT_TEMPORAL = ("source_rows_outside_base",)

SUMMARY_COLUMNS = (
    "family",
    "source_tables",
    "temporal_rule",
    "as_of_row_filter",
    "predictors",
    "coverage",
    *(f"{kind}_{s}" for kind in ("rows_excluded", "values_nulled") for s in SPLITS),
    *G2_DECISIONS.values(),
    "evidence",
)


def g2_family(table: str) -> str:
    """Familia de E4 a la que pertenece una tabla física de la compuerta G2."""
    for prefix in ("applprev", "person", "credit_bureau_a", "credit_bureau_b"):
        if table.startswith(prefix):
            return prefix
    return "static_0" if table == "base" else table


def load_g2(results: Path = RESULTS) -> pl.DataFrame:
    """Decisiones G2 de las cuatro hojas (tabla, columna, decisión)."""
    frames = [pl.read_csv(results / f, infer_schema_length=0).select("family", "column", "g2_decision") for f in G2_FILES]
    return pl.concat(frames)


def leakage_by_family(artifacts: Mapping[str, Sequence[contracts.FamilyFeatureArtifact]], g2: pl.DataFrame) -> pl.DataFrame:
    """Una fila por familia con su regla temporal, exclusiones por split, columnas G2 y evidencia."""
    rows: dict[str, dict] = {}
    for split in SPLITS:
        for art in artifacts[split]:
            manifest = json.loads(Path(art.manifest_path).read_text())
            r = rows.setdefault(art.family, {"family": art.family, "row_metrics": set(), "manifests": []})
            if split == "train" or "temporal_rule" not in r:
                features = manifest.get("features", [])
                r["source_tables"] = ";".join(sorted({f["source_table"] for f in features}))
                r["temporal_rule"] = " | ".join(sorted({f["temporal_rule"] for f in features if f.get("temporal_rule")}))
                r["predictors"] = sum(f["role"] == "predictor" for f in features)
                r["coverage"] = len(features) - r["predictors"]
            r["manifests"].append(Path(art.manifest_path).name)
            for kind in ("rows_excluded", "values_nulled"):
                r.setdefault(f"{kind}_{split}", 0)
            for metric, value in manifest.get("exclusions", {}).items():
                unit = exclusion_unit(metric)
                if unit == "rows" and metric not in NOT_TEMPORAL:
                    r[f"rows_excluded_{split}"] += int(value)
                    r["row_metrics"].add(metric)
                elif unit == "values":
                    r[f"values_nulled_{split}"] += int(value)

    counts = g2.with_columns(pl.col("family").map_elements(g2_family, return_dtype=pl.String)).group_by("family", "g2_decision").len()
    out = []
    order = {f: i for i, f in enumerate(FAMILY_ORDER)}
    for family in sorted(rows, key=lambda f: (order.get(f, len(order)), f)):
        r = rows[family]
        r["as_of_row_filter"] = bool(r.pop("row_metrics"))
        for decision, col in G2_DECISIONS.items():
            hit = counts.filter((pl.col("family") == family) & (pl.col("g2_decision") == decision))
            r[col] = int(hit["len"].sum())
        r["evidence"] = "E4 manifest: " + ", ".join(sorted(r.pop("manifests"))) + "; G2: " + ", ".join(G2_FILES)
        out.append(r)
    return pl.DataFrame(out).select(SUMMARY_COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=io.data_dir() / "artifacts" / "preparation" / "E4-v1")
    ap.add_argument("--out", type=Path, default=RESULTS / "F7_temporal_leakage_by_family.csv")
    args = ap.parse_args()
    report = leakage_by_family(load_published([args.artifacts]), load_g2())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.write_csv(args.out)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
        print(report.drop("source_tables", "temporal_rule", "evidence"))
    print(f"informe: {args.out}")


if __name__ == "__main__":
    main()
