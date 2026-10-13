"""Build the B3 depth-zero column-selection proposal from existing catalogs.

Usage:
    uv run -m dm_credit.preparation.depth0_selection
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import polars as pl

FAMILIES = ("base", "static_0", "static_cb_0")
CONTROL_COLUMNS = {
    ("base", "case_id"),
    ("base", "date_decision"),
    ("base", "MONTH"),
    ("base", "WEEK_NUM"),
    ("base", "target"),
}
QUASI_IDENTIFIER_PREFIX = "posible cuasi"
REVIEW_AVAILABILITY = "revisar disponibilidad al decidir"
REVIEW_DATE = "revisar fecha contra date_decision"
RISK_SCORE_COLUMNS = {"riskassesment_302T", "riskassesment_940T"}
REQUIRED_CATALOG_COLUMNS = {
    "family",
    "column",
    "dtype",
    "description",
    "definition_status",
    "function",
    "temporal_relevance",
    "identifier_risk",
    "leakage_review",
}
REQUIRED_REVIEW_COLUMNS = {
    "family",
    "classification",
    "temporal_columns",
    "state_or_outcome_review_columns",
    "identifier_or_quasi_identifier_columns",
    "other_domain_review_count",
    "temporal_reference",
    "pre_aggregation_rule",
    "leakage_risk",
    "recommended_action",
    "evidence",
}
OUTPUT_COLUMNS = [
    "family",
    "column",
    "dtype",
    "description",
    "proposed_action",
    "reason",
    "temporal_rule",
    "evidence",
]


def _validate_columns(
    frame: pl.DataFrame, required: set[str], input_name: str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{input_name} is missing required columns: {missing}")


def _split_semicolon_list(value: Any) -> set[str]:
    if value is None:
        return set()
    return {item.strip() for item in str(value).split(";") if item.strip()}


def _review_rows(review: pl.DataFrame) -> dict[str, dict[str, Any]]:
    _validate_columns(review, REQUIRED_REVIEW_COLUMNS, "temporal review")
    indexed: dict[str, dict[str, Any]] = {}
    for row in review.iter_rows(named=True):
        family = row["family"]
        if family in FAMILIES:
            if family in indexed:
                raise ValueError(f"duplicate temporal-review row for {family}")
            indexed[family] = row
    missing = sorted(set(FAMILIES) - set(indexed))
    if missing:
        raise ValueError(f"temporal review is missing depth-zero families: {missing}")
    return indexed


def _classify(
    row: dict[str, Any],
    family_review: dict[str, Any],
) -> tuple[str, str, str, str]:
    family, column = row["family"], row["column"]
    description = (row["description"] or "").strip()
    leakage_review = (row["leakage_review"] or "").strip().lower()
    identifier_risk = (row["identifier_risk"] or "").strip().lower()
    temporal_columns = _split_semicolon_list(family_review["temporal_columns"])
    review_columns = _split_semicolon_list(
        family_review["state_or_outcome_review_columns"]
    )
    identifier_columns = _split_semicolon_list(
        family_review["identifier_or_quasi_identifier_columns"]
    )

    if (family, column) in CONTROL_COLUMNS:
        return (
            "control_only",
            "Clave, fecha, partición u objetivo: conservar para control, nunca como predictor.",
            "No aplica; el control no se transforma en predictor.",
            "variable_catalog.csv; contrato de G1",
        )

    if identifier_risk.startswith(QUASI_IDENTIFIER_PREFIX) or column in identifier_columns:
        return (
            "exclude_predictor",
            "El catálogo o la revisión temporal la identifica como clave/cuasi-identificador o atributo de alta cardinalidad.",
            "No usar como predictor; preservar solo si se necesita trazabilidad.",
            "variable_catalog.csv; temporal_leakage_review.csv",
        )

    if family == "base":
        return (
            "exclude_predictor",
            "La columna de base no está definida como predictor en el contrato actual; no ampliar el conjunto predictor desde base.",
            "Conservar solo las columnas de control enumeradas por G1.",
            "variable_catalog.csv; contrato de G1",
        )

    if column == "riskassesment_302T":
        return (
            "exclude_predictor",
            "La definición describe una probabilidad de incumplimiento futuro, por lo que puede filtrar directamente el objetivo.",
            "Excluir por riesgo explícito de fuga del objetivo.",
            "variable_catalog.csv: descripción oficial de la columna",
        )

    if column in RISK_SCORE_COLUMNS:
        return (
            "manual_review",
            "Es un score de riesgo/creditworthiness; no se conoce su población de ajuste, fecha de cálculo ni disponibilidad al decidir.",
            "No usar hasta verificar procedencia y disponibilidad <= date_decision.",
            "variable_catalog.csv; temporal_leakage_review.csv",
        )

    if not description:
        return (
            "manual_review",
            "No hay descripción funcional en el catálogo para justificar su significado.",
            "No usar hasta documentar significado y disponibilidad temporal.",
            "variable_catalog.csv: descripción vacía",
        )

    if column in temporal_columns or column in review_columns:
        is_date = row["function"] == "temporal"
        temporal_rule = (
            "Ventana G1: all; validar la fecha contra date_decision y derivar solo una característica temporal justificable."
            if is_date
            else "Ventana G1: all; confirmar disponibilidad del valor <= date_decision."
        )
        return (
            "manual_review",
            "La revisión temporal marca la columna como fecha, estado, pago, mora, deuda o disponibilidad pendiente.",
            temporal_rule,
            "variable_catalog.csv; temporal_leakage_review.csv",
        )

    if REVIEW_AVAILABILITY in leakage_review or REVIEW_DATE in leakage_review:
        return (
            "manual_review",
            "El catálogo indica que se debe revisar la disponibilidad temporal.",
            "No usar hasta verificar disponibilidad <= date_decision; la ventana común es all.",
            "variable_catalog.csv",
        )

    if row["definition_status"] != "official":
        return (
            "manual_review",
            "La definición no está identificada como oficial; falta evidencia funcional para proponer su uso.",
            "No usar hasta documentar significado y disponibilidad <= date_decision.",
            "variable_catalog.csv: definition_status",
        )

    return (
        "provisional_candidate",
        "La definición oficial no tiene una señal automática de fuga en los reportes actuales; esto no prueba por sí solo su disponibilidad.",
        "Ventana G1: all; verificar disponibilidad <= date_decision antes de usar.",
        "variable_catalog.csv; temporal_leakage_review.csv",
    )


def select_depth0_columns(
    catalog: pl.DataFrame, temporal_review: pl.DataFrame
) -> pl.DataFrame:
    """Return one auditable B3 proposal row for every depth-zero catalog column."""
    _validate_columns(catalog, REQUIRED_CATALOG_COLUMNS, "variable catalog")
    review_by_family = _review_rows(temporal_review)
    selected = catalog.filter(pl.col("family").is_in(FAMILIES))
    if selected.is_empty():
        raise ValueError("variable catalog contains no depth-zero columns")

    duplicate_keys = selected.select(["family", "column"]).is_duplicated()
    if duplicate_keys.any():
        raise ValueError("variable catalog contains duplicate depth-zero family/column rows")

    rows: list[dict[str, str]] = []
    for row in selected.iter_rows(named=True):
        action, reason, temporal_rule, evidence = _classify(
            row, review_by_family[row["family"]]
        )
        rows.append(
            {
                "family": row["family"],
                "column": row["column"],
                "dtype": row["dtype"],
                "description": row["description"] or "",
                "proposed_action": action,
                "reason": reason,
                "temporal_rule": temporal_rule,
                "evidence": evidence,
            }
        )
    return pl.DataFrame(rows).select(OUTPUT_COLUMNS).sort(["family", "column"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", type=Path, default=Path("results/variable_catalog.csv")
    )
    parser.add_argument(
        "--temporal-review",
        type=Path,
        default=Path("results/integration_analysis/temporal_leakage_review.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/preparation/B3_depth0_columns.csv"),
    )
    args = parser.parse_args()
    result = select_depth0_columns(
        pl.read_csv(args.catalog), pl.read_csv(args.temporal_review)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.out)
    print(result.group_by("family", "proposed_action").len().sort("family", "proposed_action"))
    print(f"wrote {result.height} column decisions to {args.out}")


if __name__ == "__main__":
    main()
