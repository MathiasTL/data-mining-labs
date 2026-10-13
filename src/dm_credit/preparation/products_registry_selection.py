"""B6 (borrador) — Selección de columnas de productos y registros fiscales.

Cada columna de ``debitcard_1``, ``deposit_1``, ``other_1``, ``tax_registry_a_1``,
``tax_registry_b_1`` y ``tax_registry_c_1`` recibe una acción conservadora.
Es una propuesta que el equipo debe aprobar en G2: ningún predictor se declara
seguro solo por su nombre (ver ``development/02-comprension-datos.md``).

Acciones (las mismas de B4 para que G2 compare igual):
    excluir            claves técnicas, cuasi-identificadores y nombres de alta cardinalidad
    bloquear_hasta_G2  importes sin fecha propia ni ventana documentada, u ``other_1``
                       completo (sin referencia temporal en la familia)
    transformar_as_of  fecha utilizable solo como antigüedad/recencia si es <= date_decision
    candidato_G2       importe con ventana source_defined en el nombre; entra al catálogo si G2 lo aprueba

Uso:
    uv run -m dm_credit.preparation.products_registry_selection --out results/preparation/B6_products_registry_columns.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

FAMILIES = ("debitcard_1", "deposit_1", "other_1", "tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1")
RESERVED_KEYS = {"case_id", "num_group1", "num_group2"}
QUASI_IDENTIFIER_PREFIX = "posible cuasi"
NO_TEMPORAL_REFERENCE = "sin referencia temporal explícita"
OTHER_NO_TEMPORAL = (
    "familia sin referencia temporal: disponibilidad indemostrable "
    "(insufficient_temporal_evidence); veredicto en G2, exclusión con evidencia en C6 si no se confirma"
)


def _decide(r: dict) -> tuple[str, str, str]:
    family, column = r["family"], r["column"]
    if column in RESERVED_KEYS:
        return "excluir", "clave o índice estructural: se conserva solo para trazabilidad", ""
    if str(r["identifier_risk"]).startswith(QUASI_IDENTIFIER_PREFIX):
        return "excluir", "cuasi-identificador o alta cardinalidad (nombre fiscal/empleador): no usar como categoría libre", ""
    if family == "other_1":
        return "bloquear_hasta_G2", OTHER_NO_TEMPORAL, ""
    if r["function"] == "temporal":
        return (
            "transformar_as_of",
            "usar solo como antigüedad o recencia respecto de date_decision; un valor posterior se anula",
            "",
        )
    if r["temporal_relevance"] != NO_TEMPORAL_REFERENCE:
        return (
            "candidato_G2",
            "ventana source_defined en el nombre; confirmar disponibilidad del resumen en G2",
            "",
        )
    return (
        "bloquear_hasta_G2",
        "importe sin fecha propia ni ventana documentada: confirmar disponibilidad en G2",
        "",
    )


def select_columns(catalog: pl.DataFrame) -> pl.DataFrame:
    """Aplica las reglas al catálogo de variables y devuelve una fila por columna."""
    bad = sorted(set(catalog["family"]) - set(FAMILIES))
    if bad:
        raise ValueError(f"unsupported family: {bad}; expected one of {list(FAMILIES)}")
    rows = []
    for r in catalog.iter_rows(named=True):
        action, reason, note = _decide(r)
        rows.append(
            {
                "family": r["family"],
                "column": r["column"],
                "dtype": r["dtype"],
                "action": action,
                "reason": reason,
                "note": note,
            }
        )
    return pl.DataFrame(rows).sort(["family", "column"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", type=Path, default=Path("results/variable_catalog.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/preparation/B6_products_registry_columns.csv"))
    args = ap.parse_args()
    catalog = pl.read_csv(args.catalog).filter(pl.col("family").is_in(FAMILIES))
    out = select_columns(catalog)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out)
    print(out.group_by("action").len().sort("action"))
    print(f"escrito en {args.out}")


if __name__ == "__main__":
    main()
