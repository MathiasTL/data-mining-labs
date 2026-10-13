"""B4 (borrador) — Selección de columnas de solicitudes previas y personas.

Cada columna de ``applprev_1``, ``applprev_2``, ``person_1`` y ``person_2`` recibe una acción
conservadora. Es una propuesta que el equipo debe aprobar en G2: ningún predictor se declara
seguro solo por su nombre (ver ``development/02-comprension-datos.md``).

Acciones:
    excluir            claves técnicas, cuasi-identificadores y estados sin trazabilidad
    bloquear_hasta_G2  estados, pagos, mora, deuda o campos sin fecha: no entran hasta confirmar
    referencia_as_of   fecha que decide si una fila existía al momento de la decisión
    transformar_as_of  fecha utilizable solo como antigüedad/recencia si es <= date_decision
    candidato_G2       atributo sin señal automática de fuga; entra al catálogo si G2 lo aprueba

Uso:
    uv run -m dm_credit.preparation.applications_people_selection --out results/preparation/B4_applications_people_columns.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

FAMILIES = ("applprev_1", "applprev_2", "person_1", "person_2")
RESERVED_KEYS = {"case_id", "num_group1", "num_group2"}
AS_OF_REFERENCE = {("applprev_1", "creationdate_885D")}
SENSITIVE = {"gender_992L", "sex_738L"}
QUASI_IDENTIFIER_PREFIX = "posible cuasi"
REVIEW_AVAILABILITY = "revisar disponibilidad al decidir"
NO_OWN_DATE = "sin referencia temporal explícita"


def _decide(r: dict) -> tuple[str, str, str]:
    family, column = r["family"], r["column"]
    if column in RESERVED_KEYS:
        return "excluir", "clave o índice estructural: se conserva solo para trazabilidad", ""
    if str(r["identifier_risk"]).startswith(QUASI_IDENTIFIER_PREFIX):
        if r["dtype"] == "Boolean":
            return (
                "bloquear_hasta_G2",
                "marcada como cuasi-identificador por el prefijo de dirección, pero es un indicador booleano",
                "revisar si aporta señal sin riesgo de identificación",
            )
        return "excluir", "cuasi-identificador o alta cardinalidad (dirección, distrito, código postal, empleador)", ""
    if (family, column) in AS_OF_REFERENCE:
        return (
            "referencia_as_of",
            "fecha de creación de la solicitud previa: filtra las filas con fecha <= date_decision",
            "una fecha nula no demuestra disponibilidad: la fila se descarta",
        )
    if family == "applprev_2":
        if REVIEW_AVAILABILITY in str(r["leakage_review"]):
            return "excluir", "estado sin referencia temporal de fila ni linaje confirmado", ""
        return (
            "bloquear_hasta_G2",
            "sin fecha propia: solo se admite agregando a través de un padre válido de applprev_1",
            "",
        )
    if r["function"] == "temporal":
        return (
            "transformar_as_of",
            "usar solo como antigüedad o recencia respecto de date_decision; un valor posterior se anula",
            "",
        )
    if REVIEW_AVAILABILITY in str(r["leakage_review"]):
        return (
            "bloquear_hasta_G2",
            "estado, pago, mora, deuda o motivo que pudo actualizarse después de la decisión",
            "",
        )
    if family == "applprev_1" and r["temporal_relevance"] == NO_OWN_DATE:
        return (
            "bloquear_hasta_G2",
            "estado de una solicitud previa sin fecha propia: puede reflejar información posterior a su creación",
            "",
        )
    note = "atributo sensible: decisión del equipo (equidad y regulación)" if column in SENSITIVE else ""
    return "candidato_G2", "sin señal automática de fuga; confirmar su disponibilidad en G2", note


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
    ap.add_argument("--out", type=Path, default=Path("results/preparation/B4_applications_people_columns.csv"))
    args = ap.parse_args()
    catalog = pl.read_csv(args.catalog).filter(pl.col("family").is_in(FAMILIES))
    out = select_columns(catalog)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out)
    print(out.group_by("action").len().sort("action"))
    print(f"escrito en {args.out}")


if __name__ == "__main__":
    main()
