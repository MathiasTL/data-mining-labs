"""B5 (borrador) — Selección de columnas de bureau de crédito.

Cada columna de ``credit_bureau_a_1``, ``credit_bureau_a_2``, ``credit_bureau_b_1`` y
``credit_bureau_b_2`` recibe una acción conservadora y una regla ``as-of`` explícita respecto de
``date_decision``. Es una propuesta que el equipo debe aprobar en G2: ningún predictor se declara
seguro solo por su nombre (ver ``context/03-preparacion-A5-bureau-streaming.md`` §3.4 y G2 §1).

Acciones (las mismas de B4 y B6 para que G2 compare igual):
    excluir            claves técnicas, nombres de entidades y cobertura < 5 % (regla 4 de G2)
    bloquear_hasta_G2  estados, pagos, mora y deuda (regla 2 de G2), fechas de fin pactadas
                       (calendario, regla 3 de G2) y todo ``credit_bureau_a_2`` (puntos D1 y D6 de A5)
    referencia_as_of   fecha que decide si una fila existía al momento de la decisión (ancla)
    transformar_as_of  fecha utilizable solo como antigüedad/recencia si es <= date_decision
    candidato_G2       condición del contrato sin señal de fuga; entra al catálogo si G2 lo aprueba

Anclas temporales (evidencia: tasas de nulos de ``results/variable_catalog.csv``):
    credit_bureau_a_1  ``lastupdate_1112D`` (contratos activos, 83,3 % nulo) y ``lastupdate_388D``
                       (contratos cerrados, 57,6 % nulo): fecha de la instantánea del contrato
    credit_bureau_a_2  pares año/mes de pago, emparejados por tasa de nulos idéntica; bloqueado (D1, D6)
    credit_bureau_b_1  ``lastupdate_260D``: fecha de la instantánea del contrato
    credit_bureau_b_2  ``pmts_date_1107D``: fecha del pago, con padre válido en ``credit_bureau_b_1``

Uso:
    uv run -m dm_credit.preparation.bureau_selection --out results/preparation/B5_bureau_columns.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

from dm_credit.preparation.families.applications_people_g2 import MIN_COVERAGE

FAMILIES = ("credit_bureau_a_1", "credit_bureau_a_2", "credit_bureau_b_1", "credit_bureau_b_2")
RESERVED_KEYS = {"case_id", "num_group1", "num_group2"}
QUASI_IDENTIFIER_PREFIX = "posible cuasi"
REVIEW_AVAILABILITY = "revisar disponibilidad al decidir"
DELINQUENCY_FUNCTION = "mora/incumplimiento histórico"
OUTPUT_COLUMNS = ["family", "column", "dtype", "action", "reason", "temporal_rule", "evidence", "note"]

AS_OF_REFERENCE = {
    ("credit_bureau_a_1", "lastupdate_1112D"): "contratos activos",
    ("credit_bureau_a_1", "lastupdate_388D"): "contratos cerrados",
    ("credit_bureau_b_1", "lastupdate_260D"): "contratos de bureau B",
    ("credit_bureau_b_2", "pmts_date_1107D"): "pagos de bureau B",
}
# Nombre de acreedor sin marca automática: mismo trato que financialinstitution_* (cuasi-identificador).
CREDITOR_NAMES = {("credit_bureau_b_1", "credor_3940957M")}
# Marcadas como cuasi-identificador solo por contener "name": son roles con 9 a 11 categorías.
ROLE_NOT_IDENTIFIER = {("credit_bureau_a_2", "subjectroles_name_541M"), ("credit_bureau_a_2", "subjectroles_name_838M")}
# Fechas de fin pactadas: para un contrato activo caen después de la decisión por construcción.
CALENDAR_DATES = {
    ("credit_bureau_a_1", "dateofcredend_289D"),
    ("credit_bureau_a_1", "dateofcredend_353D"),
    ("credit_bureau_b_1", "contractmaturitydate_151D"),
}
# Saldos, cuotas pendientes, conteos vigentes y prórrogas: estados que cambian durante la vida del contrato.
STATE_STEMS = {
    "outstandingamount",
    "residualamount",
    "numberofoutstandinstls",
    "numberofcontrsvalue",
    "credquantity",
    "prolongationcount",
    "pmtnumpending",
}
_SUFFIX = re.compile(r"_\d+[A-Z]$")

ROW_RULE = {
    "credit_bureau_a_1": (
        "el valor se usa solo si el ancla de su lado del contrato (lastupdate_1112D activo, "
        "lastupdate_388D cerrado) es <= date_decision; ancla nula: el valor se descarta"
    ),
    "credit_bureau_a_2": (
        "propuesta: el mes del pago (pmts_year_1139T + pmts_month_158T, o pmts_year_507T + pmts_month_706T) "
        "es estrictamente anterior al mes de date_decision y el padre (case_id, num_group1) es válido en "
        "credit_bureau_a_1; año o mes nulo: la fila se descarta"
    ),
    "credit_bureau_b_1": "la fila se usa solo si lastupdate_260D <= date_decision; fecha nula: la fila se descarta",
    "credit_bureau_b_2": (
        "la fila se usa solo si pmts_date_1107D <= date_decision y su padre (case_id, num_group1) es válido "
        "en credit_bureau_b_1; fecha nula: la fila se descarta"
    ),
}
NOT_APPLICABLE = "no aplica: la columna no se convierte en predictor"


def _stem(column: str) -> str:
    return _SUFFIX.sub("", column)


def _coverage(r: dict) -> float | None:
    pct_null = r.get("pct_null")
    return None if pct_null is None else 1.0 - float(pct_null) / 100.0


def _evidence(r: dict) -> str:
    parts = [
        f"function={r['function']}",
        f"leakage_review={r['leakage_review']}",
        f"pct_null={r.get('pct_null')}",
    ]
    if r.get("n_unique") is not None:
        parts.append(f"n_unique={r['n_unique']}")
    return "variable_catalog.csv: " + "; ".join(parts)


def _decide(r: dict) -> tuple[str, str, str, str]:
    """Devuelve ``(acción, motivo, regla temporal, nota)`` para una fila del catálogo."""
    family, column = r["family"], r["column"]
    key = (family, column)
    row_rule = ROW_RULE[family]
    if column in RESERVED_KEYS:
        return "excluir", "clave o índice estructural: se conserva solo para trazabilidad", NOT_APPLICABLE, ""
    if key in CREDITOR_NAMES or (
        str(r["identifier_risk"]).startswith(QUASI_IDENTIFIER_PREFIX) and key not in ROLE_NOT_IDENTIFIER
    ):
        return (
            "excluir",
            "cuasi-identificador o alta cardinalidad (nombre de entidad financiera o acreedor): no usar como categoría libre",
            NOT_APPLICABLE,
            "",
        )
    if key in AS_OF_REFERENCE:
        return (
            "referencia_as_of",
            f"ancla temporal de {AS_OF_REFERENCE[key]}: filtra las filas con fecha <= date_decision",
            row_rule + "; una fecha igual a date_decision se conserva",
            "una fecha nula no demuestra disponibilidad: la fila se descarta y se cuenta en el manifiesto",
        )
    coverage = _coverage(r)
    if coverage is not None and coverage < MIN_COVERAGE:
        return (
            "excluir",
            f"cobertura {coverage:.2%} < {MIN_COVERAGE:.0%} (regla 4 de G2): sin información suficiente en el primer corte",
            NOT_APPLICABLE,
            "",
        )
    if family == "credit_bureau_a_2":
        note = (
            "D6: pmts_year_1139T y pmts_month_158T comparten 64,8803 % de nulos, y pmts_year_507T y pmts_month_706T "
            "15,7033 %; el emparejamiento de A5 §2.6 (1139T+706T, 507T+158T) no coincide con los datos"
            if r["function"] != DELINQUENCY_FUNCTION and _stem(column).startswith(("pmts_year", "pmts_month"))
            else ""
        )
        return (
            "bloquear_hasta_G2",
            "sin fecha de día: granularidad mensual (D1) y emparejamiento año/mes activo/cerrado sin confirmar (D6); "
            "solo se admite agregando a través de un padre válido de credit_bureau_a_1",
            row_rule,
            note,
        )
    if key in CALENDAR_DATES:
        return (
            "bloquear_hasta_G2",
            "fecha de fin pactada: calendario o proyección, no un hecho ocurrido (regla 3 de G2)",
            row_rule + "; además, una fecha posterior a date_decision se anula y se cuenta en el manifiesto",
            "",
        )
    if r["function"] == "temporal" and REVIEW_AVAILABILITY not in str(r["leakage_review"]):
        if _stem(column).startswith(("numberofoverdueinstlmaxdat", "overdueamountmax2date")):
            return (
                "bloquear_hasta_G2",
                "fecha del máximo de mora: describe un estado de mora que pudo actualizarse después de la decisión",
                row_rule + "; además, una fecha posterior a date_decision se anula",
                "",
            )
        return (
            "transformar_as_of",
            "usar solo como antigüedad o recencia respecto de date_decision; un valor posterior se anula",
            row_rule + "; antigüedad = date_decision - fecha, y una fecha posterior se anula y se cuenta en el manifiesto",
            "",
        )
    if REVIEW_AVAILABILITY in str(r["leakage_review"]) or r["function"] == DELINQUENCY_FUNCTION:
        note = (
            "propuesta adicional: conservar el pago solo si pmts_date_1107D + días de mora <= date_decision"
            if family == "credit_bureau_b_2"
            else ""
        )
        return (
            "bloquear_hasta_G2",
            "estado, pago, mora o deuda que pudo actualizarse después de la decisión (regla 2 de G2)",
            row_rule,
            note,
        )
    if _stem(column) in STATE_STEMS:
        return (
            "bloquear_hasta_G2",
            "saldo, cuotas pendientes, conteo vigente o prórrogas: estado que cambia durante la vida del contrato",
            row_rule,
            "",
        )
    return "candidato_G2", "condición del contrato sin señal automática de fuga; confirmar su disponibilidad en G2", row_rule, ""


def select_columns(catalog: pl.DataFrame) -> pl.DataFrame:
    """Aplica las reglas al catálogo de variables y devuelve una fila por columna."""
    bad = sorted(set(catalog["family"]) - set(FAMILIES))
    if bad:
        raise ValueError(f"unsupported family: {bad}; expected one of {list(FAMILIES)}")
    rows = []
    for r in catalog.iter_rows(named=True):
        action, reason, temporal_rule, note = _decide(r)
        rows.append(
            {
                "family": r["family"],
                "column": r["column"],
                "dtype": r["dtype"],
                "action": action,
                "reason": reason,
                "temporal_rule": temporal_rule,
                "evidence": _evidence(r),
                "note": note,
            }
        )
    return pl.DataFrame(rows, schema={c: pl.String for c in OUTPUT_COLUMNS}).sort(["family", "column"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", type=Path, default=Path("results/variable_catalog.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/preparation/B5_bureau_columns.csv"))
    args = ap.parse_args()
    catalog = pl.read_csv(args.catalog).filter(pl.col("family").is_in(FAMILIES))
    out = select_columns(catalog)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out)
    print(out.group_by("family", "action").len().sort("family", "action"))
    print(f"escrito en {args.out}")


if __name__ == "__main__":
    main()
