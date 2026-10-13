"""Fixtures sintéticas padre-hijo para las familias de solicitudes previas y personas (tarea A4).

Seis casos con fecha de decisión 2020-06-01. Cada caso cubre una situación que los
adaptadores de `applprev_*` y `person_*` deben resolver; los resultados esperados se
calculan a mano y están como constantes para poder usarlos en pruebas de caja negra.
"""

from __future__ import annotations

import polars as pl

DECISION = "2020-06-01"

# Caso 1: tres solicitudes válidas, la primera con dos hijos.
# Caso 2: una válida y una creada DESPUÉS de la decisión (debe descartarse con su hijo).
# Caso 3: sin historial de solicitudes previas (solo personas).
# Caso 4: una solicitud creada el mismo día de la decisión (se conserva, <=) y otra con
#         fecha nula (no se puede demostrar disponibilidad: se descarta con su hijo).
# Caso 5: solo una solicitud posterior a la decisión: el caso pierde toda cobertura.
# Caso 6: sin personas ni solicitudes.
CASE_IDS = [1, 2, 3, 4, 5, 6]

# Padres de applprev_1 temporalmente válidos (creationdate_885D <= date_decision).
VALID_APPLPREV_PARENTS = {(1, 0), (1, 1), (1, 2), (2, 0), (4, 0)}
# Hijos de applprev_2 cuyo padre es válido.
VALID_APPLPREV_CHILDREN = {(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 2, 0), (2, 0, 0), (4, 0, 0)}
INVALID_APPLPREV_CHILDREN = {(2, 1, 0), (4, 1, 0), (5, 0, 0)}


def base() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "case_id": CASE_IDS,
            "date_decision": [DECISION] * 6,
            "MONTH": [202006] * 6,
            "WEEK_NUM": [10, 10, 11, 11, 12, 12],
            "target": [0, 1, 0, 0, 1, 0],
        }
    )


def applprev_1() -> pl.DataFrame:
    rows = [
        (1, 0, "2020-01-10", 1000.0, "D", "CAL", "2020-01-15", True),
        (1, 1, "2020-03-01", 2000.0, "K", "REL", "2020-06-20", False),  # aprobación posterior: se anula
        (1, 2, "2020-05-31", 3000.0, "D", "CAL", None, None),
        (2, 0, "2020-02-01", 500.0, "D", "CAL", "2020-02-10", True),
        (2, 1, "2020-06-10", 700.0, "K", "REL", "2020-06-12", False),
        (4, 0, "2020-06-01", 400.0, "D", "REL", None, False),
        (4, 1, None, 900.0, "K", "CAL", "2020-01-01", True),
        (5, 0, "2020-06-15", 800.0, "D", "CAL", "2020-06-20", True),
    ]
    return pl.DataFrame(
        rows,
        schema={
            "case_id": pl.Int64,
            "num_group1": pl.Int64,
            "creationdate_885D": pl.String,
            "credamount_590A": pl.Float64,
            "status_219L": pl.String,
            "credtype_587L": pl.String,
            "approvaldate_319D": pl.String,
            "isbidproduct_390L": pl.Boolean,
        },
        orient="row",
    )


def applprev_2() -> pl.DataFrame:
    rows = [
        (1, 0, 0, "PHONE"),
        (1, 0, 1, "EMAIL"),
        (1, 1, 0, "PHONE"),
        (1, 2, 0, "PHONE"),
        (2, 0, 0, "EMAIL"),
        (2, 1, 0, "PHONE"),
        (4, 0, 0, "PHONE"),
        (4, 1, 0, "EMAIL"),
        (5, 0, 0, "PHONE"),
    ]
    return pl.DataFrame(
        rows,
        schema={
            "case_id": pl.Int64,
            "num_group1": pl.Int64,
            "num_group2": pl.Int64,
            "conts_type_509L": pl.String,
        },
        orient="row",
    )


def person_1() -> pl.DataFrame:
    rows = [
        (1, 0, "M", 1000.0, "2019-01-01", "1985-04-12", "AA"),
        (1, 1, "F", None, None, None, "BB"),
        (2, 0, "M", 800.0, "2020-07-01", "1990-01-01", "AA"),  # inicio de empleo posterior a la decisión
        (3, 0, "F", 1200.0, "2018-05-05", "1975-09-30", "AA"),
        (4, 0, "M", 900.0, None, "1980-02-02", "AA"),
        (5, 0, "F", 700.0, "2017-03-03", "1995-07-07", "AA"),
    ]
    return pl.DataFrame(
        rows,
        schema={
            "case_id": pl.Int64,
            "num_group1": pl.Int64,
            "sex_738L": pl.String,
            "mainoccupationinc_384A": pl.Float64,
            "empl_employedfrom_271D": pl.String,
            "birth_259D": pl.String,
            "role_1084L": pl.String,
        },
        orient="row",
    )


def person_2() -> pl.DataFrame:
    rows = [(1, 0, 0, "X"), (1, 0, 1, "Y"), (3, 0, 0, "X")]
    return pl.DataFrame(
        rows,
        schema={
            "case_id": pl.Int64,
            "num_group1": pl.Int64,
            "num_group2": pl.Int64,
            "empls_economicalst_849M": pl.String,
        },
        orient="row",
    )
