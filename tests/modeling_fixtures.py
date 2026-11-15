"""Matriz sintética pequeña con el contrato de la matriz E5 (sin Parquet real)."""

import numpy as np
import polars as pl

from dm_credit.preparation.preprocessing import PreprocessingSpec


def synthetic_matrix(per_week: int = 60, weeks: tuple[int, int] = (0, 91), seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    w = np.repeat(np.arange(weeks[0], weeks[1] + 1), per_week)
    n = len(w)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    logit = -2.0 + 1.5 * x1 + 0.8 * (cat == "a")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    x2 = np.where(rng.random(n) < 0.1, np.nan, x2)
    return pl.DataFrame(
        {
            "case_id": np.arange(n),
            "date_decision": ["2020-01-01"] * n,
            "MONTH": [202001] * n,
            "WEEK_NUM": w,
            "target": y,
            "x1": x1,
            "x2": x2,
            "cat": cat,
            "person_2__n_records": rng.integers(0, 3, n),
            "person_2__source_present": rng.random(n) < 0.5,
            "applprev_1__source_present": rng.random(n) < 0.6,
            "credit_bureau_a_1__source_present": rng.random(n) < 0.6,
            "credit_bureau_b_1__source_present": rng.random(n) < 0.3,
            "tax_registry_b_1__n_records": np.where(w < 60, 0, rng.integers(0, 3, n)),  # constante en todo ajuste (como E5-v2)
            "person_1__n_contact_persons__all": rng.integers(0, 4, n),
        }
    ).with_columns(pl.col("x2").fill_nan(None))


def synthetic_spec() -> PreprocessingSpec:
    return PreprocessingSpec(
        numeric=("x1", "x2", "person_2__n_records", "tax_registry_b_1__n_records", "person_1__n_contact_persons__all"),
        categorical=(
            "cat",
            "person_2__source_present",
            "applprev_1__source_present",
            "credit_bureau_a_1__source_present",
            "credit_bureau_b_1__source_present",
        ),
        min_frequency=0.01,
    )
