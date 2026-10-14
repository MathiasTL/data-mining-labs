"""F7: ablación de los indicadores de cobertura de productos y registros (sin datos reales)."""

import numpy as np
import polars as pl

from dm_credit.preparation import f7_ablation as ab


def test_flag_columns_include_linear_outputs_of_the_families():
    cols = ["case_id", "tax_registry_a_1__n_records", "tax_registry_a_1__n_records__missing",
            "deposit_1__source_present__is__true", "applprev_1__n_records", "person_1__x"]
    assert ab.flag_columns(cols) == ["tax_registry_a_1__n_records", "tax_registry_a_1__n_records__missing", "deposit_1__source_present__is__true"]


def frame(n, weeks, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pl.DataFrame({
        "case_id": range(n), "WEEK_NUM": rng.choice(weeks, size=n), "date_decision": ["2020-01-01"] * n,
        "MONTH": [202001] * n, "target": y, "x": x,
        "deposit_1__n_records": rng.integers(0, 3, size=n).astype(float),
        "kind": pl.Series(rng.choice(["a", "b"], size=n)).cast(pl.Enum(["a", "b", "__UNKNOWN__"])),
    })


def test_compare_reports_both_variants_with_the_official_metric():
    out = ab.compare(frame(400, [1, 2, 3], 0), frame(300, [5, 6, 7], 1), model="logistic")
    assert out["variant"].to_list() == ["with_flags", "without_flags"]
    assert out["n_features"].to_list() == [3, 2]
    assert out["gini_stability"].is_finite().all() and (out["gini_oot"] > 0.5).all()
