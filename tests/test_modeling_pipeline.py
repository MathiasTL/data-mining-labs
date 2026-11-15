"""Pruebas del pipeline por partición: ajuste solo con el bloque de ajuste, sin controles en X."""

import numpy as np
import polars as pl
import pytest

from dm_credit.modeling import folds as fd
from dm_credit.modeling import pipeline as pp

from modeling_fixtures import synthetic_matrix, synthetic_spec

pytestmark = pytest.mark.filterwarnings("ignore")


def test_state_depends_only_on_fit_block():
    df = synthetic_matrix()
    f = fd.get_fold("P1")
    a = pp.prepare_fold(df, synthetic_spec(), f)
    # cambiar drásticamente las filas de validación (y OOT) no altera el estado ajustado
    w = df["WEEK_NUM"]
    mutated = df.with_columns(
        pl.when(w > 39).then(pl.col("x1") * 1000 + 5).otherwise(pl.col("x1")).alias("x1"),
        pl.when(w > 39).then(pl.lit("zzz")).otherwise(pl.col("cat")).alias("cat"),
    )
    b = pp.prepare_fold(mutated, synthetic_spec(), f)
    assert a.views.to_dict() == b.views.to_dict()


def test_state_changes_when_fit_block_changes():
    df = synthetic_matrix()
    f = fd.get_fold("P1")
    a = pp.prepare_fold(df, synthetic_spec(), f)
    mutated = df.with_columns(
        pl.when(pl.col("WEEK_NUM") <= 39).then(pl.col("x1") + 10).otherwise(pl.col("x1")).alias("x1")
    )
    b = pp.prepare_fold(mutated, synthetic_spec(), f)
    assert a.views.means["x1"] != b.views.means["x1"]


def test_fit_and_val_blocks_have_exact_weeks_and_no_controls_in_x():
    df = synthetic_matrix()
    f = fd.get_fold("P2")
    ctx = pp.prepare_fold(df, synthetic_spec(), f)
    assert ctx.fit_weeks.min() == 0 and ctx.fit_weeks.max() == 49
    assert ctx.val_weeks.min() == 50 and ctx.val_weeks.max() == 59
    for view in ("linear", "tree"):
        X, names, _ = ctx.matrix(view, "fit")
        assert X.shape == (len(ctx.y_fit), len(names))
        assert not set(names) & set(pp.CONTROLS)
        assert X.dtype == np.float32


def test_tree_view_marks_categoricals_and_keeps_nan():
    ctx = pp.prepare_fold(synthetic_matrix(), synthetic_spec(), fd.get_fold("P1"))
    X, names, cat_idx = ctx.matrix("tree", "fit")
    assert "cat" in [names[i] for i in cat_idx]
    assert np.isnan(X[:, names.index("x2")]).any()


def test_oot_only_when_requested():
    df = synthetic_matrix()
    ctx = pp.prepare_fold(df, synthetic_spec(), fd.get_fold("P3"))
    with pytest.raises(ValueError):
        ctx.matrix("linear", "oot")
    oot = df.filter(pl.col("WEEK_NUM") >= 74)
    ctx2 = pp.prepare_fold(df, synthetic_spec(), fd.get_fold("P3"), oot=oot)
    X, _, _ = ctx2.matrix("linear", "oot")
    assert X.shape[0] == oot.height


def test_oot_frame_with_development_weeks_is_rejected():
    df = synthetic_matrix()
    leaky = df.filter(pl.col("WEEK_NUM") >= 73)  # incluye la semana 73 (desarrollo)
    with pytest.raises(ValueError, match="development weeks"):
        pp.prepare_fold(df, synthetic_spec(), fd.get_fold("P3"), oot=leaky)


def test_val_block_never_reaches_oot_weeks_even_if_frame_has_them():
    ctx = pp.prepare_fold(synthetic_matrix(), synthetic_spec(), fd.get_fold("P3"))
    assert ctx.val_weeks.max() == 73


def test_exclude_groups_filters_columns_and_spec():
    df = synthetic_matrix()
    spec = synthetic_spec()
    df2, spec2, dropped = pp.exclude_columns(df, spec, ["person_2__*"])
    assert set(dropped) == {"person_2__n_records", "person_2__source_present"}
    assert "person_2__n_records" not in df2.columns
    assert "person_2__n_records" not in spec2.columns
    # grupo ausente: no falla
    _, _, none = pp.exclude_columns(df, spec, ["nope__*"])
    assert none == ()


def test_spec_from_schema_uses_logical_types():
    rows = [
        {"name": "a", "logical_type": "int"},
        {"name": "b", "logical_type": "float"},
        {"name": "c", "logical_type": "bool"},
        {"name": "d", "logical_type": "string"},
    ]
    spec = pp.spec_from_schema(rows, available=["a", "b", "c"])
    assert spec.numeric == ("a", "b") and spec.categorical == ("c",)
