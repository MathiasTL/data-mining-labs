"""Pruebas D7: vistas lineal y de árboles ajustadas solo con train (Giancarlo, P2).

Tablas sintéticas pequeñas con resultados calculados a mano. No leen Parquet reales.
"""

import json

import polars as pl
import pytest

from dm_credit.preparation import views as vw
from dm_credit.preparation.preprocessing import MISSING, UNKNOWN, PreprocessingSpec

pytestmark = pytest.mark.filterwarnings(
    "ignore::dm_credit.preparation.outliers.RangeWarning"
)

CONTROLS = ["case_id", "WEEK_NUM", "date_decision", "MONTH", "target"]
LINEAR_PREDICTORS = [
    "amt",
    "const",
    "amt__missing",
    "const__missing",
    "cat__is__x",
    "cat__is__y",
    f"cat__is__{MISSING}",
    f"cat__is__{UNKNOWN}",
    "flag__is__false",
    "flag__is__true",
    f"flag__is__{MISSING}",
    f"flag__is__{UNKNOWN}",
]
TREE_PREDICTORS = ["amt", "const", "cat", "flag"]


def spec() -> PreprocessingSpec:
    return PreprocessingSpec(
        numeric=("amt", "const"), categorical=("cat", "flag"), min_frequency=0.25
    )


def train_df() -> pl.DataFrame:
    # amt: mediana 3.0; imputada [1,2,3,4,5,3,3,3] -> media 3.0, desviación sqrt(10/8)
    # const: varianza cero; cat: x e y frecuentes (3/7), 'rare' rara (1/7 < 0.25)
    return pl.DataFrame(
        {
            "case_id": [8, 3, 5, 1, 7, 2, 6, 4],
            "WEEK_NUM": [70, 70, 71, 71, 72, 72, 73, 73],
            "date_decision": ["2020-05-01"] * 8,
            "MONTH": [202005] * 8,
            "target": [0, 1, 0, 0, 1, 0, 0, 0],
            "amt": [1.0, 2.0, 3.0, 4.0, 5.0, None, None, None],
            "const": [0.1] * 8,
            "cat": ["x", "x", "x", "y", "y", "y", "rare", None],
            "flag": [True, False, True, False, True, False, True, None],
        }
    )


def oot_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "case_id": [12, 10, 11],
            "WEEK_NUM": [74, 75, 80],
            "date_decision": ["2020-06-01"] * 3,
            "MONTH": [202006] * 3,
            "target": [1, 0, 0],
            "amt": [1003.0, None, 1001.0],
            "const": [0.1, 7.0, None],
            "cat": ["new", "rare", None],
            "flag": [True, None, False],
        }
    )


def unlabeled_df() -> pl.DataFrame:
    return oot_df().drop("target").with_columns(case_id=pl.col("case_id") + 100)


def fitted() -> vw.FittedMatrixViews:
    return vw.fit_matrix_views(train_df(), spec())


def splits() -> dict[str, pl.DataFrame]:
    return {"train": train_df(), "oot": oot_df(), "test": unlabeled_df()}


def test_both_views_keep_rows_and_case_order():
    state = fitted()
    for split, frame in splits().items():
        linear, tree = state.linear(frame, split), state.tree(frame, split)
        assert linear.height == tree.height == frame.height
        assert linear["case_id"].to_list() == frame["case_id"].to_list()
        assert tree["case_id"].to_list() == frame["case_id"].to_list()


def test_linear_scales_with_train_statistics_and_has_no_nulls_or_infinities():
    state = fitted()
    assert state.means["amt"] == pytest.approx(3.0)
    assert state.scales["amt"] == pytest.approx((10 / 8) ** 0.5)
    train = state.linear(train_df(), "train")
    assert train["amt"].to_list() == pytest.approx(
        [(v - 3.0) / (10 / 8) ** 0.5 for v in [1, 2, 3, 4, 5, 3, 3, 3]]
    )
    assert train["amt__missing"].to_list() == [0, 0, 0, 0, 0, 1, 1, 1]
    for split, frame in splits().items():
        predictors = state.linear(frame, split).select(LINEAR_PREDICTORS)
        assert sum(predictors.null_count().row(0)) == 0
        assert all(predictors[c].cast(pl.Float64).is_finite().all() for c in predictors.columns)


def test_linear_zero_variance_column_is_centered_without_division():
    state = fitted()
    assert state.means["const"] == 0.1 and state.scales["const"] == 1.0
    assert state.linear(train_df(), "train")["const"].to_list() == [0.0] * 8
    # OOT: 7.0 se centra con escala 1.0; el nulo se imputa con la mediana de train
    assert state.linear(oot_df(), "oot")["const"].to_list() == pytest.approx([0.0, 6.9, 0.0])


def test_transform_does_not_change_fitted_state():
    state = fitted()
    before = json.dumps(state.to_dict(), sort_keys=True)
    oot = state.linear(oot_df(), "oot")
    state.tree(oot_df(), "oot")
    state.linear(unlabeled_df(), "test")
    state.tree(unlabeled_df(), "test")
    assert json.dumps(state.to_dict(), sort_keys=True) == before
    # valores desplazados de OOT se escalan con la media y desviación de train
    assert oot["amt"].to_list() == pytest.approx(
        [(v - 3.0) / (10 / 8) ** 0.5 for v in [1003.0, 3.0, 1001.0]]
    )


def test_linear_one_hot_names_order_and_unknown_category():
    state = fitted()
    assert list(state.predictor_columns("linear")) == LINEAR_PREDICTORS
    oot = state.linear(oot_df(), "oot")
    assert oot.columns == [*CONTROLS, *LINEAR_PREDICTORS]  # 'new' no agrega columna
    # cat de OOT: 'new' (desconocida), 'rare' (rara en train), nulo
    assert oot[f"cat__is__{UNKNOWN}"].to_list() == [1, 1, 0]
    assert oot[f"cat__is__{MISSING}"].to_list() == [0, 0, 1]
    assert oot["cat__is__x"].to_list() == [0, 0, 0]
    assert oot["flag__is__true"].to_list() == [1, 0, 0]
    onehot = [c for c in LINEAR_PREDICTORS if c.startswith("cat__is__")]
    for split, frame in splits().items():
        assert state.linear(frame, split).select(pl.sum_horizontal(onehot)).to_series().to_list() == [1] * frame.height


def test_tree_keeps_numeric_nulls_and_native_categoricals():
    state = fitted()
    train = state.tree(train_df(), "train")
    assert train["amt"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0, None, None, None]
    assert train["cat"].cast(pl.String).to_list() == ["x", "x", "x", "y", "y", "y", UNKNOWN, None]
    oot = state.tree(oot_df(), "oot")
    assert oot["amt"].to_list() == [1003.0, None, 1001.0]  # sin imputar ni escalar
    assert oot["const"].to_list() == [0.1, 7.0, None]
    assert oot["cat"].cast(pl.String).to_list() == [UNKNOWN, UNKNOWN, None]
    assert oot["flag"].cast(pl.String).to_list() == ["true", None, "false"]


def test_tree_categoricals_share_frozen_dtype_across_splits():
    state = fitted()
    expected = {
        "cat": pl.Enum(["x", "y", UNKNOWN]),
        "flag": pl.Enum(["false", "true", UNKNOWN]),
    }
    for split, frame in splits().items():
        tree = state.tree(frame, split)
        assert {c: tree.schema[c] for c in expected} == expected  # 'new' no amplía el vocabulario
        assert tree.schema["amt"] == pl.Float64 and tree.schema["const"] == pl.Float64


def test_fit_rejects_non_train_weeks_and_clipping():
    base = train_df()
    with pytest.raises(ValueError, match="74"):
        vw.fit_matrix_views(base.with_columns(WEEK_NUM=pl.lit(74)), spec())
    with pytest.raises(ValueError, match="74"):
        vw.fit_matrix_views(pl.concat([base, oot_df()]), spec())
    with pytest.raises(ValueError, match="clip_columns"):
        vw.fit_matrix_views(
            base, PreprocessingSpec(numeric=("amt",), clip_columns=("amt",))
        )


def test_test_split_without_target_and_controls_pass_through():
    state = fitted()
    for view in vw.VIEWS:
        assert not set(state.predictor_columns(view)) & set(CONTROLS)
    for split, frame in splits().items():
        controls = [c for c in CONTROLS if c in frame.columns]
        for out in (state.linear(frame, split), state.tree(frame, split)):
            assert out.columns[: len(controls)] == controls
            assert out.select(controls).equals(frame.select(controls))
    assert "target" not in state.linear(unlabeled_df(), "test").columns
    assert "target" not in state.tree(unlabeled_df(), "test").columns


def test_controls_are_validated_per_split():
    state = fitted()
    with pytest.raises(ValueError, match="test must not contain target"):
        state.linear(oot_df(), "test")
    with pytest.raises(ValueError, match="target"):
        state.tree(unlabeled_df(), "oot")
    with pytest.raises(ValueError, match="control columns differ"):
        state.tree(oot_df().drop("MONTH"), "oot")
    with pytest.raises(ValueError, match="num_group1"):
        state.linear(oot_df().with_columns(num_group1=pl.lit(0)), "oot")
    with pytest.raises(ValueError, match="reserved"):
        vw.fit_matrix_views(train_df(), PreprocessingSpec(numeric=("amt", "MONTH")))
    with pytest.raises(ValueError, match="missing"):
        state.tree(oot_df().drop("cat"), "oot")


def test_undeclared_columns_are_rejected_instead_of_dropped():
    state = fitted()
    extra = pl.lit(9.0).alias("not_in_spec")
    with pytest.raises(ValueError, match="not_in_spec"):
        vw.fit_matrix_views(train_df().with_columns(extra), spec())
    with pytest.raises(ValueError, match="not_in_spec"):
        state.linear(oot_df().with_columns(extra), "oot")
    with pytest.raises(ValueError, match="not_in_spec"):
        state.tree(unlabeled_df().with_columns(extra), "test")


def test_predictors_share_columns_order_and_dtypes_across_splits():
    state = fitted()
    for view, expected in (("linear", LINEAR_PREDICTORS), ("tree", TREE_PREDICTORS)):
        schemas = []
        for split, frame in splits().items():
            out = getattr(state, view)(frame, split)
            predictors = [c for c in out.columns if c not in CONTROLS]
            assert predictors == expected == list(state.predictor_columns(view))
            schemas.append([out.schema[c] for c in predictors])
        assert schemas[0] == schemas[1] == schemas[2]


def test_state_roundtrips_through_json(tmp_path):
    state = fitted()
    path = vw.save_matrix_views(state, tmp_path / "views" / "matrix_views.json")
    loaded = vw.load_matrix_views(path)
    assert loaded.to_dict() == state.to_dict()
    assert loaded == vw.FittedMatrixViews.from_dict(state.to_dict())
    for split, frame in splits().items():
        assert loaded.linear(frame, split).equals(state.linear(frame, split))
        tree = loaded.tree(frame, split)
        assert tree.equals(state.tree(frame, split))
        assert tree.schema == state.tree(frame, split).schema
    with pytest.raises(ValueError, match="version"):
        vw.FittedMatrixViews.from_dict({**state.to_dict(), "version": 2})


def test_matrix_views_spec_covers_every_output_column_once(tmp_path):
    state = fitted()
    table = vw.matrix_views_spec(state)
    assert table.columns == list(vw.SPEC_SCHEMA)
    assert table.select(["view", "output_column"]).is_duplicated().sum() == 0
    sources = set(train_df().columns)
    for view in vw.VIEWS:
        rows = table.filter(pl.col("view") == view)
        out = getattr(state, view)(train_df(), "train")
        assert rows["output_column"].to_list() == out.columns
        assert set(rows["source_feature"].to_list()) <= sources
        predictors = rows.filter(pl.col("role") == "predictor")
        assert predictors["output_column"].to_list() == list(state.predictor_columns(view))
        assert set(predictors["source_feature"].to_list()) == set(spec().columns)
    learned = table.filter(pl.col("fitted_on") != "not fitted")
    assert learned["fitted_on"].unique().to_list() == ["train WEEK_NUM < 74"]
    by_output = {
        (r["view"], r["output_column"]): r for r in table.iter_rows(named=True)
    }
    assert by_output[("linear", "const")]["transformation"].endswith("zero_variance")
    assert json.loads(by_output[("linear", "amt")]["parameters"])["median"] == 3.0
    assert json.loads(by_output[("tree", "cat")]["parameters"])["categories"] == ["x", "y", UNKNOWN]
    assert by_output[("linear", f"cat__is__{UNKNOWN}")]["source_feature"] == "cat"

    path = vw.write_matrix_views_spec(state, tmp_path / "preparation" / "matrix_views.csv")
    assert pl.read_csv(path).equals(table)
