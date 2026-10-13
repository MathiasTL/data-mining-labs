"""Pruebas D1-D3: preprocesador ajustado solo con train (Gabriel, P5)."""

import json

import polars as pl
import pytest

from dm_credit.preparation import preprocessing as pp
from dm_credit.preparation.outliers import RangeWarning


def spec() -> pp.PreprocessingSpec:
    return pp.PreprocessingSpec(numeric=("a",), categorical=("c",), min_frequency=0.25)


def train_df() -> pl.DataFrame:
    # a: mediana 3.0, rango [1, 5]; c: x e y frecuentes (2/5), 'rare' rara (1/5 < 0.25)
    return pl.DataFrame(
        {
            "WEEK_NUM": [70, 71, 72, 73, 70, 71],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, None],
            "c": ["x", "x", "y", "y", "rare", None],
        }
    )


def test_fit_learns_median_vocab_and_range_from_train_only():
    fitted = pp.fit_preprocessor(train_df(), spec())
    assert fitted.medians == {"a": 3.0}
    assert fitted.vocabs == {"c": ("x", "y")}
    assert fitted.ranges.train_range == {"a": (1.0, 5.0)}


def test_fit_rejects_oot_null_or_missing_week():
    base = train_df()
    with pytest.raises(ValueError, match="74"):
        pp.fit_preprocessor(base.with_columns(WEEK_NUM=pl.lit(74)), spec())
    with pytest.raises(ValueError, match="null"):
        pp.fit_preprocessor(
            base.with_columns(
                WEEK_NUM=pl.when(pl.col("WEEK_NUM") == 70).then(None).otherwise(pl.col("WEEK_NUM"))
            ),
            spec(),
        )
    with pytest.raises(ValueError, match="WEEK_NUM"):
        pp.fit_preprocessor(base.drop("WEEK_NUM"), spec())


def test_transform_imputes_maps_and_marks_missing():
    fitted = pp.fit_preprocessor(train_df(), spec())
    out = fitted.transform(pl.DataFrame({"a": [None, 2.0], "c": ["x", None]}))
    assert out.columns == ["a", "c", "a__missing", "c__missing"]
    assert out["a"].to_list() == [3.0, 2.0]
    assert out["c"].to_list() == ["x", pp.MISSING]
    assert out["a__missing"].to_list() == [1, 0]
    assert out["c__missing"].to_list() == [0, 1]


def test_unknown_and_rare_become_unknown_without_changing_state():
    fitted = pp.fit_preprocessor(train_df(), spec())
    before = fitted.to_dict()
    out = fitted.transform(pl.DataFrame({"a": [1.0, 1.0], "c": ["new", "rare"]}))
    assert out["c"].to_list() == [pp.UNKNOWN, pp.UNKNOWN]
    assert fitted.to_dict() == before  # transform no aprende


def test_train_oot_test_share_schema_order_and_types():
    fitted = pp.fit_preprocessor(train_df(), spec())
    in_range = fitted.transform(pl.DataFrame({"a": [1.0], "c": ["x"]}))
    no_target = fitted.transform(pl.DataFrame({"a": [None], "c": [None]}))  # test sin target
    with pytest.warns(RangeWarning):
        oot = fitted.transform(pl.DataFrame({"a": [99.0], "c": ["zzz"]}))  # OOT: fuera de rango + desconocida
    for frame in (in_range, oot, no_target):
        assert frame.columns == ["a", "c", "a__missing", "c__missing"]
        assert frame.schema["a"] == pl.Float64 and frame.schema["c"] == pl.String
    assert oot["c"].to_list() == [pp.UNKNOWN]


def test_out_of_range_warns_without_modifying():
    fitted = pp.fit_preprocessor(train_df(), spec())
    with pytest.warns(RangeWarning, match="a"):
        out = fitted.transform(pl.DataFrame({"a": [0.0, 500.0], "c": ["x", "y"]}))
    assert out["a"].to_list() == [0.0, 500.0]


def test_clip_only_for_approved_columns():
    train = pl.DataFrame({"WEEK_NUM": [70] * 100, "a": [float(i) for i in range(1, 101)], "c": ["x"] * 100})
    fitted = pp.fit_preprocessor(
        train, pp.PreprocessingSpec(numeric=("a",), categorical=("c",), clip_columns=("a",), clip_quantiles=(0.05, 0.95))
    )
    with pytest.warns(RangeWarning):
        out = fitted.transform(pl.DataFrame({"a": [-1000.0, 50.0, 1000.0], "c": ["x", "y", "new"]}))
    lo, hi = fitted.ranges.clip_bounds["a"]
    assert out["a"].to_list() == pytest.approx([lo, 50.0, hi])


def test_infinite_and_non_convertible_are_errors():
    fitted = pp.fit_preprocessor(train_df(), spec())
    with pytest.raises(ValueError, match="inf"):
        fitted.transform(pl.DataFrame({"a": [float("inf")], "c": ["x"]}))
    with pytest.raises(ValueError, match="a"):
        fitted.transform(pl.DataFrame({"a": ["abc"], "c": ["x"]}))
    with pytest.raises(ValueError, match="missing"):
        fitted.transform(pl.DataFrame({"a": [1.0]}))


def test_state_roundtrips_through_json(tmp_path):
    fitted = pp.fit_preprocessor(train_df(), spec())
    path = pp.save_preprocessor(fitted, tmp_path / "prep.json")
    loaded = pp.load_preprocessor(path)
    assert loaded.to_dict() == fitted.to_dict()
    assert json.loads(path.read_text())["medians"] == {"a": 3.0}
    sample = pl.DataFrame({"a": [None], "c": ["new"]})
    assert loaded.transform(sample).equals(fitted.transform(sample))


def test_spec_rejects_double_roles_and_bad_params():
    with pytest.raises(ValueError, match="two roles"):
        pp.PreprocessingSpec(numeric=("a",), categorical=("a",))
    with pytest.raises(ValueError, match="clip_columns"):
        pp.PreprocessingSpec(numeric=("a",), clip_columns=("zzz",))
    with pytest.raises(ValueError, match="date"):
        pp.build_spec_from_catalog(
            pl.DataFrame({"name": ["d"], "logical_type": ["date"]})
        )


def test_spec_from_real_c8_catalog_smoke():
    catalog = pl.read_csv("results/preparation/C8_feature_catalog.csv").filter(
        pl.col("role") == "predictor"
    )
    built = pp.build_spec_from_catalog(catalog)
    assert len(built.numeric) + len(built.categorical) == 94
    assert not set(built.numeric) & set(built.categorical)
    # marco sintético con los tipos del catálogo: fit + transform no fallan
    frame = pl.DataFrame(
        {c: ([1.0, 2.0] if t in ("int", "float") else ["k", "k"]) for c, t in zip(catalog["name"].to_list(), catalog["logical_type"].to_list())}
    ).with_columns(WEEK_NUM=pl.lit(70))
    fitted = pp.fit_preprocessor(frame, built)
    out = fitted.transform(frame.drop("WEEK_NUM"))
    assert out.columns == [*built.columns, *(f"{c}__missing" for c in built.columns)]
    assert out.height == 2
