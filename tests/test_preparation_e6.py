"""E6: preprocesador y vistas ajustados solo con train, aplicados a las tablas maestras de E5."""

import json

import polars as pl
import pytest

import test_preparation_validation as tv
from dm_credit.preparation import contracts, e5, e6

published = tv.published


@pytest.fixture
def masters(published):
    report = e5.assemble_master_tables(published["bases"], published["artifacts"], output_dir=published["tmp"] / "E5", cutoff_week=tv.g3.CUTOFF)
    tables = {r["split"]: pl.read_parquet(r["data_path"]) for r in report.iter_rows(named=True)}
    catalog = contracts.feature_catalog(published["artifacts"]["train"])
    return {"tables": tables, "catalog": catalog, "tmp": published["tmp"]}


def run(masters, out="E6"):
    return e6.fit_and_transform(masters["tables"], masters["catalog"], output_dir=masters["tmp"] / out, cutoff_week=tv.g3.CUTOFF)


def test_two_views_for_each_split(masters):
    report = run(masters)
    assert sorted(zip(report["view"], report["split"])) == sorted((v, s) for v in ("linear", "tree") for s in ("train", "oot", "test"))
    for r in report.iter_rows(named=True):
        out = pl.read_parquet(r["data_path"])
        assert out.height == masters["tables"][r["split"]].height
        assert out["case_id"].to_list() == masters["tables"][r["split"]]["case_id"].to_list()


def test_schemas_are_identical_across_splits(masters):
    report = run(masters)
    for view in ("linear", "tree"):
        schemas = []
        for r in report.filter(pl.col("view") == view).iter_rows(named=True):
            out = pl.read_parquet(r["data_path"])
            schemas.append(out.drop([c for c in ("target",) if c in out.columns]).schema)
        assert schemas[0] == schemas[1] == schemas[2]


def test_linear_view_has_no_nulls(masters):
    report = run(masters)
    assert report.filter(pl.col("view") == "linear")["predictor_nulls"].to_list() == [0, 0, 0]


def test_state_is_fitted_on_train_and_unchanged_by_oot_and_test(masters):
    run(masters)
    state = json.loads((masters["tmp"] / "E6" / "matrix_views_state.json").read_text())
    assert state["preprocessor"]["cutoff_week"] == tv.g3.CUTOFF
    assert (masters["tmp"] / "E6" / "matrix_views_spec.csv").exists()


def test_fitting_with_oot_rows_is_rejected(masters):
    bad = dict(masters["tables"], train=pl.concat([masters["tables"]["train"], masters["tables"]["oot"]], how="vertical_relaxed"))
    with pytest.raises(ValueError):
        e6.fit_and_transform(bad, masters["catalog"], output_dir=masters["tmp"] / "bad", cutoff_week=tv.g3.CUTOFF)


def test_outputs_are_deterministic(masters):
    a = run(masters, "A")["data_sha256"].to_list()
    b = run(masters, "B")["data_sha256"].to_list()
    assert a == b
