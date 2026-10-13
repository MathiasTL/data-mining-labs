"""E3 — complements ``test_preparation_e3_integration.py``: depth-zero dispatch, coverage values and the empty tax registry shard."""

import polars as pl
import pytest

import preparation_depth0_artifacts as d0art
import preparation_products_artifacts as part
from dm_credit import io
from dm_credit.preparation import build
from dm_credit.preparation.assembly import assemble_feature_matrix
from dm_credit.preparation.families import depth0
from dm_credit.preparation.families import products_registry as pr
from dm_credit.preparation.split import partition_base

SPLITS = ("train", "oot", "test")
CASE_IDS = {"train": [1, 2], "oot": [3, 4, 5, 6], "test": [101, 102, 103, 104, 105, 106]}


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Depth-zero and products/tax registry sample data over the shared base of ``preparation_fixtures``."""
    root = tmp_path / "data"
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    d0art.write_sample_data(root)  # writes the base files
    part.write_sample_data(root, write_base=False)
    return root


@pytest.fixture
def depth0_paths(tmp_path):
    return d0art.write_selection(tmp_path / "B3.csv"), d0art.write_decisions(tmp_path / "G2_depth0.csv")


@pytest.fixture
def products_paths(tmp_path):
    return part.write_selection(tmp_path / "B6.csv"), part.write_decisions(tmp_path / "G2_products.csv")


def bases() -> dict[str, pl.DataFrame]:
    parts = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=d0art.CUTOFF)
    return {"train": parts.train, "oot": parts.oot, "test": parts.test}


def expected_for(values: list, split: str) -> list:
    """Slice a per-case expectation (cases 1 to 6) to the cases of a split."""
    offset = d0art.TEST_OFFSET if split == "test" else 0
    return [values[case_id - offset - 1] for case_id in CASE_IDS[split]]


def assemble(artifacts, split, tmp_path) -> tuple[pl.DataFrame, pl.DataFrame]:
    base = bases()[split]
    matrix = assemble_feature_matrix(
        base, [a for a in artifacts if a.split == split], split, output_dir=tmp_path / "matrix"
    )
    got = pl.read_parquet(matrix.data_path)
    assert got.height == matrix.row_count == base.height
    assert got["case_id"].to_list() == base["case_id"].to_list() == CASE_IDS[split]
    assert got["case_id"].n_unique() == got.height
    assert ("target" in got.columns) == (split != "test")
    return got, base


def assert_coverage_never_null(matrix: pl.DataFrame) -> None:
    coverage = [c for c in matrix.columns if c.endswith(("__n_records", "__source_present"))]
    assert coverage and all(matrix[c].null_count() == 0 for c in coverage)


@pytest.mark.parametrize("split", SPLITS)
def test_depth0_artifacts_are_accepted_by_the_assembler(data, tmp_path, depth0_paths, split):
    artifacts = d0art.build_small_artifacts(tmp_path / "out", *depth0_paths)
    assert len(artifacts) == 6
    got, base = assemble(artifacts, split, tmp_path)
    assert set(got.columns) == {*base.columns, *d0art.EXPECTED}
    for name, values in d0art.EXPECTED.items():
        assert got[name].to_list() == expected_for(values, split), name
    assert_coverage_never_null(got)
    # cases without a static_cb_0 row keep zero coverage and a null predictor
    uncovered = got.filter(~pl.col("static_cb_0__source_present"))
    assert uncovered.height > 0
    assert uncovered["static_cb_0__n_records"].to_list() == [0] * uncovered.height
    assert uncovered["static_cb_0__static_cb_value__first__all"].null_count() == uncovered.height


@pytest.mark.parametrize("split", SPLITS)
def test_products_artifacts_are_accepted_by_the_assembler(data, tmp_path, products_paths, split):
    artifacts = part.build_small_artifacts(tmp_path / "out", *products_paths)
    assert len(artifacts) == 3 * len(pr.SOURCES)
    got, base = assemble(artifacts, split, tmp_path)
    assert len(got.columns) == len(base.columns) + 2 * len(pr.SOURCES)
    for source in pr.SOURCES:
        counts = expected_for(part.EXPECTED_N_RECORDS[source], split)
        if split == "test" and source in part.EMPTY_NULL_TEST_SOURCES:
            counts = [0] * len(counts)
        assert got[f"{source}__n_records"].to_list() == counts, source
        assert got[f"{source}__source_present"].to_list() == [c > 0 for c in counts], source
    assert_coverage_never_null(got)
    # every split has cases without records in some source
    assert any(got[f"{source}__n_records"].min() == 0 for source in pr.SOURCES)


def test_empty_null_tax_registry_c_1_reaches_the_assembler(data, tmp_path, products_paths):
    shard = pl.read_parquet(data / "parquet_files" / "test" / "test_tax_registry_c_1.parquet")
    assert shard.height == 0 and set(shard.schema.values()) == {pl.Null}
    selection_path, g2_path = products_paths
    artifact = pr.build_tax_registry_c_1_features(
        bases()["test"], "test", cutoff_week=part.CUTOFF, output_dir=tmp_path / "out",
        selection_path=selection_path, g2_path=g2_path,
    )
    got, _ = assemble([artifact], "test", tmp_path)
    assert got.schema["tax_registry_c_1__n_records"] == pl.Int64
    assert got.schema["tax_registry_c_1__source_present"] == pl.Boolean
    assert got["tax_registry_c_1__n_records"].to_list() == [0] * 6
    assert got["tax_registry_c_1__source_present"].to_list() == [False] * 6


@pytest.mark.parametrize("family", ["static_0", "static_cb_0"])
def test_build_family_features_dispatches_depth0(data, tmp_path, depth0_paths, family):
    selection_path, g2_path = depth0_paths
    base = bases()["train"].select("case_id", "WEEK_NUM", "date_decision")
    artifact = build.build_family_features(
        family, base, "train", cutoff_week=d0art.CUTOFF, output_dir=tmp_path / "out",
        selection_path=selection_path, g2_path=g2_path,
    )
    direct = depth0.BUILDERS[family](
        base, "train", cutoff_week=d0art.CUTOFF, output_dir=tmp_path / "direct",
        selection_path=selection_path, g2_path=g2_path,
    )
    assert artifact.family == family and artifact.split == "train"
    assert pl.read_parquet(artifact.data_path).equals(pl.read_parquet(direct.data_path))
    assert set(depth0.BUILDERS) == set(depth0.SOURCES) <= set(build.BUILDERS)

