"""Pruebas C7: cobertura de registros fiscales sin predictores (Gabriel, P5).

Incluye la prueba reina del backlog: ``test_tax_registry_c_1.parquet`` vacío
con columnas físicas ``Null`` debe producir el esquema correcto sin error.
"""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from dm_credit.preparation import contracts
from dm_credit.preparation.families import products_registry as pr

TAX_SOURCES = ("tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1")
TAX_BUILDERS = {
    "tax_registry_a_1": pr.build_tax_registry_a_1_features,
    "tax_registry_b_1": pr.build_tax_registry_b_1_features,
    "tax_registry_c_1": pr.build_tax_registry_c_1_features,
}


@pytest.fixture
def tax_data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    for split in ("train", "test"):
        (root / "parquet_files" / split).mkdir(parents=True)
    monkeypatch.setenv("DM_DATA_DIR", str(root))

    train = root / "parquet_files" / "train"
    # tax_a 1:N: caso 1 con 3 filas, caso 2 con 1, caso 3 sin cobertura
    pl.DataFrame({"case_id": [1, 1, 1, 2], "num_group1": [0, 1, 2, 0]}).write_parquet(
        train / "train_tax_registry_a_1.parquet"
    )
    pl.DataFrame({"case_id": [2], "num_group1": [0]}).write_parquet(
        train / "train_tax_registry_b_1.parquet"
    )
    pl.DataFrame({"case_id": [1, 3], "num_group1": [0, 0]}).write_parquet(
        train / "train_tax_registry_c_1.parquet"
    )

    test = root / "parquet_files" / "test"
    pl.DataFrame({"case_id": [10], "num_group1": [0]}).write_parquet(
        test / "test_tax_registry_a_1.parquet"
    )
    pl.DataFrame({"case_id": [10], "num_group1": [0]}).write_parquet(
        test / "test_tax_registry_b_1.parquet"
    )
    # como el real: vacío con columnas físicas Null (0 filas, 5 columnas)
    pl.DataFrame(
        {"case_id": [], "employername_160M": [], "num_group1": [], "pmtamount_36A": [], "processingdate_168D": []},
        schema={"case_id": pl.Null, "employername_160M": pl.Null, "num_group1": pl.Null, "pmtamount_36A": pl.Null, "processingdate_168D": pl.Null},
    ).write_parquet(test / "test_tax_registry_c_1.parquet")
    return root


@pytest.fixture
def b6_selection(tmp_path):
    path = tmp_path / "B6_selection.csv"
    pl.DataFrame(
        {
            "family": ["tax_registry_a_1", "tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1"],
            "column": ["amount_1A", "case_id", "amount_2A", "pmtamount_36A"],
            "dtype": ["Float64", "Int64", "Float64", "Float64"],
            "action": ["bloquear_hasta_G2", "excluir", "bloquear_hasta_G2", "bloquear_hasta_G2"],
            "reason": ["r", "clave", "r", "r"],
            "note": ["", "", "", ""],
        }
    ).write_csv(path)
    return path


@pytest.fixture
def g2_decision(tmp_path):
    path = tmp_path / "G2_tax.csv"
    pl.DataFrame(
        {
            "family": ["tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1"],
            "column": ["amount_1A", "amount_2A", "pmtamount_36A"],
            "g2_decision": ["bloqueada", "bloqueada", "bloqueada"],
        }
    ).write_csv(path)
    return path


def _base(split="train"):
    ids = [1, 2, 3] if split == "train" else [10]
    return pl.DataFrame(
        {
            "case_id": ids,
            "WEEK_NUM": [20] * len(ids),
            "date_decision": [date(2020, 1, 1)] * len(ids),
        }
    )


def _build(source, split, tmp_path, b6_selection, g2_decision):
    return TAX_BUILDERS[source](
        _base(split), split, output_dir=tmp_path / "out",
        selection_path=b6_selection, g2_path=g2_decision,
    )


@pytest.mark.usefixtures("tax_data")
@pytest.mark.parametrize(
    ("source", "train_counts", "test_counts"),
    [
        ("tax_registry_a_1", [3, 1, 0], [1]),
        ("tax_registry_b_1", [0, 1, 0], [1]),
        ("tax_registry_c_1", [1, 0, 1], [0]),
    ],
)
def test_tax_coverage_is_one_row_per_base_case(
    b6_selection, g2_decision, tmp_path, source, train_counts, test_counts
):
    for split, counts in (("train", train_counts), ("test", test_counts)):
        artifact = _build(source, split, tmp_path, b6_selection, g2_decision)
        features = pl.read_parquet(artifact.data_path)
        assert artifact.family == source and artifact.split == split
        assert features["case_id"].to_list() == _base(split)["case_id"].to_list()
        assert features[f"{source}__n_records"].to_list() == counts
        assert features[f"{source}__source_present"].to_list() == [c > 0 for c in counts]
        assert features.schema[f"{source}__n_records"] == pl.Int64
        assert features.schema[f"{source}__source_present"] == pl.Boolean
        assert all(f.role == "cobertura" for f in artifact.feature_schema)


@pytest.mark.usefixtures("tax_data")
def test_empty_null_test_file_yields_typed_zeros(b6_selection, g2_decision, tmp_path):
    artifact = _build("tax_registry_c_1", "test", tmp_path, b6_selection, g2_decision)
    features = pl.read_parquet(artifact.data_path)
    assert features.columns == ["case_id", "tax_registry_c_1__n_records", "tax_registry_c_1__source_present"]
    assert features["tax_registry_c_1__n_records"].to_list() == [0]
    assert features["tax_registry_c_1__source_present"].to_list() == [False]
    manifest = json.loads(artifact.manifest_path.read_text())
    assert manifest["exclusions"]["g2_approved"] == 0
    reloaded = contracts.load_family_artifact(artifact.manifest_path)
    assert reloaded.case_count == 1


def test_real_b6_g2_select_nothing_for_tax():
    for source in TAX_SOURCES:
        approved, counts = pr._selected_columns(
            source,
            Path("results/preparation/B6_products_registry_columns.csv"),
            Path("results/preparation/G2_products_registry_columns.csv"),
        )
        assert approved == []
        assert counts["g2_approved"] == 0
        assert counts["b6_excluded"] == 3  # case_id, num_group1 y nombre fiscal


@pytest.mark.usefixtures("tax_data")
def test_tax_rebuild_is_deterministic(b6_selection, g2_decision, tmp_path):
    arts = [
        pr.build_tax_registry_a_1_features(
            _base("train"), "train", output_dir=tmp_path / d,
            selection_path=b6_selection, g2_path=g2_decision,
        )
        for d in ("a", "b")
    ]
    assert pl.read_parquet(arts[0].data_path).equals(pl.read_parquet(arts[1].data_path))


@pytest.mark.usefixtures("tax_data")
def test_f7_exclusions_mirror_manifests_with_applprev_schema(b6_selection, g2_decision, tmp_path):
    arts = [
        pr.build_tax_registry_a_1_features(_base("train"), "train", output_dir=tmp_path / "a", selection_path=b6_selection, g2_path=g2_decision),
        pr.build_tax_registry_a_1_features(_base("test"), "test", output_dir=tmp_path / "b", selection_path=b6_selection, g2_path=g2_decision),
    ]
    table = pr.collect_f7_exclusions(arts)
    assert table.columns == ["split", "family", "metric", "value"]
    assert table.schema["value"] == pl.Int64
    assert table.rows() == sorted(table.rows())
    assert set(table["metric"].to_list()) >= {"b6_excluded", "b6_blocked", "g2_approved", "source_rows_outside_base"}
    out = pr.write_f7_exclusions(arts, tmp_path / "F7.csv")
    assert out.is_file()
    # mismo esquema que el aporte F7 de applprev/person
    assert pl.read_csv("results/preparation/F7_applications_people_exclusions.csv").columns == table.columns
    with pytest.raises(ValueError, match="empty"):
        pr.write_f7_exclusions([], tmp_path / "F7empty.csv")


@pytest.mark.usefixtures("tax_data")
@pytest.mark.parametrize("split", ["train", "test"])
def test_build_family_features_dispatches_tax_registry_c_1(b6_selection, g2_decision, tmp_path, split):
    from dm_credit.preparation import build

    artifact = build.build_family_features(
        "tax_registry_c_1", _base(split), split, output_dir=tmp_path / "out",
        selection_path=b6_selection, g2_path=g2_decision,
    )
    features = pl.read_parquet(artifact.data_path)
    assert artifact.family == "tax_registry_c_1" and artifact.split == split
    assert features["case_id"].to_list() == _base(split)["case_id"].to_list()
