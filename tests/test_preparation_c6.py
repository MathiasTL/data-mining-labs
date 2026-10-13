"""Pruebas C6: cobertura de productos sin predictores (Gabriel, P5).

Datos sintéticos con la estructura real de archivos; resultados calculados a mano.
"""

import json
from datetime import date

import polars as pl
import pytest

from dm_credit.preparation import contracts
from dm_credit.preparation.families import products_registry as pr


@pytest.fixture
def product_data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    for split in ("train", "test"):
        (root / "parquet_files" / split).mkdir(parents=True)
    monkeypatch.setenv("DM_DATA_DIR", str(root))

    train = root / "parquet_files" / "train"
    # debitcard_1 1:N: caso 1 con 3 filas (una duplicada en num_group1 distinto), caso 2 con 1
    pl.DataFrame(
        {"case_id": [1, 1, 1, 2, 99], "num_group1": [0, 1, 2, 0, 0]}
    ).write_parquet(train / "train_debitcard_1.parquet")
    # deposit_1 1:N: caso 1 con 2 filas, caso 3 con 1
    pl.DataFrame({"case_id": [1, 1, 3], "num_group1": [0, 1, 0]}).write_parquet(
        train / "train_deposit_1.parquet"
    )
    # other_1 1:0..1: solo caso 2
    pl.DataFrame({"case_id": [2], "num_group1": [0]}).write_parquet(
        train / "train_other_1.parquet"
    )

    test = root / "parquet_files" / "test"
    pl.DataFrame({"case_id": [10, 10], "num_group1": [0, 1]}).write_parquet(
        test / "test_debitcard_1.parquet"
    )
    pl.DataFrame({"case_id": [10], "num_group1": [0]}).write_parquet(
        test / "test_deposit_1.parquet"
    )
    pl.DataFrame({"case_id": [], "num_group1": []}, schema={"case_id": pl.Int64, "num_group1": pl.Int64}).write_parquet(
        test / "test_other_1.parquet"
    )
    return root


@pytest.fixture
def b6_selection(tmp_path):
    path = tmp_path / "B6_selection.csv"
    pl.DataFrame(
        {
            "family": ["debitcard_1", "debitcard_1", "deposit_1", "other_1", "other_1"],
            "column": ["amount_1A", "case_id", "amount_2A", "amt_1A", "num_group1"],
            "dtype": ["Float64", "Int64", "Float64", "Float64", "Int64"],
            "action": ["bloquear_hasta_G2", "excluir", "bloquear_hasta_G2", "bloquear_hasta_G2", "excluir"],
            "reason": ["r", "clave", "r", "r", "clave"],
            "note": ["", "", "", "", ""],
        }
    ).write_csv(path)
    return path


@pytest.fixture
def g2_decision(tmp_path):
    path = tmp_path / "G2_products.csv"
    pl.DataFrame(
        {
            "family": ["debitcard_1", "deposit_1", "other_1"],
            "column": ["amount_1A", "amount_2A", "amt_1A"],
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


def _build(builder, split, tmp_path, b6_selection, g2_decision, **kw):
    return builder(
        _base(split), split, output_dir=tmp_path / "out",
        selection_path=b6_selection, g2_path=g2_decision, **kw,
    )


@pytest.mark.usefixtures("product_data")
@pytest.mark.parametrize(
    ("builder", "family", "train_counts", "test_counts"),
    [
        (pr.build_debitcard_1_features, "debitcard_1", [3, 1, 0], [2]),
        (pr.build_deposit_1_features, "deposit_1", [2, 0, 1], [1]),
        (pr.build_other_1_features, "other_1", [0, 1, 0], [0]),
    ],
)
def test_coverage_is_one_row_per_base_case(
    b6_selection, g2_decision, tmp_path, builder, family, train_counts, test_counts
):
    for split, counts in (("train", train_counts), ("test", test_counts)):
        artifact = _build(builder, split, tmp_path, b6_selection, g2_decision)
        features = pl.read_parquet(artifact.data_path)
        assert artifact.family == family and artifact.split == split
        assert features["case_id"].to_list() == _base(split)["case_id"].to_list()
        assert features[f"{family}__n_records"].to_list() == counts
        assert features[f"{family}__source_present"].to_list() == [c > 0 for c in counts]
        # caso 99 existe en la fuente pero no en la base: se ignora
        assert 99 not in features["case_id"].to_list()
        # cero predictores en el primer corte
        assert all(f.role == "cobertura" for f in artifact.feature_schema)


@pytest.mark.usefixtures("product_data")
def test_manifest_records_counts_rules_and_inputs(b6_selection, g2_decision, tmp_path):
    artifact = _build(pr.build_other_1_features, "train", tmp_path, b6_selection, g2_decision)
    manifest = json.loads(artifact.manifest_path.read_text())
    assert manifest["rules_version"] == pr.RULES_VERSION
    assert manifest["exclusions"]["b6_blocked"] == 1
    assert manifest["exclusions"]["g2_approved"] == 0
    assert manifest["inputs"]["cutoff_week"] == "74"
    assert len(manifest["inputs"]["base_sha256"]) == 64
    assert "train_other_1.parquet" in manifest["inputs"]["shards"]
    # el artefacto se recarga desde su manifiesto
    reloaded = contracts.load_family_artifact(artifact.manifest_path)
    assert reloaded.case_count == artifact.case_count == 3


def test_duplicate_keys_are_rejected():
    base = pl.DataFrame({"case_id": [1]})
    dup_1n = pl.DataFrame({"case_id": [1, 1], "num_group1": [0, 0]})
    with pytest.raises(ValueError, match="not unique"):
        pr.aggregate_coverage("debitcard_1", base, dup_1n)
    dup_01 = pl.DataFrame({"case_id": [1, 1], "num_group1": [0, 1]})
    with pytest.raises(ValueError, match="1:0..1"):
        pr.aggregate_coverage("other_1", base, dup_01)
    nulls = pl.DataFrame({"case_id": [None], "num_group1": [0]})
    with pytest.raises(ValueError, match="nulls"):
        pr.aggregate_coverage("other_1", base, nulls)


def test_empty_source_yields_zeros_without_error():
    base = pl.DataFrame({"case_id": [1, 2]})
    empty = pl.DataFrame({"case_id": [], "num_group1": []}, schema={"case_id": pl.Int64, "num_group1": pl.Int64})
    out = pr.aggregate_coverage("deposit_1", base, empty).features
    assert out["deposit_1__n_records"].to_list() == [0, 0]
    assert out["deposit_1__source_present"].to_list() == [False, False]


def test_unknown_source_is_rejected():
    base = pl.DataFrame({"case_id": [1]})
    with pytest.raises(ValueError, match="unknown C6 source"):
        pr.aggregate_coverage("bureau_x", base, base)


@pytest.mark.usefixtures("product_data")
def test_approved_predictors_require_a_follow_up(b6_selection, tmp_path):
    g2 = tmp_path / "G2_approved.csv"
    pl.DataFrame(
        {"family": ["debitcard_1"], "column": ["amount_1A"], "g2_decision": ["aprobada"]}
    ).write_csv(g2)
    with pytest.raises(ValueError, match="follow-up"):
        _build(pr.build_debitcard_1_features, "train", tmp_path, b6_selection, g2)


@pytest.mark.usefixtures("product_data")
def test_g2_cannot_approve_b6_excluded_or_reserved(b6_selection, tmp_path):
    bad_excluded = tmp_path / "G2_bad.csv"
    pl.DataFrame(
        {"family": ["debitcard_1"], "column": ["case_id"], "g2_decision": ["aprobada"]}
    ).write_csv(bad_excluded)
    with pytest.raises(ValueError, match="reserved"):
        _build(pr.build_debitcard_1_features, "train", tmp_path, b6_selection, bad_excluded)


@pytest.mark.usefixtures("product_data")
def test_missing_manifests_are_an_error(b6_selection, g2_decision, tmp_path):
    with pytest.raises(FileNotFoundError, match="B6"):
        _build(pr.build_debitcard_1_features, "train", tmp_path, tmp_path / "nope.csv", g2_decision)
    with pytest.raises(FileNotFoundError, match="G2"):
        _build(pr.build_debitcard_1_features, "train", tmp_path, b6_selection, tmp_path / "nope.csv")


@pytest.mark.usefixtures("product_data")
def test_invalid_base_and_split_are_rejected(b6_selection, g2_decision, tmp_path):
    bad_week = _base("train").with_columns(WEEK_NUM=pl.lit(99))
    with pytest.raises(ValueError, match="74"):
        pr.build_debitcard_1_features(bad_week, "train", output_dir=tmp_path / "o", selection_path=b6_selection, g2_path=g2_decision)
    with pytest.raises(ValueError, match="split"):
        pr.build_debitcard_1_features(_base("train"), "valid", output_dir=tmp_path / "o", selection_path=b6_selection, g2_path=g2_decision)


@pytest.mark.usefixtures("product_data")
def test_rebuild_is_deterministic(b6_selection, g2_decision, tmp_path):
    arts = [
        pr.build_deposit_1_features(
            _base("train"), "train", output_dir=tmp_path / d,
            selection_path=b6_selection, g2_path=g2_decision,
        )
        for d in ("a", "b")
    ]
    assert pl.read_parquet(arts[0].data_path).equals(pl.read_parquet(arts[1].data_path))
    assert json.loads(arts[0].manifest_path.read_text()) == json.loads(arts[1].manifest_path.read_text())


@pytest.mark.usefixtures("product_data")
def test_catalog_lists_only_coverage_features(b6_selection, g2_decision, tmp_path):
    arts = [
        _build(b, "train", tmp_path / p, b6_selection, g2_decision)
        for b, p in [
            (pr.build_debitcard_1_features, "a"),
            (pr.build_deposit_1_features, "b"),
            (pr.build_other_1_features, "c"),
        ]
    ]
    catalog = contracts.feature_catalog(arts)
    assert catalog.height == 6 and set(catalog["role"].to_list()) == {"cobertura"}
    out = contracts.write_feature_catalog(arts, tmp_path / "C8_products.csv")
    assert out.is_file()


@pytest.mark.usefixtures("product_data")
@pytest.mark.parametrize("family", ["debitcard_1", "deposit_1", "other_1"])
def test_build_family_features_dispatches_products(b6_selection, g2_decision, tmp_path, family):
    from dm_credit.preparation import build

    artifact = build.build_family_features(
        family, _base("train"), "train", output_dir=tmp_path / "out",
        selection_path=b6_selection, g2_path=g2_decision,
    )
    direct = pr.BUILDERS[family](
        _base("train"), "train", output_dir=tmp_path / "direct",
        selection_path=b6_selection, g2_path=g2_decision,
    )
    assert artifact.family == family and artifact.split == "train"
    assert pl.read_parquet(artifact.data_path).equals(pl.read_parquet(direct.data_path))


def test_products_and_registry_families_are_registered_without_collisions():
    from dm_credit.preparation import build
    from dm_credit.preparation.families import applications_people, bureau

    assert set(pr.BUILDERS) <= set(build.BUILDERS)
    assert {"tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1"} <= set(build.BUILDERS)
    registries = (applications_people.BUILDERS, bureau.BUILDERS, pr.BUILDERS)
    assert len(build.BUILDERS) == sum(len(r) for r in registries)
