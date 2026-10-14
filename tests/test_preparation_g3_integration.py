"""G3: integración sintética del pipeline completo de preparación.

Todas las familias (adaptadores reales C1-C7) -> ensamblador (E1-E2) -> preprocesador (D1-D3,
con D4 y D5) -> vistas lineal y de árboles (D7), para train, OOT y test, con las decisiones G2
publicadas. Reutiliza las fuentes sintéticas de la prueba E3 y extiende la cadena hasta las vistas.
"""

import hashlib
import json
from dataclasses import replace

import polars as pl
import preparation_bureau_fixtures as bureau_fx
import preparation_fixtures as app_fx
import pytest
import test_preparation_e3_integration as e3

from dm_credit.baseline import RESERVED
from dm_credit.preparation import contracts
from dm_credit.preparation.assembly import assemble_feature_matrix
from dm_credit.preparation.families import applications_people as ap
from dm_credit.preparation.families import applications_people_g2 as ap_g2
from dm_credit.preparation.families import bureau, depth0, products_registry
from dm_credit.preparation.preprocessing import UNKNOWN, build_spec_from_catalog
from dm_credit.preparation.schema import validate_matrix_schemas
from dm_credit.preparation.views import (
    fit_matrix_views,
    load_matrix_views,
    save_matrix_views,
)

CUTOFF = e3.CUTOFF
SPLITS = ("train", "oot", "test")
UNSEEN_ROLE = "ZZ"  # categoría que train nunca vio: debe llegar como __UNKNOWN__


def _write_sources(root):
    ids = [1, 2, 3, 4, 5, 6]
    e3._write_depth0_sources(root, ids)
    e3._write_product_sources(root, ids)
    e3._write_bureau_sources(root, bureau_fx, ids)
    e3._write_app_people_sources(root, app_fx, ids)
    folder = root / "parquet_files" / "test"
    folder.mkdir(parents=True, exist_ok=True)
    for source, make in (
        ("applprev_1", app_fx.applprev_1),
        ("applprev_2", app_fx.applprev_2),
        ("person_1", app_fx.person_1),
        ("person_2", app_fx.person_2),
    ):
        frame = make().with_columns(pl.col("case_id") + 100)
        if source == "person_1":  # el caso de test 101 trae un rol que train no conoce
            frame = frame.with_columns(
                pl.when((pl.col("case_id") == 101) & (pl.col("num_group1") == 0))
                .then(pl.lit(UNSEEN_ROLE))
                .otherwise(pl.col("role_1084L"))
                .alias("role_1084L")
            )
        frame.write_parquet(folder / f"test_{source}.parquet")


def _configs():
    """Columnas aprobadas en G2 que los fixtures tienen; no se simula ninguna aprobación."""
    app_cfg, people_cfg = ap_g2.load_g2_configs()
    app_cfg = replace(
        app_cfg,
        numeric={"credamount_590A": app_cfg.numeric["credamount_590A"]},
        categorical_distinct=("credtype_587L",),
        category_shares={"credtype_587L": app_cfg.category_shares["credtype_587L"]},
        bool_cols=("isbidproduct_390L",),
        date_cols=("approvaldate_319D",),
    )
    assert "role_1084L" in people_cfg.main_categorical  # aprobada en G2
    people_cfg = replace(
        people_cfg,
        main_numeric=("mainoccupationinc_384A",),
        main_categorical=("role_1084L",),
        birth_col="birth_259D",
        employment_col="empl_employedfrom_271D",
        contact_category_shares={},
    )
    return app_cfg, people_cfg


def _artifacts(split, base, out, app_cfg, people_cfg):
    arts = []
    for source, builder in (
        ("static_0", depth0.build_static_0_features),
        ("static_cb_0", depth0.build_static_cb_0_features),
    ):
        arts.append(
            builder(
                base,
                split,
                cutoff_week=CUTOFF,
                output_dir=out / source,
                selection_path=depth0.DEFAULT_SELECTION_PATH,
                g2_path=depth0.DEFAULT_G2_PATH,
            )
        )
    arts.append(
        ap.build_applprev_features(
            base, split, cutoff_week=CUTOFF, output_dir=out / "applprev", config=app_cfg
        )
    )
    arts.append(
        ap.build_person_features(
            base,
            split,
            cutoff_week=CUTOFF,
            output_dir=out / "person",
            config=people_cfg,
        )
    )
    for family, builder in bureau.BUILDERS.items():
        arts.append(
            builder(
                base,
                split,
                cutoff_week=CUTOFF,
                output_dir=out / family,
                selection_path=bureau.DEFAULT_SELECTION_PATH,
                g2_path=bureau.DEFAULT_G2_PATH,
                mode="approved",
            )
        )
    for family, builder in products_registry.BUILDERS.items():
        arts.append(
            builder(
                base,
                split,
                cutoff_week=CUTOFF,
                output_dir=out / family,
                selection_path=products_registry.DEFAULT_SELECTION_PATH,
                g2_path=products_registry.DEFAULT_G2_PATH,
            )
        )
    return arts


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    _write_sources(root)
    base = e3._base([1, 2, 3, 4, 5, 6], [10, 10, 11, 11, 12, 12])
    bases = {
        "train": base.filter(pl.col("WEEK_NUM") < CUTOFF),
        "oot": base.filter(pl.col("WEEK_NUM") >= CUTOFF),
        "test": e3._base([101, 102], [20, 20], labeled=False),
    }
    app_cfg, people_cfg = _configs()
    artifacts, matrices = {}, {}
    for split in SPLITS:
        artifacts[split] = _artifacts(
            split, bases[split], tmp_path / "art" / split, app_cfg, people_cfg
        )
        m = assemble_feature_matrix(
            bases[split],
            artifacts[split],
            split,
            output_dir=tmp_path / "matrix" / split,
        )
        matrices[split] = pl.read_parquet(m.data_path)
    spec = build_spec_from_catalog(contracts.feature_catalog(artifacts["train"]))
    views = fit_matrix_views(matrices["train"], spec, cutoff_week=CUTOFF)
    return {
        "bases": bases,
        "artifacts": artifacts,
        "matrices": matrices,
        "spec": spec,
        "views": views,
        "tmp": tmp_path,
    }


def _predictors(frame, split, views):
    controls = [
        c for c in views.control_columns if not (split == "test" and c == "target")
    ]
    return frame.drop(controls)


def test_every_family_reaches_the_assembled_matrix(pipeline):
    families = {a.family for a in pipeline["artifacts"]["train"]}
    assert families == {
        "static_0",
        "static_cb_0",
        "applprev",
        "person",
        "credit_bureau_a",
        "credit_bureau_b",
        *products_registry.SOURCES,
    }
    for split in SPLITS:
        m, base = pipeline["matrices"][split], pipeline["bases"][split]
        assert m["case_id"].to_list() == base["case_id"].to_list()
        assert m.columns[: base.width] == base.columns


def test_partition_and_as_of_filter_survive_aggregation_and_join(pipeline):
    train = pipeline["matrices"]["train"]
    oot = pipeline["matrices"]["oot"]
    assert train["case_id"].to_list() == [1, 2]
    assert oot["case_id"].to_list() == [3, 4, 5, 6]

    def value(matrix, case_id, column):
        return matrix.filter(pl.col("case_id") == case_id)[column].item()

    # El caso 2 tiene una solicitud posterior y su hijo: ninguno debe entrar al agregado.
    assert value(train, 2, "applprev_1__n_records") == 1
    assert value(train, 2, "applprev_1__credamount_590a__sum__all") == 500.0
    assert value(train, 2, "applprev_2__n_records") == 1
    # La solicitud del mismo día se conserva, la de fecha nula se descarta con su hijo.
    assert value(oot, 4, "applprev_1__n_records") == 1
    assert value(oot, 4, "applprev_1__credamount_590a__sum__all") == 400.0
    assert value(oot, 4, "applprev_2__n_records") == 1
    assert value(oot, 5, "applprev_1__source_present") is False
    assert value(oot, 5, "applprev_1__n_records") == 0


def test_assembled_matrices_pass_the_final_schema_contract(pipeline):
    m = pipeline["matrices"]
    validate_matrix_schemas(m["train"], m["oot"], m["test"], cutoff_week=CUTOFF)


def test_feature_names_follow_the_frozen_grammar_and_never_use_reserved_columns(
    pipeline,
):
    for name in pipeline["spec"].columns:
        contracts.parse_feature_name(name)
        assert name not in RESERVED


def test_only_applprev_and_person_contribute_predictors_in_the_first_cut(pipeline):
    for art in pipeline["artifacts"]["train"]:
        roles = {d.role for d in art.feature_schema}
        if art.family in {"applprev", "person"}:
            assert "predictor" in roles
        else:
            assert roles == {"cobertura"}


@pytest.mark.parametrize("view", ["linear", "tree"])
def test_views_keep_rows_order_and_controls(pipeline, view):
    v = pipeline["views"]
    for split in SPLITS:
        out = getattr(v, view)(pipeline["matrices"][split], split)
        base = pipeline["bases"][split]
        assert out["case_id"].to_list() == base["case_id"].to_list()
        assert ("target" in out.columns) == (split != "test")
        assert out.columns[: len(v.control_columns) - (split == "test")] == [
            c for c in v.control_columns if not (split == "test" and c == "target")
        ]


@pytest.mark.parametrize("view", ["linear", "tree"])
def test_views_have_identical_predictors_in_every_split(pipeline, view):
    v = pipeline["views"]
    schemas = {
        split: _predictors(
            getattr(v, view)(pipeline["matrices"][split], split), split, v
        ).schema
        for split in SPLITS
    }
    assert schemas["train"] == schemas["oot"] == schemas["test"]
    assert list(schemas["train"]) == list(v.predictor_columns(view))


def test_linear_view_has_no_nulls_and_finite_numbers(pipeline):
    v = pipeline["views"]
    for split in SPLITS:
        pred = _predictors(v.linear(pipeline["matrices"][split], split), split, v)
        assert sum(pred[c].null_count() for c in pred.columns) == 0
        assert all(pred[c].is_finite().all() for c in v.spec.numeric if c in pred.columns)


def test_tree_view_keeps_native_nulls(pipeline):
    v = pipeline["views"]
    pred = _predictors(v.tree(pipeline["matrices"]["oot"], "oot"), "oot", v)
    assert (
        sum(pred[c].null_count() for c in pred.columns) > 0
    )  # casos sin historial de applprev en OOT


def test_unseen_category_in_test_becomes_unknown_in_both_views(pipeline):
    col = "person_1__role_1084l__main__all"
    # en los dos casos de train el rol es el mismo: la columna es constante y D1 la descarta
    assert col in pipeline["views"].dropped_features
    # con dos roles en train la columna queda, y el rol nuevo de test llega como __UNKNOWN__
    train = pipeline["matrices"]["train"]
    varied = train.with_columns(pl.when(pl.col("case_id") == train["case_id"].max()).then(pl.lit("OTHER")).otherwise(pl.col(col)).alias(col))
    v = fit_matrix_views(varied, pipeline["spec"], cutoff_week=CUTOFF)
    tree = v.tree(pipeline["matrices"]["test"], "test")
    assert tree.filter(pl.col("case_id") == 101)[col].cast(pl.String).item() == UNKNOWN
    linear = v.linear(pipeline["matrices"]["test"], "test")
    # lineal: train nunca vio __UNKNOWN__, así que ese nivel es constante y se descarta (D1);
    # el rol nuevo queda con todos los niveles conocidos en cero, sin ampliar el vocabulario
    assert f"{col}__is__{UNKNOWN}" in v.dropped_linear
    onehot = [c for c in linear.columns if c.startswith(f"{col}__is__")]
    assert onehot and linear.filter(pl.col("case_id") == 101).select(onehot).sum_horizontal().item() == 0


def test_fitting_rejects_oot_and_transforming_oot_and_test_learns_nothing(pipeline):
    v, m = pipeline["views"], pipeline["matrices"]
    with pytest.raises(ValueError):
        fit_matrix_views(m["oot"], pipeline["spec"], cutoff_week=CUTOFF)
    before = v.to_dict()
    for split in ("oot", "test"):
        v.linear(m[split], split)
        v.tree(m[split], split)
    assert v.to_dict() == before


def test_fitted_state_round_trips_and_reproduces_the_views(pipeline):
    v = pipeline["views"]
    path = save_matrix_views(v, pipeline["tmp"] / "views.json")
    restored = load_matrix_views(path)
    for split in SPLITS:
        m = pipeline["matrices"][split]
        assert restored.linear(m, split).equals(v.linear(m, split))
        assert restored.tree(m, split).equals(v.tree(m, split))


def test_published_artifacts_are_verified_by_checksum(pipeline):
    for split in SPLITS:
        for art in pipeline["artifacts"][split]:
            assert contracts.load_family_artifact(art.manifest_path) == art


def test_assembled_matrix_manifest_records_inputs_and_verified_bytes(pipeline):
    for split in SPLITS:
        path = pipeline["tmp"] / "matrix" / split
        data = path / f"feature_matrix__{split}.parquet"
        manifest = json.loads(
            (path / f"feature_matrix__{split}.manifest.json").read_text()
        )
        assert manifest["split"] == split
        assert manifest["row_count"] == pipeline["bases"][split].height
        assert manifest["data_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
        assert set(manifest["input_manifests"]) == {
            str(art.manifest_path) for art in pipeline["artifacts"][split]
        }
