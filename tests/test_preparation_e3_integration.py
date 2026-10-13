"""E3: builders reales y artefactos sintéticos de los cuatro grupos de familias."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import polars as pl

from dm_credit.preparation import contracts
from dm_credit.preparation.assembly import assemble_feature_matrix
from dm_credit.preparation.families import applications_people as ap
from dm_credit.preparation.families import applications_people_g2 as ap_g2
from dm_credit.preparation.families import bureau, depth0, products_registry

CUTOFF = 11


def _base(ids: list[int], weeks: list[int], *, labeled: bool = True) -> pl.DataFrame:
    data = {
        "case_id": ids,
        "WEEK_NUM": weeks,
        "date_decision": [date(2020, 6, 1)] * len(ids),
        "MONTH": [202006] * len(ids),
    }
    if labeled:
        data["target"] = [i % 2 for i in ids]
    return pl.DataFrame(data)


def _write_depth0_sources(root: Path, ids: list[int]) -> None:
    folder = root / "parquet_files" / "train"
    folder.mkdir(parents=True, exist_ok=True)
    # El G2 publicado no aprueba predictores depth0; solo se necesita case_id.
    for source in depth0.SOURCES:
        pl.DataFrame({"case_id": ids}).write_parquet(folder / f"train_{source}.parquet")
    test_folder = root / "parquet_files" / "test"
    test_folder.mkdir(parents=True, exist_ok=True)
    for source in depth0.SOURCES:
        pl.DataFrame({"case_id": [101, 102]}).write_parquet(
            test_folder / f"test_{source}.parquet"
        )


def _write_product_sources(root: Path, ids: list[int]) -> None:
    for split, split_ids in (("train", ids), ("test", [i + 100 for i in ids])):
        folder = root / "parquet_files" / split
        folder.mkdir(parents=True, exist_ok=True)
        for source in products_registry.SOURCES:
            source_ids = split_ids[::2]
            pl.DataFrame(
                {"case_id": source_ids, "num_group1": list(range(len(source_ids)))}
            ).write_parquet(folder / f"{split}_{source}.parquet")


def _write_bureau_sources(root: Path, fixture, ids: list[int]) -> None:
    # Fixture reutilizado de C4/C5. Solo claves e índices bastan: G2 real deja
    # predictores no aprobados y los anclajes se mantienen en los builders.
    for split, offset in (("train", 0), ("test", 100)):
        folder = root / "parquet_files" / split
        folder.mkdir(parents=True, exist_ok=True)
        for source, make in fixture.TABLES.items():
            frame = make()
            if split == "train":
                frame = frame.filter(pl.col("case_id").is_in(ids))
            else:
                frame = frame.filter(pl.col("case_id").is_in(ids)).with_columns(
                    pl.col("case_id") + offset
                )
            frame.write_parquet(folder / f"{split}_{source}.parquet")


def _write_app_people_sources(root: Path, fixtures, ids: list[int]) -> None:
    folder = root / "parquet_files" / "train"
    folder.mkdir(parents=True, exist_ok=True)
    for source, make in (
        ("applprev_1", fixtures.applprev_1),
        ("applprev_2", fixtures.applprev_2),
        ("person_1", fixtures.person_1),
        ("person_2", fixtures.person_2),
    ):
        make().filter(pl.col("case_id").is_in(ids)).write_parquet(
            folder / f"train_{source}.parquet"
        )


def test_e3_real_family_builders_publish_and_assemble_manifests(tmp_path, monkeypatch):
    import preparation_bureau_fixtures as bureau_fx
    import preparation_fixtures as app_fx

    root = tmp_path / "data"
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    ids = [1, 2, 3, 4, 5, 6]
    base = _base(ids, [10, 10, 11, 11, 12, 12])
    test_base = _base([101, 102], [20, 20], labeled=False)
    _write_depth0_sources(root, ids)
    _write_product_sources(root, ids)
    _write_bureau_sources(root, bureau_fx, ids)
    _write_app_people_sources(root, app_fx, ids)
    # Matching synthetic test sources, using the same fixture rows with shifted IDs.
    for source, make in (
        ("applprev_1", app_fx.applprev_1),
        ("applprev_2", app_fx.applprev_2),
        ("person_1", app_fx.person_1),
        ("person_2", app_fx.person_2),
    ):
        folder = root / "parquet_files" / "test"
        folder.mkdir(parents=True, exist_ok=True)
        (make().with_columns(pl.col("case_id") + 100)).write_parquet(
            folder / f"test_{source}.parquet"
        )

    # Manifests de decisión reales del repositorio: depth0, bureau y productos
    # no producen predictores aprobados. No se reemplazan sus decisiones.
    selection_depth0 = depth0.DEFAULT_SELECTION_PATH
    g2_depth0 = depth0.DEFAULT_G2_PATH
    selection_bureau = bureau.DEFAULT_SELECTION_PATH
    g2_bureau = bureau.DEFAULT_G2_PATH
    selection_products = products_registry.DEFAULT_SELECTION_PATH
    g2_products = products_registry.DEFAULT_G2_PATH

    parts = {
        "train": base.filter(pl.col("WEEK_NUM") < CUTOFF),
        "oot": base.filter(pl.col("WEEK_NUM") >= CUTOFF),
        "test": test_base,
    }
    app_cfg, people_cfg = ap_g2.load_g2_configs()
    # Subconjuntos soportados por los fixtures, tomados únicamente de columnas
    # que load_g2_configs publicó como aprobadas; no se simulan aprobaciones.
    app_cfg = replace(
        app_cfg,
        numeric={"credamount_590A": app_cfg.numeric["credamount_590A"]},
        categorical_distinct=("credtype_587L",),
        category_shares={"credtype_587L": app_cfg.category_shares["credtype_587L"]},
        bool_cols=("isbidproduct_390L",),
        date_cols=("approvaldate_319D",),
    )
    people_cfg = replace(
        people_cfg,
        main_numeric=("mainoccupationinc_384A",),
        main_categorical=(),
        birth_col="birth_259D",
        employment_col="empl_employedfrom_271D",
        contact_category_shares={},
    )
    artifacts_by_split = {}
    for split, split_base in parts.items():
        output = tmp_path / "artifacts" / split
        artifacts = []
        for source, builder in (
            ("static_0", depth0.build_static_0_features),
            ("static_cb_0", depth0.build_static_cb_0_features),
        ):
            artifacts.append(
                builder(
                    split_base,
                    split,
                    cutoff_week=CUTOFF,
                    output_dir=output / source,
                    selection_path=selection_depth0,
                    g2_path=g2_depth0,
                )
            )
        artifacts.append(
            ap.build_applprev_features(
                split_base,
                split,
                cutoff_week=CUTOFF,
                output_dir=output / "applprev",
                config=app_cfg,
            )
        )
        artifacts.append(
            ap.build_person_features(
                split_base,
                split,
                cutoff_week=CUTOFF,
                output_dir=output / "person",
                config=people_cfg,
            )
        )
        for family, builder in bureau.BUILDERS.items():
            artifacts.append(
                builder(
                    split_base,
                    split,
                    cutoff_week=CUTOFF,
                    output_dir=output / family,
                    selection_path=selection_bureau,
                    g2_path=g2_bureau,
                    mode="approved",
                )
            )
        for family, builder in products_registry.BUILDERS.items():
            artifacts.append(
                builder(
                    split_base,
                    split,
                    cutoff_week=CUTOFF,
                    output_dir=output / family,
                    selection_path=selection_products,
                    g2_path=g2_products,
                )
            )
        artifacts_by_split[split] = artifacts

    # Cada builder produjo un artefacto recargable desde su manifest; el ensamblador
    # verifica el manifest y los bytes de datos antes de unirlos.
    schemas_by_split = {}
    for split, split_base in parts.items():
        artifacts = artifacts_by_split[split]
        assert all(
            contracts.load_family_artifact(a.manifest_path) == a for a in artifacts
        )
        assert {a.family for a in artifacts} == {
            "static_0",
            "static_cb_0",
            "applprev",
            "person",
            "credit_bureau_a",
            "credit_bureau_b",
            *products_registry.SOURCES,
        }
        schemas_by_split[split] = {
            artifact.family: artifact.feature_schema for artifact in artifacts
        }
        matrix_artifact = assemble_feature_matrix(
            split_base, artifacts, split, output_dir=tmp_path / "matrix"
        )
        matrix = pl.read_parquet(matrix_artifact.data_path)
        assert matrix.height == split_base.height
        assert matrix["case_id"].to_list() == split_base["case_id"].to_list()
        assert matrix.columns[: len(split_base.columns)] == split_base.columns
        assert matrix_artifact.control_columns == tuple(split_base.columns)
        assert "target" not in matrix.columns if split == "test" else True
        for artifact in artifacts:
            family_frame = pl.read_parquet(artifact.data_path)
            assert family_frame["case_id"].to_list() == sorted(
                split_base["case_id"].to_list()
            )
            assert artifact.case_count == split_base.height
        # G2 publicado: depth0, bureau y products son coverage-only.
        for artifact in artifacts:
            if artifact.family in {
                *depth0.SOURCES,
                "credit_bureau_a",
                "credit_bureau_b",
                *products_registry.SOURCES,
            }:
                assert all(d.role == "cobertura" for d in artifact.feature_schema)
    assert (
        schemas_by_split["train"] == schemas_by_split["oot"] == schemas_by_split["test"]
    )
