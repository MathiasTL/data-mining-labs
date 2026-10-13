"""F1 — Huecos de prueba detectados en la auditoría de los adaptadores.

Cada prueba cubre un caso que las pruebas de su responsable no ejercían: nulos, tipos físicos ``Null``,
``NaN``, claves inválidas y conteos calculados a mano. Están aquí para no editar archivos ajenos.
"""

import polars as pl
import pytest

import preparation_bureau_fixtures as bfx
import preparation_depth0_artifacts as d0a
import preparation_fixtures as fx
import test_preparation_bureau as tb
from dm_credit.preparation import c8_catalog
from dm_credit.preparation.contracts import CATALOG_SCHEMA
from dm_credit.preparation.families import applications_people as ap
from dm_credit.preparation.families import bureau, depth0
from dm_credit.preparation.families import products_registry as pr

APP = ap.ApplicationsConfig(
    numeric={"credamount_590A": ("count", "mean", "sum")},
    categorical_distinct=(),
    category_shares={},
    bool_cols=(),
    date_cols=(),
    children_category_shares={},
)


def apps(a1):
    return ap.aggregate_applications(fx.base(), a1, fx.applprev_2(), APP)


def get(df, case, col):
    return df.filter(pl.col("case_id") == case)[col].item()


# ------------------------------------------------------------------ applprev y person (Andre)
def test_all_null_numeric_column_gives_zero_count_and_null_stats():
    out = apps(fx.applprev_1().with_columns(pl.lit(None, dtype=pl.Float64).alias("credamount_590A")))
    assert get(out, 1, "applprev_1__credamount_590a__count__all") == 0
    assert get(out, 1, "applprev_1__credamount_590a__mean__all") is None
    assert get(out, 1, "applprev_1__credamount_590a__sum__all") is None


def test_physical_null_dtype_column_behaves_like_all_missing():
    out = apps(fx.applprev_1().with_columns(pl.lit(None, dtype=pl.Null).alias("credamount_590A")))
    assert get(out, 1, "applprev_1__credamount_590a__count__all") == 0


def test_nan_amount_is_missing_not_a_value():
    a1 = fx.applprev_1().with_columns(
        pl.when((pl.col("case_id") == 1) & (pl.col("num_group1") == 0)).then(float("nan")).otherwise(pl.col("credamount_590A")).alias("credamount_590A")
    )
    out = apps(a1)
    assert get(out, 1, "applprev_1__credamount_590a__count__all") == 2  # 2000 y 3000; el NaN no cuenta
    assert get(out, 1, "applprev_1__credamount_590a__mean__all") == 2500.0


def test_null_case_id_in_the_source_is_rejected():
    a1 = pl.concat([fx.applprev_1(), fx.applprev_1().head(1).with_columns(pl.lit(None, dtype=pl.Int64).alias("case_id"), pl.lit(9, dtype=pl.Int64).alias("num_group1"))])
    with pytest.raises(ValueError, match="base"):
        apps(a1)


# ------------------------------------------------------------------- profundidad 0 (Giancarlo)
@pytest.fixture
def depth0_data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    d0a.write_sample_data(root)
    return root


def _oot_base():
    return fx.base().filter(pl.col("WEEK_NUM") >= d0a.CUTOFF).select("case_id", "WEEK_NUM", "date_decision")


def test_depth0_first_value_is_null_when_the_source_value_is_null(depth0_data, tmp_path):
    s0 = d0a.static_0().with_columns(pl.when(pl.col("case_id") == 4).then(None).otherwise(pl.col("static_value")).alias("static_value"))
    tr = depth0_data / "parquet_files" / "train"
    s0.head(3).write_parquet(tr / "train_static_0_0.parquet")
    s0.tail(4).write_parquet(tr / "train_static_0_1.parquet")
    art = depth0.build_static_0_features(
        _oot_base(), "oot", cutoff_week=d0a.CUTOFF, output_dir=tmp_path / "o",
        selection_path=d0a.write_selection(tmp_path / "sel.csv"), g2_path=d0a.write_decisions(tmp_path / "g2.csv"),
    )
    out = pl.read_parquet(art.data_path)
    assert get(out, 4, "static_0__static_value__first__all") is None
    assert get(out, 4, "static_0__source_present") is True  # el caso tiene fila; solo falta el valor
    assert get(out, 3, "static_0__static_value__first__all") == 30.0


def test_depth0_nan_case_id_is_rejected(depth0_data, tmp_path):
    bad = pl.DataFrame({"case_id": [1.0, float("nan")], "static_cb_value": [5, 7]})
    bad.write_parquet(depth0_data / "parquet_files" / "train" / "train_static_cb_0.parquet")
    with pytest.raises(ValueError, match="case_id"):
        depth0.build_static_cb_0_features(
            _oot_base(), "oot", cutoff_week=d0a.CUTOFF, output_dir=tmp_path / "o",
            selection_path=d0a.write_selection(tmp_path / "sel.csv"), g2_path=d0a.write_decisions(tmp_path / "g2.csv"),
        )


# --------------------------------------------------------- productos y registros fiscales (Gabriel)
BASE_KEYS = pl.DataFrame({"case_id": [1, 2, 3]})


def test_tax_registry_duplicate_keys_are_rejected():
    rows = pl.DataFrame({"case_id": [1, 1], "num_group1": [0, 0]})
    with pytest.raises(ValueError, match="num_group1"):
        pr.aggregate_coverage("tax_registry_a_1", BASE_KEYS, rows)


def test_tax_registry_null_and_nan_keys_are_rejected():
    with pytest.raises(ValueError, match="null"):
        pr.aggregate_coverage("tax_registry_b_1", BASE_KEYS, pl.DataFrame({"case_id": [1, None], "num_group1": [0, 1]}))
    with pytest.raises(ValueError, match="null"):
        pr.aggregate_coverage("tax_registry_c_1", BASE_KEYS, pl.DataFrame({"case_id": [1.0, float("nan")], "num_group1": [0, 1]}))


def test_tax_registry_counts_are_hand_computed():
    rows = pl.DataFrame({"case_id": [1, 1, 1, 3, 99], "num_group1": [0, 1, 2, 0, 0]})  # 99 no está en la base
    out = pr.aggregate_coverage("tax_registry_a_1", BASE_KEYS, rows).features
    assert out["tax_registry_a_1__n_records"].to_list() == [3, 0, 1]
    assert out["tax_registry_a_1__source_present"].to_list() == [True, False, True]


# ---------------------------------------------------------------------------- bureau (Mathias)
def _bureau_a(a1):
    p = tb.plan("credit_bureau_a")
    tables = {t.table: [a1 if t.table == "credit_bureau_a_1" else bfx.TABLES[t.table]()] for t in p.tables}
    return bureau.aggregate_family(bfx.base(), p, tables)


def test_bureau_duplicate_parent_key_is_rejected():
    a1 = bfx.TABLES["credit_bureau_a_1"]()
    with pytest.raises(ValueError):
        _bureau_a(pl.concat([a1, a1.head(1)]))


def test_bureau_anchor_equal_to_decision_is_kept():
    # el fixture tiene un ancla igual a la fecha de decisión; quitarla debe cambiar el conteo de ese caso
    a1 = bfx.TABLES["credit_bureau_a_1"]()
    same_day = a1.filter(pl.col("lastupdate_388D") == bfx.DECISION)
    assert same_day.height == 1
    with_row = _bureau_a(a1).features
    without = _bureau_a(a1.filter(pl.col("lastupdate_388D").ne_missing(bfx.DECISION))).features
    case = same_day["case_id"].item()
    assert get(with_row, case, "credit_bureau_a_1__n_records") == get(without, case, "credit_bureau_a_1__n_records") + 1


# ------------------------------------------------------------------------------------- C8 (Giancarlo)
def _catalog(path, names, family):
    rows = [{c: "" for c in CATALOG_SCHEMA} | {"name": n, "role": "cobertura", "logical_type": "int", "source_table": family,
             "aggregation": "n_records", "window": "all", "temporal_rule": "r", "description": "d", "source_columns": "x"} for n in names]
    pl.DataFrame(rows, schema=CATALOG_SCHEMA).write_csv(path)
    return path


def test_c8_consolidates_three_families_with_literal_counts(tmp_path):
    paths = [
        _catalog(tmp_path / "a.csv", ["a__n_records", "a__source_present"], "a"),
        _catalog(tmp_path / "b.csv", ["b__n_records"], "b"),
        _catalog(tmp_path / "c.csv", ["c__n_records", "c__source_present", "c__x__sum__all"], "c"),
    ]
    out = c8_catalog.consolidate_catalogs(paths)
    assert out.height == 6
    assert out["name"].to_list() == sorted(out["name"].to_list())


def test_c8_rejects_a_catalog_with_a_different_schema(tmp_path):
    good = _catalog(tmp_path / "a.csv", ["a__n_records"], "a")
    bad = tmp_path / "b.csv"
    pl.DataFrame({"name": ["b__n_records"], "role": ["cobertura"]}).write_csv(bad)
    with pytest.raises(ValueError, match="missing columns"):
        c8_catalog.consolidate_catalogs([good, bad])
