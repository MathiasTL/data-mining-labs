import re

import polars as pl
import pytest

import preparation_fixtures as fx
from dm_credit.baseline import RESERVED
from dm_credit.preparation.families import applications_people as ap

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(__[a-z0-9_]+){1,3}$")  # convención de nombres de A6

APP_CONFIG = ap.ApplicationsConfig(
    numeric={"credamount_590A": ("count", "min", "max", "mean", "sum")},
    categorical_distinct=("credtype_587L",),
    category_shares={"credtype_587L": ("CAL", "REL")},
    bool_cols=("isbidproduct_390L",),
    date_cols=("approvaldate_319D",),
    children_category_shares={"conts_type_509L": ("PHONE", "EMAIL")},
)
PEOPLE_CONFIG = ap.PeopleConfig(
    main_numeric=("mainoccupationinc_384A",),
    main_categorical=("sex_738L",),
    birth_col="birth_259D",
    employment_col="empl_employedfrom_271D",
    contact_category_shares={"role_1084L": ("AA", "BB")},
    child_category_shares={"empls_economicalst_849M": ("X", "Y")},
)


@pytest.fixture
def apps():
    return ap.aggregate_applications(fx.base(), fx.applprev_1(), fx.applprev_2(), APP_CONFIG)


@pytest.fixture
def people():
    return ap.aggregate_people(fx.base(), fx.person_1(), fx.person_2(), PEOPLE_CONFIG)


def get(df, case, col):
    return df.filter(pl.col("case_id") == case)[col].item()


# ------------------------------------------------------------ solicitudes previas
def test_only_cases_with_a_valid_application_are_returned(apps):
    # caso 3 y 6 sin historial; caso 5 solo tiene una solicitud posterior a la decisión
    assert apps["case_id"].to_list() == [1, 2, 4]
    assert apps["case_id"].is_unique().all()


def test_counts_use_only_rows_available_at_decision(apps):
    assert [get(apps, c, "applprev_1__n_records") for c in (1, 2, 4)] == [3, 1, 1]
    assert apps["applprev_1__source_present"].to_list() == [True, True, True]


def test_same_day_is_kept_and_null_creation_date_is_dropped(apps):
    assert get(apps, 4, "applprev_1__n_records") == 1  # g0 mismo día se conserva; g1 con fecha nula se descarta
    assert get(apps, 4, "applprev_1__days_since_last_creationdate_885d__all") == 0


def test_creation_recency_and_age(apps):
    c = "applprev_1__days_since_last_creationdate_885d__all"
    f = "applprev_1__days_since_first_creationdate_885d__all"
    assert (get(apps, 1, c), get(apps, 1, f)) == (1, 143)
    assert (get(apps, 2, c), get(apps, 2, f)) == (121, 121)


def test_numeric_aggregates_ignore_excluded_rows(apps):
    p = "applprev_1__credamount_590a__"
    assert get(apps, 1, p + "count__all") == 3
    assert get(apps, 1, p + "min__all") == 1000.0 and get(apps, 1, p + "max__all") == 3000.0
    assert get(apps, 1, p + "mean__all") == 2000.0 and get(apps, 1, p + "sum__all") == 6000.0
    # caso 2: la fila posterior (700) no entra
    assert get(apps, 2, p + "sum__all") == 500.0


def test_category_distinct_and_shares(apps):
    assert get(apps, 1, "applprev_1__credtype_587l__n_distinct__all") == 2
    assert get(apps, 1, "applprev_1__credtype_587l__share_cal__all") == pytest.approx(2 / 3)
    assert get(apps, 1, "applprev_1__credtype_587l__share_rel__all") == pytest.approx(1 / 3)
    assert get(apps, 4, "applprev_1__credtype_587l__share_rel__all") == 1.0


def test_bool_share_is_over_valid_observations_only(apps):
    b = "applprev_1__isbidproduct_390l__share_true__all"
    assert get(apps, 1, b) == 0.5  # True, False y un nulo que no cuenta
    assert get(apps, 2, b) == 1.0
    assert get(apps, 4, b) == 0.0


def test_future_dates_become_null_not_features(apps):
    a = "applprev_1__days_since_last_approvaldate_319d__all"
    assert get(apps, 1, a) == 138  # la aprobación del 2020-06-20 se anula
    assert get(apps, 2, a) == 112
    assert get(apps, 4, a) is None
    assert not [c for c in apps.columns if "future" in c]  # contar fechas futuras sería fuga


def test_children_use_only_valid_parents(apps):
    assert [get(apps, c, "applprev_2__n_records") for c in (1, 2, 4)] == [4, 1, 1]
    assert get(apps, 1, "applprev_2__children_per_parent_mean__all") == pytest.approx(4 / 3)
    assert get(apps, 1, "applprev_2__children_per_parent_max__all") == 2


def test_children_category_shares(apps):
    p = "applprev_2__conts_type_509l__share_"
    assert get(apps, 1, p + "phone__all") == 0.75 and get(apps, 1, p + "email__all") == 0.25


def test_valid_parent_without_children_has_zero_children_and_null_stats():
    out = ap.aggregate_applications(fx.base(), fx.applprev_1(), fx.applprev_2().filter(pl.col("case_id") != 2), APP_CONFIG)
    assert get(out, 2, "applprev_2__n_records") == 0
    assert get(out, 2, "applprev_2__children_per_parent_mean__all") == 0.0  # un padre válido sin hijos tiene 0 hijos
    assert get(out, 2, "applprev_2__children_per_parent_max__all") == 0


def test_exclusions_are_reported_for_the_manifest_only():
    res = ap.aggregate_applications_with_exclusions(fx.base(), fx.applprev_1(), fx.applprev_2(), APP_CONFIG)
    ex = res.exclusions
    assert ex["applprev_1_rows_after_decision"] == 2
    assert ex["applprev_1_rows_null_creation"] == 1
    assert ex["applprev_2_rows_invalid_parent"] == 3
    assert ex["approvaldate_319D_values_after_decision"] == 1  # solo entre filas válidas


def test_application_feature_names_follow_the_convention(apps):
    bad = [c for c in apps.columns if c != "case_id" and not NAME_RE.match(c)]
    assert not bad
    assert not set(apps.columns) & set(RESERVED) - {"case_id"}
    assert len(set(apps.columns)) == len(apps.columns)


def test_missing_configured_column_is_an_error():
    cfg = ap.ApplicationsConfig(numeric={"does_not_exist": ("sum",)}, categorical_distinct=(), bool_cols=(), date_cols=())
    with pytest.raises(ValueError, match="does_not_exist"):
        ap.aggregate_applications(fx.base(), fx.applprev_1(), fx.applprev_2(), cfg)


def test_unknown_aggregation_is_an_error():
    cfg = ap.ApplicationsConfig(numeric={"credamount_590A": ("median",)}, categorical_distinct=(), bool_cols=(), date_cols=())
    with pytest.raises(ValueError, match="median"):
        ap.aggregate_applications(fx.base(), fx.applprev_1(), fx.applprev_2(), cfg)


def test_duplicated_parent_key_is_rejected():
    dup = pl.concat([fx.applprev_1(), fx.applprev_1().head(1)])
    with pytest.raises(ValueError, match="num_group1"):
        ap.aggregate_applications(fx.base(), dup, fx.applprev_2(), APP_CONFIG)


def test_case_missing_from_base_is_rejected():
    orphan = fx.applprev_1().with_columns(case_id=pl.when(pl.col("case_id") == 1).then(99).otherwise(pl.col("case_id")))
    with pytest.raises(ValueError, match="base"):
        ap.aggregate_applications(fx.base(), orphan, fx.applprev_2(), APP_CONFIG)


def test_aggregation_does_not_depend_on_row_order():
    a = ap.aggregate_applications(fx.base(), fx.applprev_1(), fx.applprev_2(), APP_CONFIG)
    b = ap.aggregate_applications(fx.base(), fx.applprev_1().reverse(), fx.applprev_2().reverse(), APP_CONFIG)
    assert a.equals(b)


# ----------------------------------------------------------------------- personas
def test_people_one_row_per_covered_case(people):
    assert people["case_id"].to_list() == [1, 2, 3, 4, 5]  # el caso 6 no tiene personas
    assert people["case_id"].is_unique().all()


def test_people_coverage_and_contacts(people):
    assert get(people, 1, "person_1__n_records") == 2
    assert get(people, 1, "person_1__n_contacts__all") == 1
    assert get(people, 2, "person_1__n_contacts__all") == 0
    assert people["person_1__source_present"].to_list() == [True] * 5


def test_main_person_attributes(people):
    assert get(people, 1, "person_1__mainoccupationinc_384a__main__all") == 1000.0
    assert get(people, 3, "person_1__sex_738l__main__all") == "F"


def test_age_in_completed_years_at_decision(people):
    a = "person_1__birth_259d__age_years__all"
    assert [get(people, c, a) for c in (1, 2, 3, 4, 5)] == [35, 30, 44, 40, 24]


def test_employment_days_nulls_future_and_missing(people):
    e = "person_1__empl_employedfrom_271d__days_since__all"
    assert get(people, 1, e) == 517
    assert get(people, 2, e) is None  # 2020-07-01 es posterior a la decisión
    assert get(people, 3, e) == 758
    assert get(people, 4, e) is None
    assert get(people, 5, e) == 1186


def test_contact_shares_use_non_main_rows_only(people):
    assert get(people, 1, "person_1__role_1084l__share_bb__all") == 1.0
    assert get(people, 1, "person_1__role_1084l__share_aa__all") == 0.0
    assert get(people, 2, "person_1__role_1084l__share_bb__all") is None  # sin contactos


def test_person_2_counts_and_shares(people):
    assert [get(people, c, "person_2__n_records") for c in (1, 2, 3)] == [2, 0, 1]
    assert get(people, 1, "person_2__empls_economicalst_849m__share_x__all") == 0.5
    assert get(people, 3, "person_2__empls_economicalst_849m__share_y__all") == 0.0
    assert get(people, 2, "person_2__empls_economicalst_849m__share_x__all") is None


def test_people_exclusions_for_the_manifest():
    res = ap.aggregate_people_with_exclusions(fx.base(), fx.person_1(), fx.person_2(), PEOPLE_CONFIG)
    assert res.exclusions["empl_employedfrom_271D_values_after_decision"] == 1
    assert res.exclusions["birth_259D_values_after_decision"] == 0


def test_people_feature_names_follow_the_convention(people):
    bad = [c for c in people.columns if c != "case_id" and not NAME_RE.match(c)]
    assert not bad


def test_person_without_main_row_is_an_error():
    no_main = fx.person_1().filter(~((pl.col("case_id") == 1) & (pl.col("num_group1") == 0)))  # el caso 1 conserva solo un contacto
    with pytest.raises(ValueError, match="num_group1 = 0"):
        ap.aggregate_people(fx.base(), no_main, fx.person_2(), PEOPLE_CONFIG)


def test_parents_without_children_count_in_the_denominator():
    # caso 1: padres (1,0) con 2 hijos, (1,1) con 1, (1,2) sin hijos -> 3 hijos entre 3 padres válidos
    kids = fx.applprev_2().filter(~((pl.col("case_id") == 1) & (pl.col("num_group1") == 2)))
    out = ap.aggregate_applications(fx.base(), fx.applprev_1(), kids, APP_CONFIG)
    assert get(out, 1, "applprev_2__n_records") == 3
    assert get(out, 1, "applprev_2__children_per_parent_mean__all") == 1.0
    assert get(out, 1, "applprev_2__children_per_parent_max__all") == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"categorical_distinct": ("credtype_587L", "credtype_587L")},
        {"bool_cols": ("isbidproduct_390L", "isbidproduct_390L")},
        {"date_cols": ("approvaldate_319D", "approvaldate_319D")},
        {"date_cols": ("creationdate_885D",)},
        {"category_shares": {"credtype_587L": ("CAL", "cal")}},
        {"category_shares": {"credtype_587L": ("A-B", "A_B")}},
        {"numeric": {"credtype_587L": ("count",)}, "categorical_distinct": ("credtype_587L",)},
    ],
)
def test_invalid_applications_config_is_rejected_up_front(kwargs):
    base = dict(numeric={"credamount_590A": ("count",)}, categorical_distinct=(), category_shares={}, bool_cols=(), date_cols=(), children_category_shares={})
    cfg = ap.ApplicationsConfig(**{**base, **kwargs})
    with pytest.raises(ValueError, match="config"):
        ap.aggregate_applications(fx.base(), fx.applprev_1(), fx.applprev_2(), cfg)


def test_invalid_people_config_is_rejected_up_front():
    cfg = ap.PeopleConfig(main_numeric=("mainoccupationinc_384A",), main_categorical=("mainoccupationinc_384A",), contact_category_shares={}, child_category_shares={})
    with pytest.raises(ValueError, match="config"):
        ap.aggregate_people(fx.base(), fx.person_1(), fx.person_2(), cfg)


def test_null_decision_date_is_counted_in_exclusions():
    base = fx.base().with_columns(date_decision=pl.when(pl.col("case_id") == 2).then(None).otherwise(pl.col("date_decision")))
    res = ap.aggregate_applications_with_exclusions(base, fx.applprev_1(), fx.applprev_2(), APP_CONFIG)
    assert 2 not in res.features["case_id"].to_list()
    assert res.exclusions["applprev_1_rows_null_decision"] == 2  # las dos filas del caso 2
    pres = ap.aggregate_people_with_exclusions(base, fx.person_1(), fx.person_2(), PEOPLE_CONFIG)
    assert pres.exclusions["person_1_rows_null_decision"] == 1


def test_child_tables_have_their_own_source_present(apps, people):
    # cobertura por fuente física: un padre válido sin hijos no tiene cobertura en la tabla hija
    assert apps["applprev_2__source_present"].to_list() == [True, True, True]
    no_kids = ap.aggregate_applications(fx.base(), fx.applprev_1(), fx.applprev_2().filter(pl.col("case_id") != 2), APP_CONFIG)
    assert get(no_kids, 2, "applprev_2__source_present") is False
    assert get(no_kids, 1, "applprev_2__source_present") is True
    assert [get(people, c, "person_2__source_present") for c in (1, 2, 3, 4, 5)] == [True, False, True, False, False]
