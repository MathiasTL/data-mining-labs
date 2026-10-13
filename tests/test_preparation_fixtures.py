import polars as pl

import preparation_fixtures as fx


def keys(df, cols):
    return set(map(tuple, df.select(cols).rows()))


def test_primary_keys_are_unique():
    assert fx.applprev_1().select(["case_id", "num_group1"]).is_duplicated().sum() == 0
    assert fx.applprev_2().select(["case_id", "num_group1", "num_group2"]).is_duplicated().sum() == 0
    assert fx.person_1().select(["case_id", "num_group1"]).is_duplicated().sum() == 0
    assert fx.person_2().select(["case_id", "num_group1", "num_group2"]).is_duplicated().sum() == 0


def test_no_orphans_like_the_real_data():
    # en los datos reales no hay hijos sin padre (cero huérfanos en applprev_2 y person_2)
    p1 = keys(fx.applprev_1(), ["case_id", "num_group1"])
    assert {k[:2] for k in keys(fx.applprev_2(), ["case_id", "num_group1", "num_group2"])} <= p1
    pp = keys(fx.person_1(), ["case_id", "num_group1"])
    assert {k[:2] for k in keys(fx.person_2(), ["case_id", "num_group1", "num_group2"])} <= pp
    assert set(fx.applprev_1()["case_id"]) <= set(fx.base()["case_id"])


def test_hand_computed_temporal_expectations():
    a1 = fx.applprev_1().with_columns(pl.col("creationdate_885D").str.to_date().alias("d"))
    decision = pl.lit(fx.DECISION).str.to_date()
    valid = a1.filter(pl.col("d") <= decision)  # nulos quedan fuera: comparación nula no es verdadera
    assert keys(valid, ["case_id", "num_group1"]) == fx.VALID_APPLPREV_PARENTS
    assert a1.filter(pl.col("d") > decision).height == 2  # eventos posteriores
    assert a1["creationdate_885D"].null_count() == 1
    assert a1.filter(pl.col("d") == decision).height == 1  # borde: mismo día


def test_children_split_is_consistent():
    children = keys(fx.applprev_2(), ["case_id", "num_group1", "num_group2"])
    assert children == fx.VALID_APPLPREV_CHILDREN | fx.INVALID_APPLPREV_CHILDREN
    assert not fx.VALID_APPLPREV_CHILDREN & fx.INVALID_APPLPREV_CHILDREN
    # todo hijo válido tiene padre válido; todo hijo inválido tiene padre inválido
    assert all(k[:2] in fx.VALID_APPLPREV_PARENTS for k in fx.VALID_APPLPREV_CHILDREN)
    assert all(k[:2] not in fx.VALID_APPLPREV_PARENTS for k in fx.INVALID_APPLPREV_CHILDREN)


def test_coverage_edge_cases_exist():
    base_ids = set(fx.base()["case_id"])
    assert base_ids - set(fx.applprev_1()["case_id"]) == {3, 6}  # sin historial
    assert base_ids - set(fx.person_1()["case_id"]) == {6}  # sin personas
    # caso 5: solo solicitudes posteriores, pierde toda su cobertura tras el filtro
    assert {k[0] for k in fx.VALID_APPLPREV_PARENTS} == {1, 2, 4}


def test_person_employment_after_decision_is_present():
    p = fx.person_1().with_columns(pl.col("empl_employedfrom_271D").str.to_date().alias("d"))
    assert p.filter(pl.col("d") > pl.lit(fx.DECISION).str.to_date())["case_id"].to_list() == [2]
