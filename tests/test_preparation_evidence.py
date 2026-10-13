import polars as pl
import preparation_fixtures as fx

from dm_credit.preparation import applications_people_evidence as ev


def metrics():
    return ev.compute_evidence(fx.base(), fx.applprev_1(), fx.applprev_2(), fx.person_1(), fx.person_2())


def test_applprev_1_metrics_match_hand_computed_values():
    m = metrics()
    assert m["applprev_1_rows"] == 8
    assert m["applprev_1_cases"] == 4
    assert m["applprev_1_rows_after_decision"] == 2
    assert m["applprev_1_rows_null_creation"] == 1
    assert m["applprev_1_cases_after_filter"] == 3  # el caso 5 pierde toda su cobertura


def test_applprev_2_metrics_match_hand_computed_values():
    m = metrics()
    assert m["applprev_2_rows"] == 9
    assert m["applprev_2_orphans"] == 0
    assert m["applprev_2_rows_valid_parent"] == len(fx.VALID_APPLPREV_CHILDREN) == 6


def test_person_metrics_match_hand_computed_values():
    m = metrics()
    assert m["person_1_rows"] == 6
    assert m["person_1_cases"] == 5
    assert m["person_1_cases_with_main"] == 5
    assert m["person_1_employment_after_decision"] == 1
    assert m["person_1_birth_after_decision"] == 0
    assert m["person_2_rows"] == 3
    assert m["person_2_orphans"] == 0


def test_orphans_are_detected():
    orphan = fx.applprev_2().vstack(
        fx.applprev_2().head(1).with_columns(num_group1=pl.lit(99, dtype=pl.Int64))  # padre inexistente
    )
    m = ev.compute_evidence(fx.base(), fx.applprev_1(), orphan, fx.person_1(), fx.person_2())
    assert m["applprev_2_orphans"] == 1


def test_accepts_lazy_frames():
    lazy = ev.compute_evidence(fx.base().lazy(), fx.applprev_1().lazy(), fx.applprev_2().lazy(), fx.person_1().lazy(), fx.person_2().lazy())
    assert lazy == metrics()
