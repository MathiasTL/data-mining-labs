"""E4 portable: reconstrucción completa en orden A5, medición sin /proc y lectura del time de macOS."""

import os

import pytest

from dm_credit.preparation import e4_rebuild as rb
from dm_credit.preparation.build import BUILDERS


def test_all_jobs_cover_every_family_and_split_once():
    jobs = rb.all_jobs()
    assert sorted(jobs) == sorted((f, s) for f in BUILDERS for s in ("train", "oot", "test"))
    assert len(jobs) == len(set(jobs)) == 36


def test_all_jobs_follow_the_backlog_order():
    jobs = rb.all_jobs()
    assert all(s == "test" for _, s in jobs[:12])  # primero la prueba pequeña de cada familia
    heavy = [f for f, s in jobs[12:]]
    order = list(dict.fromkeys(heavy))
    assert order[:2] == ["static_0", "static_cb_0"]
    assert order[-1] == "credit_bureau_a"  # bureau A al final
    assert order.index("applprev") < order.index("credit_bureau_b") < order.index("tax_registry_a_1")


def test_family_tables_never_include_bureau_a2():
    assert rb.family_tables("credit_bureau_a") == ("credit_bureau_a_1",)
    assert rb.family_tables("applprev") == ("applprev_1", "applprev_2")
    assert rb.family_tables("static_0") == ("static_0",)


def test_parse_bsd_time_output():
    log = "algo del builder\n       14.05 real        12.00 user  1.00 sys\n  1951444992  maximum resident set size\n"
    assert rb.parse_bsd_time(log) == {"seconds": 14.05, "peak_rss_kib": 1951444992 // 1024}


def test_parse_bsd_time_rejects_missing_metrics():
    with pytest.raises(ValueError):
        rb.parse_bsd_time("sin métricas")


def test_process_tree_rss_measures_the_current_process():
    assert rb.process_tree_rss(os.getpid()) > 0


def test_available_memory_is_positive():
    assert rb.available_kib() > 0
