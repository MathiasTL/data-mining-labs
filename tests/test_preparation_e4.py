"""E4 helper contracts using only synthetic data; no audit or builders are run."""

import hashlib
from dataclasses import asdict, replace
from datetime import date
from types import SimpleNamespace

import polars as pl

from dm_credit import io
import pytest

from dm_credit.preparation import contracts, e4


@pytest.fixture
def parquet(tmp_path):
    def write(frame):
        path = tmp_path / "synthetic.parquet"
        frame.write_parquet(path)
        return path

    return write


@pytest.mark.parametrize(
    "keys,error",
    [
        ([1, None, 3], "contains nulls"),
        ([1, 1, 3], "not unique"),
        ([1, 2, 4], "claves/tipo"),
        ([3, 2, 1], "no ordenado"),
    ],
)
def test_validate_keys_rejects_invalid_keys(parquet, keys, error):
    path = parquet(pl.DataFrame({"case_id": keys}))
    with pytest.raises(ValueError, match=error):
        e4.validate_keys(
            path,
            pl.DataFrame({"case_id": [1, 2, 3]}),
            3,
            {"row_count": 3, "case_count": 3},
        )


def test_validate_keys_accepts_exact_set_with_unsorted_base(parquet):
    path = parquet(pl.DataFrame({"case_id": [1, 2, 3], "unused": [None] * 3}))
    e4.validate_keys(
        path, pl.DataFrame({"case_id": [3, 1, 2]}), 3, {"row_count": 3, "case_count": 3}
    )


@pytest.mark.parametrize("field", ["row_count", "case_count", "physical"])
def test_validate_keys_rejects_count_mismatch(parquet, field):
    manifest = {"row_count": 2, "case_count": 2}
    if field != "physical":
        manifest[field] = 3
    path = parquet(pl.DataFrame({"case_id": [1] if field == "physical" else [1, 2]}))
    with pytest.raises(
        ValueError, match="filas físicas" if field == "physical" else field
    ):
        e4.validate_keys(path, pl.DataFrame({"case_id": [1, 2]}), 2, manifest)


@pytest.mark.parametrize(
    "counts,present,error",
    [
        ([0, 2], [True, True], "source_present !="),
        ([0, 2], [False, False], "source_present !="),
        ([-1, 2], [False, True], "n_records negativo"),
        ([None, 2], [False, True], "cobertura nula"),
        ([0, 2], [None, True], "cobertura nula"),
        ([0.0, 2.0], [False, True], "tipos de cobertura"),
        ([0, 2], [0, 1], "tipos de cobertura"),
    ],
)
def test_coverage_metrics_rejects_inconsistent_coverage(
    parquet, counts, present, error
):
    path = parquet(
        pl.DataFrame({"source__n_records": counts, "source__source_present": present})
    )
    with pytest.raises(ValueError, match=error):
        e4.coverage_metrics(path, "source", 2)


def test_coverage_metrics_distinguishes_rows_and_cases(parquet):
    path = parquet(
        pl.DataFrame(
            {
                "source__n_records": [0, 2, 4, 0],
                "source__source_present": [False, True, True, False],
            }
        )
    )
    assert e4.coverage_metrics(path, "source", 4) == {
        "rows": 6,
        "cases": 2,
        "base_cases": 4,
        "cases_without_source": 2,
        "coverage_pct": 50.0,
    }


def test_coverage_metrics_empty_base(parquet):
    path = parquet(
        pl.DataFrame(
            schema={"source__n_records": pl.Int64, "source__source_present": pl.Boolean}
        )
    )
    assert e4.coverage_metrics(path, "source", 0) == {
        "rows": 0,
        "cases": 0,
        "base_cases": 0,
        "cases_without_source": 0,
        "coverage_pct": 0.0,
    }


@pytest.mark.parametrize(
    "rows,zero,error",
    [
        ([("person_1", "income", "inventada")], False, "desconocidas"),
        ([("person_1", "income", "aprobada")], True, "cero aprobaciones"),
        ([("person_1", "income", "aprobada")] * 2, False, "duplicadas"),
        ([(None, "income", "aprobada")], False, "nulas"),
        ([("person_1", None, "aprobada")], False, "nulas"),
        ([("person_1", "income", None)], False, "nulas"),
        ([("person_2", "income", "aprobada")], False, "faltan tablas"),
    ],
)
def test_load_decisions_rejects_invalid_g2(tmp_path, rows, zero, error):
    path = tmp_path / "g2.csv"
    pl.DataFrame(
        rows, schema=["family", "column", "g2_decision"], orient="row"
    ).write_csv(path)
    with pytest.raises(ValueError, match=error):
        e4.load_decisions(path, ("person_1",), zero=zero)


def test_load_decisions_accepts_approved_column_when_allowed(tmp_path):
    path = tmp_path / "g2.csv"
    frame = pl.DataFrame(
        {"family": ["person_1"], "column": ["income"], "g2_decision": ["aprobada"]}
    )
    frame.write_csv(path)
    assert e4.load_decisions(path, ("person_1",), zero=False).equals(frame)


@pytest.fixture
def definition():
    return contracts.FeatureDefinition(
        name="person_1__income__main__all",
        role="predictor",
        logical_type="float",
        source_table="person_1",
        source_columns=("income",),
        aggregation="main",
        window="all",
        temporal_rule="synthetic rule",
        description="Synthetic income",
    )


@pytest.mark.parametrize(
    "approved_table,approved_column", [("person_2", "income"), ("person_1", "other")]
)
def test_validate_definitions_rejects_unapproved_source(
    definition, approved_table, approved_column
):
    spec = e4.FamilySpec("person", "unused", "test", ("person_1",), "unused", "unused")
    decisions = pl.DataFrame(
        {
            "family": [approved_table],
            "column": [approved_column],
            "g2_decision": ["aprobada"],
        }
    )
    with pytest.raises(ValueError, match="columna no aprobada"):
        e4.validate_definitions(
            SimpleNamespace(feature_schema=(definition,)),
            pl.Schema({"case_id": pl.Int64, definition.name: pl.Float64}),
            spec,
            spec.tables,
            decisions,
            {definition.name: ("income",)},
        )


@pytest.mark.parametrize(
    "metric,unit",
    [
        ("applprev_1_rows_after_decision", "rows"),
        ("source_rows_total", "rows"),
        ("person_1_values_after_decision", "values"),
        ("b6_excluded", "columns"),
        ("g2_approved", "columns"),
        ("manual_review", "columns"),
        ("exclude_predictor", "columns"),
        ("not_selected_from_source", "columns"),
        ("unrecognized", "unknown"),
    ],
)
def test_exclusion_unit(metric, unit):
    assert e4.exclusion_unit(metric) == unit


@pytest.fixture
def reconciliation():
    coverage = [
        {
            "table": "applprev_1",
            "family": "applprev",
            "split": split,
            "status": "ok",
            "rows": 3,
            "cases": 2,
            "base_cases": 4,
        }
        for split in ("train", "oot")
    ]
    exclusions = [
        {"split": split, "family": "applprev", "metric": metric, "value": value}
        for split in ("train", "oot")
        for metric, value in [
            ("applprev_1_rows_null_creation", 1),
            ("applprev_1_rows_after_decision", 1),
            ("applprev_1_rows_null_decision", 0),
            ("applprev_1_values_after_decision", 100),
            ("exclude_predictor", 50),
        ]
    ]
    cardinality = pl.DataFrame(
        {
            "family": ["applprev_1"],
            "rows": [10],
            "covered_base_cases": [5],
            "orphan_rows": [0],
        }
    )
    return coverage, exclusions, cardinality


def test_reconcile_balances_only_disjoint_row_exclusions(reconciliation):
    coverage, exclusions, cardinality = reconciliation
    coverage.append({**coverage[0], "split": "test", "rows": 999, "cases": 999})
    assert e4.reconcile_cardinality(coverage, exclusions, cardinality) == [
        {
            "table": "applprev_1",
            "status": "ok",
            "error": None,
            "raw_rows": 10,
            "raw_cases": 5,
            "rows": 6,
            "cases": 4,
            "row_exclusions_disjoint": 4,
            "lost_cases": 1,
            "coverage_pct": 50.0,
        }
    ]


@pytest.mark.parametrize(
    "problem,error",
    [
        ("missing_split", "faltan coberturas"),
        ("unverified", "cobertura no comprobada"),
        ("overlap", "null_decision"),
        ("balance", "exclusiones disjuntas"),
        ("orphans", "fuera de base"),
        ("cases", "cobertura supera bruto"),
    ],
)
def test_reconcile_reports_invalid_balance(reconciliation, problem, error):
    coverage, exclusions, cardinality = reconciliation
    if problem == "missing_split":
        coverage.pop()
    elif problem == "unverified":
        coverage[0]["status"] = "invalid"
    elif problem == "overlap":
        exclusions[2]["value"] = 1
    elif problem == "balance":
        cardinality = cardinality.with_columns(pl.lit(11).alias("rows"))
    elif problem == "orphans":
        cardinality = cardinality.with_columns(pl.lit(1).alias("orphan_rows"))
    else:
        cardinality = cardinality.with_columns(pl.lit(3).alias("covered_base_cases"))
    (result,) = e4.reconcile_cardinality(coverage, exclusions, cardinality)
    assert result["status"] == "invalid"
    assert error in result["error"]


def test_reconcile_rejects_duplicate_cardinality(reconciliation):
    coverage, exclusions, cardinality = reconciliation
    with pytest.raises(ValueError, match="tablas duplicadas"):
        e4.reconcile_cardinality(coverage, exclusions, pl.concat([cardinality] * 2))


def test_compare_catalog_accepts_identical_definitions_across_splits(
    tmp_path, definition
):
    path = tmp_path / "catalog.csv"
    row = {**asdict(definition), "source_columns": "income"}
    pl.DataFrame([row], schema=contracts.CATALOG_SCHEMA).write_csv(path)
    artifacts = [
        SimpleNamespace(family="person", split=split, feature_schema=(definition,))
        for split in ("train", "oot", "test")
    ]
    e4.compare_catalog(artifacts, path)


@pytest.mark.parametrize(
    "problem,error",
    [
        ("duplicate", "catálogo duplicado"),
        ("changed", "catálogo completo difiere"),
        ("missing", "catálogo completo difiere"),
        ("split", "definición entre splits"),
        ("owner", "nombre compartido entre familias"),
    ],
)
def test_compare_catalog_rejects_divergence(tmp_path, definition, problem, error):
    path = tmp_path / "catalog.csv"
    row = {**asdict(definition), "source_columns": "income"}
    rows = [row]
    artifacts = [SimpleNamespace(family="person", feature_schema=(definition,))]
    if problem == "duplicate":
        rows *= 2
    elif problem == "changed":
        rows = [{**row, "description": "Altered"}]
    elif problem == "missing":
        rows = []
    elif problem == "split":
        artifacts.append(
            SimpleNamespace(
                family="person",
                feature_schema=(replace(definition, description="Altered"),),
            )
        )
    else:
        artifacts.append(SimpleNamespace(family="other", feature_schema=(definition,)))
    pl.DataFrame(rows, schema=contracts.CATALOG_SCHEMA).write_csv(path)
    with pytest.raises(ValueError, match=error):
        e4.compare_catalog(artifacts, path)


def test_sha256_file_hashes_all_chunks(tmp_path):
    path = tmp_path / "synthetic.bin"
    content = b"x" * (2**20 + 17) + b"end"
    path.write_bytes(content)
    assert e4.sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_base_fingerprint_is_canonical_and_excludes_target():
    base = pl.DataFrame(
        {
            "case_id": [2, 1],
            "WEEK_NUM": [4, 3],
            "date_decision": [date(2020, 1, 2), date(2020, 1, 1)],
            "target": [0, 1],
        }
    )
    expected = "case_id,WEEK_NUM,date_decision\n1,3,2020-01-01\n2,4,2020-01-02\n"
    assert e4.base_fingerprint(base) == hashlib.sha256(expected.encode()).hexdigest()
    assert e4.base_fingerprint(base) == e4.base_fingerprint(
        base.reverse().drop("target")
    )
    for column, value in [
        ("case_id", 10),
        ("WEEK_NUM", 10),
        ("date_decision", date(2021, 1, 1)),
    ]:
        assert e4.base_fingerprint(base) != e4.base_fingerprint(
            base.with_columns(pl.lit(value).alias(column))
        )


@pytest.mark.parametrize(
    "split,source_split", [("train", "train"), ("oot", "train"), ("test", "test")]
)
def test_shard_fingerprint_order_and_split_without_real_data(
    tmp_path, monkeypatch, split, source_split
):
    paths = [tmp_path / name for name in ("shard_2.parquet", "shard_10.parquet")]
    for path in paths:
        path.write_bytes(b"synthetic")
    calls = []

    def shard_paths(actual_split, table):
        calls.append((actual_split, table))
        return paths

    monkeypatch.setattr(e4.io, "shard_paths", shard_paths)
    parts = [io.file_fingerprint(p) for p in paths]  # contenido, no fecha
    assert e4.shard_fingerprint(split, ("synthetic",)) == ";".join(sorted(parts))
    assert e4.shard_fingerprint(split, ("synthetic",), ordered=True) == ";".join(parts)
    assert calls == [(source_split, "synthetic")] * 2


def test_shard_fingerprint_rejects_a2_before_any_io(monkeypatch):
    def forbidden(*args):
        pytest.fail("A2 must be rejected before looking up any shards")

    monkeypatch.setattr(e4.io, "shard_paths", forbidden)
    with pytest.raises(ValueError, match="no debe consultar shards"):
        e4.shard_fingerprint("train", ("synthetic", e4.A2))
