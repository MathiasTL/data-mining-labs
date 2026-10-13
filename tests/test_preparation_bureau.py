import json
import math

import polars as pl
import pytest

import preparation_bureau_artifacts as bart
import preparation_bureau_fixtures as bfx
from dm_credit import io
from dm_credit.preparation import build
from dm_credit.preparation import contracts as ct
from dm_credit.preparation.families import bureau
from dm_credit.preparation.families.applications_people_run import assert_same_feature_schema


def plan(family, *, approve_a2=False, shares=bfx.CATEGORY_SHARES):
    shares = shares if family == "credit_bureau_a" else None
    return bureau.plan_from_selection(family, bfx.selection(approve_a2=approve_a2), category_shares=shares)


def batches_of(frame, size):
    return [frame.slice(i, size) for i in range(0, frame.height, size)] or [frame]


def aggregate(family, *, size=1000, approve_a2=False, base=None):
    p = plan(family, approve_a2=approve_a2)
    tables = {t.table: batches_of(bfx.TABLES[t.table](), size) for t in p.tables}
    return bureau.aggregate_family(bfx.base() if base is None else base, p, tables)


def assert_columns(features, expected):
    for name, values in expected.items():
        got = features[name].to_list()
        assert len(got) == len(values), name
        for g, v in zip(got, values):
            assert (g is None and v is None) or (g is not None and v is not None and math.isclose(g, v, rel_tol=1e-12)), (name, got)


# ----------------------------------------------------------------------------- pure core
def test_bureau_a_hand_computed_aggregates_and_exclusions():
    result = aggregate("credit_bureau_a")
    assert result.features["case_id"].to_list() == bfx.CASE_IDS  # one row per base case, 99 ignored
    assert_columns(result.features, bfx.EXPECTED_A1)
    assert result.exclusions == bfx.EXPECTED_A1_EXCLUSIONS


def test_bureau_b_children_need_a_temporally_valid_parent():
    result = aggregate("credit_bureau_b")
    assert result.features["case_id"].to_list() == bfx.CASE_IDS
    assert_columns(result.features, bfx.EXPECTED_B)
    assert result.exclusions == bfx.EXPECTED_B_EXCLUSIONS


def test_blocked_and_excluded_columns_never_produce_features():
    names = [*aggregate("credit_bureau_a").features.columns, *aggregate("credit_bureau_b").features.columns]
    for column in bfx.BLOCKED_OR_EXCLUDED:
        assert not any(f"__{column.lower()}__" in n for n in names), column
    # anchors filter rows; they are not predictors
    for anchor in ("lastupdate_1112d", "lastupdate_388d", "lastupdate_260d", "pmts_date_1107d"):
        assert not any(anchor in n for n in names)


def test_blocked_columns_are_not_even_read():
    p = plan("credit_bureau_a")
    read = {c for t in p.tables for c in t.columns}
    assert not read & bfx.BLOCKED_OR_EXCLUDED
    assert [t.table for t in p.tables] == ["credit_bureau_a_1"]
    assert "credit_bureau_a_2" in p.skipped  # no approved anchor: never read


def test_real_b5_selection_only_includes_allowed_actions():
    selection = bureau.load_selection()
    allowed = selection.filter(pl.col("action").is_in(bureau.INCLUDED_ACTIONS))
    for family in bureau.FAMILIES:
        p = bureau.plan_from_selection(family, selection)
        used = {(t.table, c) for t in p.tables for c in (*t.numeric, *t.bools, *t.dates, *t.categories)}
        assert used <= set(zip(allowed["family"], allowed["column"]))
        # without an approved vocabulary, categorical candidates are deferred, not invented
        assert all(not t.categories for t in p.tables)
    assert "credit_bureau_a_2" in bureau.plan_from_selection("credit_bureau_a", selection).skipped


def test_approving_a2_at_g2_needs_only_a_selection_change():
    result = aggregate("credit_bureau_a", approve_a2=True)
    assert_columns(result.features, bfx.EXPECTED_A1 | bfx.EXPECTED_A2_APPROVED)
    assert {k: result.exclusions[k] for k in bfx.EXPECTED_A2_EXCLUSIONS} == bfx.EXPECTED_A2_EXCLUSIONS


@pytest.mark.parametrize("family", ["credit_bureau_a", "credit_bureau_b"])
@pytest.mark.parametrize("size", [1, 2, 3])
def test_batch_boundaries_do_not_change_the_result(family, size):
    one = aggregate(family, approve_a2=True)
    split = aggregate(family, size=size, approve_a2=True)
    assert split.features.equals(one.features) or _close(split.features, one.features)
    assert split.exclusions == one.exclusions


def _close(a, b):
    assert a.columns == b.columns and a.schema == b.schema
    for c in a.columns:
        if a.schema[c] == pl.Float64:
            for x, y in zip(a[c].to_list(), b[c].to_list()):
                assert (x is None and y is None) or math.isclose(x, y, rel_tol=1e-9), c
        else:
            assert a[c].to_list() == b[c].to_list(), c
    return True


def test_merge_is_order_independent():
    p = plan("credit_bureau_b").tables[0]
    decision = bureau._decision(bfx.base())
    rows = bfx.credit_bureau_b_1()
    parts = [bureau.process_batches([rows.slice(i, 1)], p, decision, None).partial for i in range(rows.height)]
    assert bureau.merge_partials(parts, p).equals(bureau.merge_partials(parts[::-1], p))


def test_std_is_stable_for_large_amounts_across_batches():
    p = plan("credit_bureau_b").tables[0]
    rows = pl.DataFrame(
        {
            "case_id": [1, 1, 1], "num_group1": [0, 1, 2], "lastupdate_260D": ["2020-01-01"] * 3,
            "amount_1115A": [1e9, 1e9 + 1, 1e9 + 2], "contractdate_551D": [None] * 3, "isbidproduct_1L": [None] * 3,
        },
        schema_overrides={"contractdate_551D": pl.String, "isbidproduct_1L": pl.Boolean},
    )
    decision = bureau._decision(bfx.base())
    state = bureau.process_batches([rows.slice(i, 1) for i in range(3)], p, decision, None)
    features = bureau.finalize_table(state.partial, p, decision)
    assert features.filter(pl.col("case_id") == 1)["credit_bureau_b_1__amount_1115a__std__all"].item() == pytest.approx(1.0)


def test_empty_and_null_typed_input_returns_the_full_schema():
    p = plan("credit_bureau_b")
    full = aggregate("credit_bureau_b")
    empty = {t.table: [pl.DataFrame(schema={c: pl.Null for c in bfx.TABLES[t.table]().columns})] for t in p.tables}
    result = bureau.aggregate_family(bfx.base(), p, empty)
    assert result.features.schema == full.features.schema
    assert result.features["credit_bureau_b_1__n_records"].to_list() == [0] * 5
    assert not result.features["credit_bureau_b_2__source_present"].any()
    no_batches = bureau.aggregate_family(bfx.base(), p, {})
    assert no_batches.features.equals(result.features)
    assert set(no_batches.exclusions) == set(full.exclusions) and not any(no_batches.exclusions.values())


def test_category_shares_for_a_blocked_column_are_rejected():
    with pytest.raises(ValueError, match="not included"):
        plan("credit_bureau_a", shares={"credit_bureau_a_1": {"financialinstitution_382M": ("X",)}})


def test_unknown_anchor_in_selection_is_an_error():
    sel = bfx.selection().with_columns(
        action=pl.when(pl.col("column") == "credlmt_935A").then(pl.lit("referencia_as_of")).otherwise("action")
    )
    with pytest.raises(ValueError, match="no anchor rule"):
        bureau.plan_from_selection("credit_bureau_a", sel)


# ----------------------------------------------------------------------- builder and cache
@pytest.fixture
def data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    bart.write_sample_data(root)
    return root


@pytest.fixture
def selection_path(tmp_path):
    return bfx.write_selection(tmp_path / "B5.csv")


def base_for(split, cutoff=bfx.CUTOFF):
    b = bfx.base().select("case_id", "WEEK_NUM", "date_decision")
    if split == "test":
        return b.with_columns(pl.col("case_id") + bfx.TEST_OFFSET)
    return b.filter(pl.col("WEEK_NUM") < cutoff) if split == "train" else b.filter(pl.col("WEEK_NUM") >= cutoff)


def build_a(split, out, selection_path, **kw):
    return bureau.build_credit_bureau_a_features(
        base_for(split), split, cutoff_week=bfx.CUTOFF, output_dir=out, selection_path=selection_path,
        category_shares=bfx.CATEGORY_SHARES, **kw,
    )


def test_shards_and_row_groups_are_streamed(data):
    import pyarrow.parquet as pq

    shards = io.shard_paths("train", "credit_bureau_a_1")
    assert [p.name for p in shards] == ["train_credit_bureau_a_1_0.parquet", "train_credit_bureau_a_1_1.parquet"]
    assert len(io.shard_paths("test", "credit_bureau_a_1")) == 3  # discovered by pattern, counts differ
    assert sum(pq.ParquetFile(p).num_row_groups for p in shards) > len(shards)


@pytest.mark.parametrize("split", ["train", "oot"])
def test_builder_matches_the_pure_core(data, tmp_path, selection_path, split):
    art = build_a(split, tmp_path / "out", selection_path)
    expected = aggregate("credit_bureau_a").features.join(base_for(split).select("case_id"), on="case_id", how="semi")
    assert _close(pl.read_parquet(art.data_path), expected)
    assert art.case_count == base_for(split).height and art.row_count == art.case_count


def test_manifest_records_rules_inputs_and_skipped_tables(data, tmp_path, selection_path):
    art = build_a("train", tmp_path / "out", selection_path)
    m = json.loads(art.manifest_path.read_text())
    assert m["rules_version"] == bureau.RULES_VERSION and m["inputs"]["cutoff_week"] == str(bfx.CUTOFF)
    assert "train_credit_bureau_a_1_1.parquet" in m["inputs"]["shards"] and "a_2" not in m["inputs"]["shards"]
    assert "skipped_table:credit_bureau_a_2" in m["inputs"]
    assert m["exclusions"]["credit_bureau_a_1_rows_after_decision"] == 1  # train cases only: (1,2)
    roles = {f["name"]: f["role"] for f in m["features"]}
    assert roles["credit_bureau_a_1__n_records"] == "cobertura" and roles["credit_bureau_a_1__credlmt_935a__std__all"] == "predictor"


def test_bureau_b_builder_and_dispatch(data, tmp_path, selection_path):
    art = build.build_family_features(
        "credit_bureau_b", base_for("train"), "train", cutoff_week=bfx.CUTOFF, output_dir=tmp_path / "b", selection_path=selection_path
    )
    assert isinstance(art, ct.FamilyFeatureArtifact) and art.family == "credit_bureau_b"
    got = pl.read_parquet(art.data_path)
    assert got["credit_bureau_b_2__n_records"].to_list() == [1, 2]
    assert {"credit_bureau_a", "credit_bureau_b"} <= set(build.BUILDERS)


def test_identical_feature_set_across_train_oot_and_test(data, tmp_path, selection_path):
    arts = bart.build_small_artifacts(tmp_path / "out", selection_path)
    assert len(arts) == 6
    assert_same_feature_schema(arts)
    for a in arts:
        assert pl.read_parquet(a.data_path).height == a.case_count  # every base case of the split
    catalog = ct.feature_catalog(arts)
    assert catalog["name"].n_unique() == catalog.height and "credit_bureau_b_2__n_records" in catalog["name"].to_list()


def test_test_split_with_empty_null_typed_shard(data, tmp_path, selection_path):
    te = data / "parquet_files" / "test"
    pl.DataFrame(schema={c: pl.Null for c in bfx.credit_bureau_b_1().columns}).write_parquet(te / "test_credit_bureau_b_1.parquet")
    art = bureau.build_credit_bureau_b_features(
        base_for("test"), "test", cutoff_week=bfx.CUTOFF, output_dir=tmp_path / "t", selection_path=selection_path
    )
    got = pl.read_parquet(art.data_path)
    assert got.height == 5 and got["credit_bureau_b_1__n_records"].sum() == 0
    assert m_exclusion(art, "credit_bureau_b_2_rows_invalid_parent") == 6  # no valid parent at all (other two: null and after)


def m_exclusion(art, key):
    return json.loads(art.manifest_path.read_text())["exclusions"][key]


def test_cache_resume_gives_identical_output(data, tmp_path, selection_path, monkeypatch):
    fresh = build_a("train", tmp_path / "fresh", selection_path)
    cache = tmp_path / "cache"
    real = io.iter_row_groups

    def fail_on_second_shard(path, columns=None):
        if path.name.endswith("_1.parquet"):
            raise RuntimeError("interrupted")
        return real(path, columns)

    monkeypatch.setattr(io, "iter_row_groups", fail_on_second_shard)
    with pytest.raises(RuntimeError, match="interrupted"):
        build_a("train", tmp_path / "resumed", selection_path, cache_dir=cache)
    assert not (tmp_path / "resumed" / "credit_bureau_a__train.parquet").exists()
    assert len(list((cache / "credit_bureau_a_1").glob("*_0.*.json"))) == 1  # first shard checkpointed

    def fail_on_first_shard(path, columns=None):
        if path.name.endswith("_0.parquet"):
            raise AssertionError("checkpointed shard must not be read again")
        return real(path, columns)

    monkeypatch.setattr(io, "iter_row_groups", fail_on_first_shard)
    resumed = build_a("train", tmp_path / "resumed", selection_path, cache_dir=cache)
    assert pl.read_parquet(resumed.data_path).equals(pl.read_parquet(fresh.data_path))
    assert json.loads(resumed.manifest_path.read_text()) == json.loads(fresh.manifest_path.read_text())


def test_cache_is_invalidated_by_base_or_corruption(data, tmp_path, selection_path, monkeypatch):
    cache = tmp_path / "cache"
    build_a("train", tmp_path / "a", selection_path, cache_dir=cache)
    calls = []
    real = io.iter_row_groups
    monkeypatch.setattr(io, "iter_row_groups", lambda path, columns=None: calls.append(path.name) or real(path, columns))
    build_a("oot", tmp_path / "b", selection_path, cache_dir=cache)  # other base: new key, shards are read
    assert len(calls) == 2
    calls.clear()
    for f in (cache / "credit_bureau_a_1").glob("*.partial.parquet"):
        f.write_bytes(b"corrupt")
    build_a("train", tmp_path / "c", selection_path, cache_dir=cache)
    assert len(calls) == 2


def test_failed_build_does_not_overwrite_a_valid_artifact(data, tmp_path, selection_path):
    out = tmp_path / "out"
    good = build_a("train", out, selection_path)
    before = good.data_path.read_bytes()
    with pytest.raises(ValueError):
        build_a("train", out, selection_path, included_actions=("candidato_G2", "referencia_as_of"))
    assert good.data_path.read_bytes() == before


def test_child_uses_cached_parent_keys_when_a2_is_approved(data, tmp_path, monkeypatch):
    sel = bfx.write_selection(tmp_path / "B5_g2.csv", approve_a2=True)
    cache = tmp_path / "cache"
    first = build_a("train", tmp_path / "a", sel, cache_dir=cache)
    got = pl.read_parquet(first.data_path)
    assert got["credit_bureau_a_2__n_records"].to_list() == [2, 1]
    assert got["credit_bureau_a_2__pmts_overdue_1140a__sum__all"].to_list() == [40.0, 60.0]
    assert list((cache / "credit_bureau_a_1").glob("*.keys.parquet"))  # parent keys checkpointed with the parent

    def no_reads(path, columns=None):
        raise AssertionError("every shard is checkpointed")

    monkeypatch.setattr(io, "iter_row_groups", no_reads)
    again = build_a("train", tmp_path / "b", sel, cache_dir=cache)
    assert pl.read_parquet(again.data_path).equals(got)
