"""C4/C5 — Streaming adapters for the credit bureau families (design A5, interfaces G1).

Families (G1 point 2): ``credit_bureau_a`` = ``credit_bureau_a_1`` + ``credit_bureau_a_2`` and
``credit_bureau_b`` = ``credit_bureau_b_1`` + ``credit_bureau_b_2``. A family never joins another one.

Design:
    - The B5 selection (``results/preparation/B5_bureau_columns.csv``) drives everything. Only columns whose
      action is in ``INCLUDED_ACTIONS`` produce features; ``bloquear_hasta_G2`` and ``excluir`` columns are
      never read. ``referencia_as_of`` columns are temporal anchors (they filter rows, they are not predictors).
      Approving a blocked column at G2 only changes the selection file, not this code.
    - A table is processed only if at least one of its anchors (``ANCHORS``) is fully approved as
      ``referencia_as_of``; a child table also needs its parent. With the current B5 file ``credit_bureau_a_2``
      has no approved anchor (A5 D1/D6), so it is skipped and never read.
    - Rows are read one row group at a time (``io.iter_row_groups``) and filtered temporally BEFORE aggregating.
      A row is kept only if it has at least one non-null approved anchor and every non-null approved anchor
      satisfies its rule (conservative row-level reading of B5's per-side rule for ``credit_bureau_a_1``).
      Children are kept only if their parent ``(case_id, num_group1)`` is valid (semi-join, same family).
    - Each batch yields a mergeable partial per ``case_id`` (count, sum, M2, min, max, true and category
      counts); partials merge with the stable parallel-variance formula (A5 §3.2: M2, not sum of squares).
    - Checkpoint unit: the shard (A5 §3.6). The cache key is G1 §5: shard fingerprint + rules version (with the
      plan) + base fingerprint + cutoff. A resumed run skips shards with a valid checkpoint.
    - Output: one row per base ``case_id`` (cases without records get ``n_records = 0`` and
      ``source_present = False``), published atomically with ``contracts.write_family_features``.

Aggregations (G1 decision 4): count, min, max, mean, sum and std for numerics; proportion for booleans;
shares only for approved categories (``category_shares``); age/recency in days for ``transformar_as_of``
dates. No ``n_distinct``, median or percentiles in bureau.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping

import polars as pl

from dm_credit import io
from dm_credit.preparation import coercion, contracts
from dm_credit.preparation.families import applications_people as ap

KEY = contracts.KEY
PARENT_KEY = (KEY, "num_group1")
WINDOW = "all"
RULES_VERSION = "C4-C5-v1"
DEFAULT_SELECTION_PATH = Path(__file__).resolve().parents[4] / "results" / "preparation" / "B5_bureau_columns.csv"
INCLUDED_ACTIONS = ("candidato_G2", "transformar_as_of")
ANCHOR_ACTION = "referencia_as_of"
RESERVED = {KEY, "num_group1", "num_group2"}
NUMERIC_AGGS = ("count", "min", "max", "mean", "sum", "std")
SELECTION_COLUMNS = {"family", "column", "dtype", "action"}

FAMILIES = {
    "credit_bureau_a": ("credit_bureau_a_1", "credit_bureau_a_2"),
    "credit_bureau_b": ("credit_bureau_b_1", "credit_bureau_b_2"),
}
PARENTS = {"credit_bureau_a_2": "credit_bureau_a_1", "credit_bureau_b_2": "credit_bureau_b_1"}


@dataclass(frozen=True)
class Anchor:
    """Temporal reference of a row. ``day``: date <= decision. ``month``: (year, month) strictly before the
    decision month (A5 D1). Pairing of ``credit_bureau_a_2`` follows B5 (shared null rates), not A5 §2.6."""

    kind: str  # "day" | "month"
    columns: tuple[str, ...]

    def ok(self, decision: pl.Expr) -> pl.Expr:
        if self.kind == "day":
            return pl.col(self.columns[0]) <= decision
        year, month = (pl.col(c).cast(pl.Int32) for c in self.columns)
        return pl.date(year, month, 1) < decision.dt.month_start()

    def present(self) -> pl.Expr:
        return pl.all_horizontal([pl.col(c).is_not_null() for c in self.columns])


ANCHORS = {
    "credit_bureau_a_1": (Anchor("day", ("lastupdate_1112D",)), Anchor("day", ("lastupdate_388D",))),
    "credit_bureau_a_2": (
        Anchor("month", ("pmts_year_1139T", "pmts_month_158T")),
        Anchor("month", ("pmts_year_507T", "pmts_month_706T")),
    ),
    "credit_bureau_b_1": (Anchor("day", ("lastupdate_260D",)),),
    "credit_bureau_b_2": (Anchor("day", ("pmts_date_1107D",)),),
}


@dataclass(frozen=True)
class TablePlan:
    table: str
    parent: str | None
    anchors: tuple[Anchor, ...]
    numeric: tuple[str, ...] = ()
    bools: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()
    categories: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    has_children: bool = False

    @property
    def anchor_columns(self) -> tuple[str, ...]:
        return tuple(c for a in self.anchors for c in a.columns)

    @property
    def columns(self) -> list[str]:
        keys = [KEY, "num_group1"] if (self.parent or self.has_children) else [KEY]
        return list(dict.fromkeys([*keys, *self.anchor_columns, *self.numeric, *self.bools, *self.dates, *self.categories]))

    def schema(self) -> dict[str, str]:
        out = {c: "int" for c in (KEY, "num_group1") if c in self.columns}
        for a in self.anchors:
            out.update({c: "date" if a.kind == "day" else "float" for c in a.columns})
        out.update({c: "float" for c in self.numeric})
        out.update({c: "bool" for c in self.bools})
        out.update({c: "date" for c in self.dates})
        out.update({c: "string" for c in self.categories})
        return out


@dataclass(frozen=True)
class FamilyPlan:
    family: str
    tables: tuple[TablePlan, ...]  # parent before child
    skipped: Mapping[str, str] = field(default_factory=dict)  # table -> reason (never read)
    without_vocabulary: tuple[str, ...] = ()  # included categorical columns with no approved categories yet

    def fingerprint(self) -> str:
        payload = {"rules_version": RULES_VERSION, "plan": asdict(self)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=list).encode()).hexdigest()


# ------------------------------------------------------------------------- selection -> plan
def load_selection(path: Path = DEFAULT_SELECTION_PATH) -> pl.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"B5 selection not found: {path}")
    selection = pl.read_csv(path, infer_schema=False)
    missing = sorted(SELECTION_COLUMNS - set(selection.columns))
    if missing:
        raise ValueError(f"B5 selection is missing columns {missing}")
    if selection.select("family", "column").is_duplicated().any():
        raise ValueError("B5 selection has repeated (family, column) rows")
    return selection


def _logical(column: str, dtype: str) -> str:
    physical = getattr(pl, dtype, None)
    if physical is None:
        raise ValueError(f"unknown dtype {dtype!r} for {column!r} in the B5 selection")
    return coercion.logical_type_for(column, physical)


def plan_from_selection(
    family: str,
    selection: pl.DataFrame,
    *,
    included_actions: Iterable[str] = INCLUDED_ACTIONS,
    category_shares: Mapping[str, Mapping[str, Iterable[str]]] | None = None,
) -> FamilyPlan:
    """Build the aggregation plan of a family from the B5 selection (data-driven, no fixed column lists)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown bureau family {family!r}; available: {sorted(FAMILIES)}")
    included_actions = set(included_actions)
    if ANCHOR_ACTION in included_actions:
        raise ValueError(f"{ANCHOR_ACTION} columns are anchors, not predictors")
    shares = {t: {c: tuple(v) for c, v in cols.items()} for t, cols in (category_shares or {}).items()}
    unknown = sorted(set(shares) - set(FAMILIES[family]))
    if unknown:
        raise ValueError(f"category_shares names tables outside {family}: {unknown}")
    plans: list[TablePlan] = []
    skipped: dict[str, str] = {}
    without_vocabulary: list[str] = []
    for table in FAMILIES[family]:
        rows = selection.filter(pl.col("family") == table)
        if rows.is_empty():
            raise ValueError(f"B5 selection has no rows for {table}")
        refs = set(rows.filter(pl.col("action") == ANCHOR_ACTION)["column"])
        known = {c for a in ANCHORS[table] for c in a.columns}
        if refs - known:
            raise ValueError(f"{table}: {sorted(refs - known)} marked {ANCHOR_ACTION} but no anchor rule exists for them")
        anchors = tuple(a for a in ANCHORS[table] if set(a.columns) <= refs)
        parent = PARENTS.get(table)
        if not anchors:
            skipped[table] = "sin ancla temporal aprobada (referencia_as_of) en B5: la tabla no se lee"
        elif parent and parent in skipped:
            skipped[table] = f"su tabla padre {parent} no se procesa"
        table_shares = shares.get(table, {})
        if table in skipped:
            if table_shares:
                raise ValueError(f"category_shares given for skipped table {table}")
            continue
        numeric, bools, dates, categories = [], [], [], {}
        for r in rows.filter(pl.col("action").is_in(sorted(included_actions))).sort("column").iter_rows(named=True):
            column, logical = r["column"], _logical(r["column"], r["dtype"])
            if column in RESERVED or column in known:
                raise ValueError(f"{table}: {column} is a key or an anchor and cannot be a predictor")
            if r["action"] == "transformar_as_of" and logical != "date":
                raise ValueError(f"{table}: {column} is transformar_as_of but not a date")
            if logical == "date":
                dates.append(column)
            elif logical in ("float", "int"):
                numeric.append(column)
            elif logical == "bool":
                bools.append(column)
            elif column in table_shares:
                categories[column] = table_shares[column]
            else:
                without_vocabulary.append(f"{table}.{column}")
        not_included = sorted(set(table_shares) - set(categories))
        if not_included:
            raise ValueError(f"{table}: category_shares for columns that are not included string columns: {not_included}")
        for column, cats in categories.items():
            ap._check_categories({column: cats}, f"category_shares[{table!r}]")
        plans.append(TablePlan(table, parent, anchors, tuple(numeric), tuple(bools), tuple(dates), categories))
    parents_used = {p.parent for p in plans if p.parent}
    plans = [replace(p, has_children=p.table in parents_used) for p in plans]
    return FamilyPlan(family, tuple(plans), skipped, tuple(without_vocabulary))


# --------------------------------------------------------------------------- pure core
def _prepare_batch(
    batch: pl.DataFrame, plan: TablePlan, decision: pl.DataFrame, parent_keys: pl.DataFrame | None
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Coerce, keep base cases and apply the temporal and parent rules before any aggregation."""
    t = plan.table
    rows = coercion.coerce_to_logical(batch.select(plan.columns), plan.schema()).join(decision, on=KEY, how="inner")
    d = pl.col("decision")
    present = pl.sum_horizontal([a.present() for a in plan.anchors])
    bad = pl.sum_horizontal([a.present() & ~a.ok(d) for a in plan.anchors])
    flags = rows.select(null_anchor=present == 0, after=(present > 0) & (bad > 0))
    counts = {
        f"{t}_rows_null_anchor": int(flags["null_anchor"].sum()),
        f"{t}_rows_after_decision": int(flags["after"].sum()),
    }
    rows = rows.filter((present > 0) & (bad == 0))
    if plan.parent:
        valid = rows.join(parent_keys, on=list(PARENT_KEY), how="semi")
        counts[f"{t}_rows_invalid_parent"] = rows.height - valid.height
        rows = valid
    for c in plan.dates:  # internal dates after the decision become null; the total goes to the manifest only
        counts[f"{t}_{c}_values_after_decision"] = int(rows.select((pl.col(c) > d).sum()).item() or 0)
        rows = rows.with_columns(pl.when(pl.col(c) <= d).then(pl.col(c)).alias(c))
    return rows, counts


def _partial_exprs(plan: TablePlan) -> list[pl.Expr]:
    out = [pl.len().cast(pl.Int64).alias("n_rows")]
    for c in plan.numeric:
        x = pl.col(c)
        out += [
            x.count().cast(pl.Int64).alias(f"{c}|n"),
            x.sum().cast(pl.Float64).alias(f"{c}|sum"),
            (x - x.mean()).pow(2).sum().cast(pl.Float64).alias(f"{c}|m2"),  # two-pass: stable for large amounts
            x.min().cast(pl.Float64).alias(f"{c}|min"),
            x.max().cast(pl.Float64).alias(f"{c}|max"),
        ]
    for c in plan.bools:
        out += [pl.col(c).count().cast(pl.Int64).alias(f"{c}|n"), pl.col(c).sum().cast(pl.Int64).alias(f"{c}|true")]
    for c in plan.dates:
        out += [pl.col(c).min().alias(f"{c}|min"), pl.col(c).max().alias(f"{c}|max")]
    for c, cats in plan.categories.items():
        out.append(pl.col(c).count().cast(pl.Int64).alias(f"{c}|n"))
        out += [(pl.col(c) == cat).sum().cast(pl.Int64).alias(f"{c}|cat{i}") for i, cat in enumerate(cats)]
    return out


def partial_aggregate(rows: pl.DataFrame, plan: TablePlan) -> pl.DataFrame:
    """Mergeable per-case statistics of already filtered rows."""
    return rows.group_by(KEY).agg(_partial_exprs(plan)).sort(KEY)


def empty_partial(plan: TablePlan) -> pl.DataFrame:
    rows = pl.DataFrame(schema={c: coercion.LOGICAL_TYPES[t] for c, t in plan.schema().items()})
    return partial_aggregate(rows, plan)


def merge_partials(partials: Iterable[pl.DataFrame], plan: TablePlan) -> pl.DataFrame:
    """Combine partials of the same table (associative and commutative)."""
    frames = [p for p in partials if p.height] or [empty_partial(plan)]
    stacked = pl.concat(frames, how="vertical")
    devs = []
    for c in plan.numeric:
        n, s = pl.col(f"{c}|n"), pl.col(f"{c}|sum")
        grand_mean = s.sum().over(KEY) / n.sum().over(KEY)
        devs.append(pl.when(n > 0).then(n * (s / n - grand_mean).pow(2)).otherwise(0.0).alias(f"{c}|dev"))
    stacked = stacked.with_columns(devs)
    aggs = [pl.col("n_rows").sum()]
    for c in plan.numeric:
        aggs += [
            pl.col(f"{c}|n").sum(),
            pl.col(f"{c}|sum").sum(),
            (pl.col(f"{c}|m2").sum() + pl.col(f"{c}|dev").sum()).alias(f"{c}|m2"),
            pl.col(f"{c}|min").min(),
            pl.col(f"{c}|max").max(),
        ]
    for c in plan.bools:
        aggs += [pl.col(f"{c}|n").sum(), pl.col(f"{c}|true").sum()]
    for c in plan.dates:
        aggs += [pl.col(f"{c}|min").min(), pl.col(f"{c}|max").max()]
    for c, cats in plan.categories.items():
        aggs += [pl.col(f"{c}|n").sum(), *(pl.col(f"{c}|cat{i}").sum() for i in range(len(cats)))]
    return stacked.group_by(KEY).agg(aggs).sort(KEY)


def _when_any(n: pl.Expr, value: pl.Expr) -> pl.Expr:
    return pl.when(n > 0).then(value)


def finalize_table(partial: pl.DataFrame, plan: TablePlan, decision: pl.DataFrame) -> pl.DataFrame:
    """Features of one table for every base case (cases without records: count 0, not present)."""
    t = plan.table
    d = pl.col("decision")
    exprs = [pl.col("n_rows").fill_null(0).alias(f"{t}__n_records"), (pl.col("n_rows").fill_null(0) > 0).alias(f"{t}__source_present")]
    for c in plan.numeric:
        n = pl.col(f"{c}|n").fill_null(0)
        values = {
            "count": n,
            "min": _when_any(n, pl.col(f"{c}|min")),
            "max": _when_any(n, pl.col(f"{c}|max")),
            "mean": _when_any(n, pl.col(f"{c}|sum") / n),
            "sum": _when_any(n, pl.col(f"{c}|sum")),
            "std": pl.when(n > 1).then((pl.col(f"{c}|m2").clip(lower_bound=0) / (n - 1)).sqrt()),
        }
        exprs += [values[agg].alias(ap._name(t, c, agg)) for agg in NUMERIC_AGGS]
    for c in plan.bools:
        n = pl.col(f"{c}|n").fill_null(0)
        exprs.append(_when_any(n, pl.col(f"{c}|true") / n).cast(pl.Float64).alias(ap._name(t, c, "share_true")))
    for c in plan.dates:
        exprs += [
            (d - pl.col(f"{c}|max")).dt.total_days().alias(ap._name(t, c, "days_since_last")),
            (d - pl.col(f"{c}|min")).dt.total_days().alias(ap._name(t, c, "days_since_first")),
        ]
    for c, cats in plan.categories.items():
        n = pl.col(f"{c}|n").fill_null(0)
        exprs += [
            _when_any(n, pl.col(f"{c}|cat{i}") / n).cast(pl.Float64).alias(ap._name(t, c, f"share_{ap._norm(cat)}"))
            for i, cat in enumerate(cats)
        ]
    return decision.join(partial, on=KEY, how="left").select(KEY, *exprs).sort(KEY)


@dataclass
class TableState:
    """Running result of one table: merged partial, exclusion totals and valid parent keys."""

    partial: pl.DataFrame
    exclusions: dict[str, int]
    parent_keys: pl.DataFrame | None


def _add(total: dict[str, int], counts: Mapping[str, int]) -> None:
    for k, v in counts.items():
        total[k] = total.get(k, 0) + int(v)


def process_batches(
    batches: Iterable[pl.DataFrame], plan: TablePlan, decision: pl.DataFrame, parent_keys: pl.DataFrame | None
) -> TableState:
    """Stream batches: filter, aggregate to a partial and merge, keeping memory bounded by the cases seen."""
    partial = empty_partial(plan)
    exclusions: dict[str, int] = {}
    keys: list[pl.DataFrame] = []
    for batch in batches:
        rows, counts = _prepare_batch(batch, plan, decision, parent_keys)
        _add(exclusions, counts)
        partial = merge_partials([partial, partial_aggregate(rows, plan)], plan)
        if plan.has_children:
            keys.append(rows.select(PARENT_KEY))
    if not exclusions:  # no batch at all: report zeros, not missing keys
        _, exclusions = _prepare_batch(pl.DataFrame(schema={c: pl.Null for c in plan.columns}), plan, decision, parent_keys)
    parent = pl.concat(keys).unique() if keys else pl.DataFrame(schema={KEY: pl.Int64, "num_group1": pl.Int64})
    return TableState(partial, exclusions, parent if plan.has_children else None)


def _decision(base: pl.DataFrame) -> pl.DataFrame:
    return ap._decision(base)


def aggregate_family(
    base: pl.DataFrame, plan: FamilyPlan, batches: Mapping[str, Iterable[pl.DataFrame]]
) -> ap.FamilyAggregate:
    """Pure core: ``batches`` maps each processed table to an iterable of row batches (any split into batches)."""
    decision = _decision(base)
    states: dict[str, TableState] = {}
    for tp in plan.tables:
        parent_keys = states[tp.parent].parent_keys if tp.parent else None
        states[tp.table] = process_batches(batches.get(tp.table, ()), tp, decision, parent_keys)
    return _assemble(plan, states, decision)


def _assemble(plan: FamilyPlan, states: Mapping[str, TableState], decision: pl.DataFrame) -> ap.FamilyAggregate:
    features = decision.select(KEY).sort(KEY)
    exclusions: dict[str, int] = {}
    for tp in plan.tables:
        features = features.join(finalize_table(states[tp.table].partial, tp, decision), on=KEY, how="left")
        _add(exclusions, states[tp.table].exclusions)
    ap._check_names(features)
    return ap.FamilyAggregate(features=features.sort(KEY), exclusions=exclusions)


# ------------------------------------------------------------------- shards and checkpoints
def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def _read_shard(path: Path, columns: list[str]) -> Iterable[pl.DataFrame]:
    missing = [c for c in columns if c not in pl.read_parquet_schema(path)]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")
    return io.iter_row_groups(path, columns=columns)


def _write_atomic(frame: pl.DataFrame, path: Path) -> str:
    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.write_parquet(tmp)
        digest = contracts._sha256(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return digest


class ShardCache:
    """Per-shard checkpoints. A checkpoint is valid only if its key (G1 §5) and checksums match."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _paths(self, table: str, shard: Path, key: Mapping[str, str]) -> tuple[Path, Path, Path]:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]
        folder, stem = self.root / table, f"{shard.stem}.{digest}"
        return folder / f"{stem}.partial.parquet", folder / f"{stem}.keys.parquet", folder / f"{stem}.json"

    def load(self, table: str, shard: Path, key: Mapping[str, str]) -> TableState | None:
        partial_path, keys_path, meta_path = self._paths(table, shard, key)
        try:
            meta = json.loads(meta_path.read_text())
            if meta["key"] != dict(key) or contracts._sha256(partial_path) != meta["partial_sha256"]:
                return None
            keys = None
            if meta["keys_sha256"] is not None:
                if contracts._sha256(keys_path) != meta["keys_sha256"]:
                    return None
                keys = pl.read_parquet(keys_path)
            return TableState(pl.read_parquet(partial_path), {k: int(v) for k, v in meta["exclusions"].items()}, keys)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return None  # absent or corrupt: recompute

    def save(self, table: str, shard: Path, key: Mapping[str, str], state: TableState) -> None:
        partial_path, keys_path, meta_path = self._paths(table, shard, key)
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "key": dict(key),
            "exclusions": state.exclusions,
            "partial_sha256": _write_atomic(state.partial, partial_path),
            "keys_sha256": _write_atomic(state.parent_keys, keys_path) if state.parent_keys is not None else None,
        }
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
        os.replace(tmp, meta_path)  # the metadata is written last: a visible checkpoint is complete


def _build_family(
    family: str,
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int,
    output_dir: Path,
    selection_path: Path,
    category_shares: Mapping[str, Mapping[str, Iterable[str]]] | None,
    included_actions: Iterable[str],
    cache_dir: Path | None,
) -> contracts.FamilyFeatureArtifact:
    contracts.validate_base(base, split, cutoff_week)
    plan = plan_from_selection(
        family, load_selection(selection_path), included_actions=included_actions, category_shares=category_shares
    )
    file_split = "test" if split == "test" else "train"
    decision = _decision(base)
    base_fp = ap._base_fingerprint(base)
    cache = ShardCache(Path(cache_dir) if cache_dir is not None else Path(output_dir) / "_partial_cache")

    states: dict[str, TableState] = {}
    shard_fps: dict[str, str] = {}
    for tp in plan.tables:
        shards = io.shard_paths(file_split, tp.table)  # discovered by pattern: train and test counts differ
        shard_fps[tp.table] = ";".join(_file_fingerprint(p) for p in shards)
        parent_keys = states[tp.parent].parent_keys if tp.parent else None
        partial, exclusions, keys = empty_partial(tp), {}, []
        for shard in shards:
            key = {
                "rules_version": RULES_VERSION,
                "plan": plan.fingerprint(),
                "shard": _file_fingerprint(shard),
                "parent_shards": shard_fps[tp.parent] if tp.parent else "",
                "base_sha256": base_fp,
                "cutoff_week": str(cutoff_week),
            }
            state = cache.load(tp.table, shard, key)
            if state is None:
                state = process_batches(_read_shard(shard, tp.columns), tp, decision, parent_keys)
                cache.save(tp.table, shard, key, state)
            partial = merge_partials([partial, state.partial], tp)
            _add(exclusions, state.exclusions)
            if state.parent_keys is not None:
                keys.append(state.parent_keys)
        parent = (pl.concat(keys).unique() if keys else None) if tp.has_children else None
        states[tp.table] = TableState(partial, exclusions, parent)

    result = _assemble(plan, states, decision)
    definitions = tuple(
        d
        for tp in plan.tables
        for d in contracts.describe_features(
            result.features.select([c for c in result.features.columns if c.startswith(f"{tp.table}__")]),
            temporal_rule=TEMPORAL_RULES[tp.table],
            source_columns={c.lower(): c for c in tp.columns},
        )
    )
    inputs = {
        "cutoff_week": str(cutoff_week),
        "base_sha256": base_fp,
        "shards": ";".join(shard_fps[tp.table] for tp in plan.tables),
        "selection_manifest": Path(selection_path).name,
        "plan_sha256": plan.fingerprint(),
    }
    inputs.update({f"skipped_table:{t}": reason for t, reason in plan.skipped.items()})
    if plan.without_vocabulary:
        inputs["categorical_without_vocabulary"] = ";".join(plan.without_vocabulary)
    return contracts.write_family_features(
        result.features,
        definitions,
        family=family,
        split=split,
        output_dir=output_dir,
        exclusions=result.exclusions,
        rules_version=RULES_VERSION,
        inputs=inputs,
    )


_ROW_RULE = "fila válida si tiene al menos un ancla no nula y toda ancla no nula cumple su regla; ancla nula: fila descartada"
TEMPORAL_RULES = {
    "credit_bureau_a_1": (
        "lastupdate_1112D (activos) y lastupdate_388D (cerrados) <= date_decision; " + _ROW_RULE
        + "; fechas internas posteriores a la decisión se anulan"
    ),
    "credit_bureau_a_2": (
        "mes de pago (pmts_year_1139T + pmts_month_158T, o pmts_year_507T + pmts_month_706T) estrictamente anterior "
        "al mes de date_decision y padre válido en credit_bureau_a_1; " + _ROW_RULE
    ),
    "credit_bureau_b_1": "lastupdate_260D <= date_decision; " + _ROW_RULE + "; fechas internas posteriores a la decisión se anulan",
    "credit_bureau_b_2": "pmts_date_1107D <= date_decision y padre (case_id, num_group1) válido en credit_bureau_b_1; " + _ROW_RULE,
}


def build_credit_bureau_a_features(
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    category_shares: Mapping[str, Mapping[str, Iterable[str]]] | None = None,
    included_actions: Iterable[str] = INCLUDED_ACTIONS,
    cache_dir: Path | None = None,
) -> contracts.FamilyFeatureArtifact:
    """Family ``credit_bureau_a`` (C4): one row per base case, streamed by row group with shard checkpoints."""
    return _build_family(
        "credit_bureau_a", base, split, cutoff_week=cutoff_week, output_dir=output_dir, selection_path=selection_path,
        category_shares=category_shares, included_actions=included_actions, cache_dir=cache_dir,
    )


def build_credit_bureau_b_features(
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    category_shares: Mapping[str, Mapping[str, Iterable[str]]] | None = None,
    included_actions: Iterable[str] = INCLUDED_ACTIONS,
    cache_dir: Path | None = None,
) -> contracts.FamilyFeatureArtifact:
    """Family ``credit_bureau_b`` (C5): children only with a temporally valid parent, no cross-family joins."""
    return _build_family(
        "credit_bureau_b", base, split, cutoff_week=cutoff_week, output_dir=output_dir, selection_path=selection_path,
        category_shares=category_shares, included_actions=included_actions, cache_dir=cache_dir,
    )


BUILDERS = {"credit_bureau_a": build_credit_bureau_a_features, "credit_bureau_b": build_credit_bureau_b_features}
