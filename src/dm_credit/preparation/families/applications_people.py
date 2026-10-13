"""C2/C3 — Núcleo de agregación de solicitudes previas y personas (propuesta A4).

Funciones puras sobre tablas ya leídas: no leen disco, no escriben manifiestos y no conocen
la interfaz ``build_family_features`` (que G1 aún no congela). Devuelven una fila por
``case_id`` **solo para los casos con cobertura**. Los adaptadores (``build_*_features``) completan
después una fila por cada caso de la base, con ``n_records = 0`` y ``source_present = False``, igual
que bureau y productos, porque el ensamblador no rellena cobertura.

Reglas temporales (A4 §2, ventana ``all`` de A7):
    - ``applprev_1``: una fila existe solo si ``creationdate_885D <= date_decision``; fecha nula
      se descarta; el mismo día se conserva.
    - ``applprev_2``: un hijo se usa solo si su padre ``(case_id, num_group1)`` es válido.
    - Fechas internas posteriores a la decisión pasan a nulo; su total va solo al manifiesto
      (``exclusions``), nunca como característica por caso.
    - ``person_*`` no tiene fecha por fila: son atributos de la solicitud actual (pendiente de
      aprobar en G1); la persona principal es ``num_group1 = 0``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import polars as pl

from dm_credit import io
from dm_credit.preparation import contracts
from dm_credit.preparation.coercion import coerce_to_logical

KEY = "case_id"
WINDOW = "all"
RULES_VERSION = "A4-v1"
CREATION = "creationdate_885D"
AGGREGATIONS = ("count", "min", "max", "mean", "sum", "std")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(__[a-z0-9_]+){1,4}$")  # gramática de G1: hasta 4 segmentos tras la fuente


@dataclass(frozen=True)
class FamilyAggregate:
    features: pl.DataFrame  # una fila por case_id con cobertura
    exclusions: dict[str, int]  # totales para el manifiesto y el informe de fuga


@dataclass(frozen=True)
class ApplicationsConfig:
    """Columnas a agregar. Los valores por defecto son los candidatos de B4 (pendientes de G2)."""

    numeric: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "credamount_590A": ("count", "min", "max", "mean", "sum"),
            "annuity_853A": ("count", "min", "max", "mean", "sum"),
            "tenor_203L": ("count", "min", "max", "mean"),
            "mainoccupationinc_437A": ("count", "min", "max", "mean"),
            "byoccupationinc_3656910L": ("count", "min", "max", "mean"),
            "childnum_21L": ("count", "min", "max", "mean"),
        }
    )
    categorical_distinct: tuple[str, ...] = ("credtype_587L", "inittransactioncode_279L", "education_1138M", "familystate_726L")
    category_shares: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # vocabulario ajustado en train
    bool_cols: tuple[str, ...] = ("isbidproduct_390L", "isdebitcard_527L")
    date_cols: tuple[str, ...] = (
        "approvaldate_319D",
        "dateactivated_425D",
        "dtlastpmt_581D",
        "dtlastpmtallstes_3545839D",
        "employedfrom_700D",
    )
    children_category_shares: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class PeopleConfig:
    main_numeric: tuple[str, ...] = ("mainoccupationinc_384A",)
    main_categorical: tuple[str, ...] = ("education_927M", "incometype_1044T")
    birth_col: str | None = "birth_259D"
    employment_col: str | None = "empl_employedfrom_271D"
    contact_category_shares: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    child_category_shares: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


# ----------------------------------------------------------------------- utilidades
def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _name(family: str, column: str, agg: str, window: str = WINDOW) -> str:
    return f"{family}__{_norm(column)}__{agg}__{window}"


def _require(df: pl.DataFrame, columns, table: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing required columns {missing}")


def _check_unique(df: pl.DataFrame, keys: list[str], table: str) -> None:
    if df.select(keys).is_duplicated().any():
        raise ValueError(f"{table}: key {keys} is not unique")


def _check_names(frame: pl.DataFrame) -> None:
    names = [c for c in frame.columns if c != KEY]
    bad = [c for c in names if not NAME_RE.match(c)]
    if bad:
        raise ValueError(f"feature names violate the naming convention: {bad}")
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("duplicated feature names")


def _check_no_repeats(values, what: str) -> None:
    repeated = sorted({v for v in values if list(values).count(v) > 1})
    if repeated:
        raise ValueError(f"invalid config: {what} has repeated entries {repeated}")


def _check_categories(shares: Mapping[str, tuple[str, ...]], what: str) -> None:
    for col, cats in shares.items():
        names = [_norm(c) for c in cats]
        _check_no_repeats(names, f"{what}[{col!r}] (categories collide after normalizing to feature names)")


def _validate_applications_config(cfg: "ApplicationsConfig", creation_col: str) -> None:
    for what, values in (("categorical_distinct", cfg.categorical_distinct), ("bool_cols", cfg.bool_cols), ("date_cols", cfg.date_cols)):
        _check_no_repeats(values, what)
    if creation_col in cfg.date_cols:
        raise ValueError(f"invalid config: {creation_col} is the temporal reference and cannot be in date_cols")
    numeric, text = set(cfg.numeric), {*cfg.categorical_distinct, *cfg.category_shares}
    for a, b, label in (
        (numeric, text, "numeric and categorical"),
        (numeric, set(cfg.bool_cols), "numeric and bool"),
        (numeric, set(cfg.date_cols), "numeric and date"),
        (text, set(cfg.bool_cols), "categorical and bool"),
        (text, set(cfg.date_cols), "categorical and date"),
        (set(cfg.bool_cols), set(cfg.date_cols), "bool and date"),
    ):
        both = sorted(a & b)
        if both:
            raise ValueError(f"invalid config: columns used with two incompatible roles ({label}): {both}")
    _check_categories(cfg.category_shares, "category_shares")
    _check_categories(cfg.children_category_shares, "children_category_shares")


def _validate_people_config(cfg: "PeopleConfig") -> None:
    _check_no_repeats(cfg.main_numeric, "main_numeric")
    _check_no_repeats(cfg.main_categorical, "main_categorical")
    both = sorted(set(cfg.main_numeric) & set(cfg.main_categorical))
    if both:
        raise ValueError(f"invalid config: columns used as numeric and categorical: {both}")
    _check_categories(cfg.contact_category_shares, "contact_category_shares")
    _check_categories(cfg.child_category_shares, "child_category_shares")


def _numeric_exprs(family: str, numeric: Mapping[str, tuple[str, ...]]) -> list[pl.Expr]:
    out = []
    for col, aggs in numeric.items():
        bad = [a for a in aggs if a not in AGGREGATIONS]
        if bad:
            raise ValueError(f"unknown aggregation(s) {bad} for {col}; valid: {list(AGGREGATIONS)}")
        c = pl.col(col)
        for agg in aggs:
            if agg == "count":
                expr = c.count()
            else:
                expr = pl.when(c.count() > 0).then(getattr(c, agg)())  # sin observaciones válidas -> nulo
            out.append(expr.alias(_name(family, col, agg)))
    return out


def _share_exprs(family: str, shares: Mapping[str, tuple[str, ...]]) -> list[pl.Expr]:
    out = []
    for col, cats in shares.items():
        c = pl.col(col)
        for cat in cats:
            expr = pl.when(c.count() > 0).then((c == cat).sum() / c.count())
            out.append(expr.cast(pl.Float64).alias(_name(family, col, f"share_{_norm(cat)}")))
    return out


def _applications_columns(cfg: "ApplicationsConfig") -> tuple[list[str], list[str]]:
    a1 = [KEY, "num_group1", CREATION, *cfg.numeric, *cfg.categorical_distinct, *cfg.category_shares, *cfg.bool_cols, *cfg.date_cols]
    return list(dict.fromkeys(a1)), [KEY, "num_group1", "num_group2", *cfg.children_category_shares]


def _people_columns(cfg: "PeopleConfig") -> tuple[list[str], list[str]]:
    dates = [c for c in (cfg.birth_col, cfg.employment_col) if c]
    p1 = [KEY, "num_group1", *cfg.main_numeric, *cfg.main_categorical, *dates, *cfg.contact_category_shares]
    return list(dict.fromkeys(p1)), [KEY, "num_group1", "num_group2", *cfg.child_category_shares]


def _decision(base: pl.DataFrame) -> pl.DataFrame:
    _require(base, [KEY, "date_decision"], "base")
    _check_unique(base, [KEY], "base")
    return coerce_to_logical(base.select(KEY, "date_decision"), {KEY: "int", "date_decision": "date"}).rename(
        {"date_decision": "decision"}
    )


def _join_decision(frame: pl.DataFrame, decision: pl.DataFrame, table: str) -> pl.DataFrame:
    if frame.join(decision, on=KEY, how="anti").height:
        raise ValueError(f"{table}: contains case_id values missing from the base table")
    return frame.join(decision, on=KEY, how="inner")


def _null_if_after(expr_col: str, count_key: str, frame: pl.DataFrame, exclusions: dict[str, int]) -> pl.DataFrame:
    """Anula los valores posteriores a la decisión y registra su total solo para el manifiesto."""
    exclusions[count_key] = int(frame.select((pl.col(expr_col) > pl.col("decision")).sum()).item() or 0)
    return frame.with_columns(pl.when(pl.col(expr_col) <= pl.col("decision")).then(pl.col(expr_col)).alias(expr_col))


def _days_since(col: str, agg: str) -> pl.Expr:
    value = getattr(pl.col(col), agg)()
    return (pl.col("decision").first() - value).dt.total_days()


# ------------------------------------------------------------------ solicitudes previas
def aggregate_applications_with_exclusions(
    base: pl.DataFrame,
    applprev_1: pl.DataFrame,
    applprev_2: pl.DataFrame,
    config: ApplicationsConfig | None = None,
) -> FamilyAggregate:
    cfg = config or ApplicationsConfig()
    f1, f2 = "applprev_1", "applprev_2"
    create = "creationdate_885D"
    _validate_applications_config(cfg, create)

    a1_cols, a2_cols = _applications_columns(cfg)
    _require(applprev_1, a1_cols, f1)
    _require(applprev_2, a2_cols, f2)
    _check_unique(applprev_1, [KEY, "num_group1"], f1)
    _check_unique(applprev_2, [KEY, "num_group1", "num_group2"], f2)

    schema = {KEY: "int", "num_group1": "int", create: "date"}
    schema.update({c: "float" for c in cfg.numeric})
    schema.update({c: "string" for c in (*cfg.categorical_distinct, *cfg.category_shares)})
    schema.update({c: "bool" for c in cfg.bool_cols})
    schema.update({c: "date" for c in cfg.date_cols})
    a1 = coerce_to_logical(applprev_1.select(a1_cols), schema)
    a2 = coerce_to_logical(
        applprev_2.select(a2_cols),
        {KEY: "int", "num_group1": "int", "num_group2": "int", **{c: "string" for c in cfg.children_category_shares}},
    )

    a1 = _join_decision(a1, _decision(base), f1)
    exclusions = {
        "applprev_1_rows_null_decision": int(a1.select(pl.col("decision").is_null().sum()).item()),
        "applprev_1_rows_after_decision": int(a1.select((pl.col(create) > pl.col("decision")).sum()).item() or 0),
        "applprev_1_rows_null_creation": int(a1.select(pl.col(create).is_null().sum()).item()),
    }
    valid = a1.filter(pl.col(create) <= pl.col("decision"))  # la fecha nula no cumple y queda fuera
    for col in cfg.date_cols:
        valid = _null_if_after(col, f"{col}_values_after_decision", valid, exclusions)

    # recencia y antigüedad: <fuente>__<campo>__days_since_last|first__<ventana> (gramática de G1)
    exprs = [
        pl.len().alias(f"{f1}__n_records"),
        pl.lit(True).alias(f"{f1}__source_present"),
        _days_since(create, "max").alias(_name(f1, create, "days_since_last")),
        _days_since(create, "min").alias(_name(f1, create, "days_since_first")),
    ]
    exprs += _numeric_exprs(f1, cfg.numeric)
    exprs += [pl.col(c).drop_nulls().n_unique().alias(_name(f1, c, "n_distinct")) for c in cfg.categorical_distinct]
    exprs += _share_exprs(f1, cfg.category_shares)
    for c in cfg.bool_cols:
        col = pl.col(c)
        exprs.append(pl.when(col.count() > 0).then(col.sum() / col.count()).cast(pl.Float64).alias(_name(f1, c, "share_true")))
    exprs += [_days_since(c, "max").alias(_name(f1, c, "days_since_last")) for c in cfg.date_cols]
    features = valid.group_by(KEY).agg(exprs)

    # profundidad 2: se valida el padre antes de agregar los hijos
    parents = valid.select(KEY, "num_group1")
    exclusions["applprev_2_rows_invalid_parent"] = a2.join(parents, on=[KEY, "num_group1"], how="anti").height
    children = a2.join(parents, on=[KEY, "num_group1"], how="semi")
    # los padres válidos sin hijos cuentan con 0 en el denominador
    per_parent = parents.join(children.group_by(KEY, "num_group1").agg(pl.len().alias("n")), on=[KEY, "num_group1"], how="left").with_columns(
        pl.col("n").fill_null(0)
    )
    child_features = per_parent.group_by(KEY).agg(
        pl.col("n").sum().alias(f"{f2}__n_records"),
        pl.col("n").mean().alias(f"{f2}__children_per_parent_mean__{WINDOW}"),
        pl.col("n").max().alias(f"{f2}__children_per_parent_max__{WINDOW}"),
    )
    share_exprs = _share_exprs(f2, cfg.children_category_shares)
    if share_exprs:
        child_features = child_features.join(children.group_by(KEY).agg(share_exprs), on=KEY, how="left")
    features = features.join(child_features, on=KEY, how="left").with_columns(
        pl.col(f"{f2}__n_records").fill_null(0)
    )
    features = features.with_columns((pl.col(f"{f2}__n_records") > 0).alias(f"{f2}__source_present"))  # cobertura por tabla física

    features = features.sort(KEY)
    _check_names(features)
    return FamilyAggregate(features=features, exclusions=exclusions)


def aggregate_applications(base, applprev_1, applprev_2, config: ApplicationsConfig | None = None) -> pl.DataFrame:
    return aggregate_applications_with_exclusions(base, applprev_1, applprev_2, config).features


# ------------------------------------------------------------------------- personas
def aggregate_people_with_exclusions(
    base: pl.DataFrame,
    person_1: pl.DataFrame,
    person_2: pl.DataFrame,
    config: PeopleConfig | None = None,
) -> FamilyAggregate:
    cfg = config or PeopleConfig()
    f1, f2 = "person_1", "person_2"
    _validate_people_config(cfg)
    dates = [c for c in (cfg.birth_col, cfg.employment_col) if c]

    p1_cols, p2_cols = _people_columns(cfg)
    _require(person_1, p1_cols, f1)
    _require(person_2, p2_cols, f2)
    _check_unique(person_1, [KEY, "num_group1"], f1)
    _check_unique(person_2, [KEY, "num_group1", "num_group2"], f2)

    schema = {KEY: "int", "num_group1": "int"}
    schema.update({c: "float" for c in cfg.main_numeric})
    schema.update({c: "string" for c in (*cfg.main_categorical, *cfg.contact_category_shares)})
    schema.update({c: "date" for c in dates})
    p1 = coerce_to_logical(person_1.select(p1_cols), schema)
    p2 = coerce_to_logical(
        person_2.select(p2_cols),
        {KEY: "int", "num_group1": "int", "num_group2": "int", **{c: "string" for c in cfg.child_category_shares}},
    )
    p1 = _join_decision(p1, _decision(base), f1)

    mains = p1.filter(pl.col("num_group1") == 0).group_by(KEY).len()
    if mains.height != p1[KEY].n_unique() or (mains["len"] != 1).any():
        raise ValueError(f"{f1}: every case must have exactly one main person row (num_group1 = 0)")
    if p2.join(p1.select(KEY, "num_group1"), on=[KEY, "num_group1"], how="anti").height:
        raise ValueError(f"{f2}: contains rows whose parent (case_id, num_group1) is not in {f1}")

    exclusions: dict[str, int] = {"person_1_rows_null_decision": int(p1.select(pl.col("decision").is_null().sum()).item())}
    for col in dates:
        p1 = _null_if_after(col, f"{col}_values_after_decision", p1, exclusions)

    main = p1.filter(pl.col("num_group1") == 0)
    main_exprs = [pl.col(KEY)]
    main_exprs += [pl.col(c).alias(_name(f1, c, "main")) for c in (*cfg.main_numeric, *cfg.main_categorical)]
    if cfg.birth_col:
        b, d = pl.col(cfg.birth_col), pl.col("decision")
        birthday_pending = (d.dt.month() < b.dt.month()) | ((d.dt.month() == b.dt.month()) & (d.dt.day() < b.dt.day()))
        main_exprs.append((d.dt.year() - b.dt.year() - birthday_pending.cast(pl.Int32)).cast(pl.Int64).alias(_name(f1, cfg.birth_col, "age_years")))
    if cfg.employment_col:
        main_exprs.append(
            (pl.col("decision") - pl.col(cfg.employment_col)).dt.total_days().alias(_name(f1, cfg.employment_col, "days_since_last"))
        )
    features = main.select(main_exprs)

    totals = p1.group_by(KEY).agg(
        pl.len().alias(f"{f1}__n_records"),
        (pl.col("num_group1") >= 1).sum().alias(f"{f1}__n_contacts__{WINDOW}"),
    )
    features = features.join(totals, on=KEY, how="left").with_columns(pl.lit(True).alias(f"{f1}__source_present"))
    contact_shares = _share_exprs(f1, cfg.contact_category_shares)
    if contact_shares:
        contacts = p1.filter(pl.col("num_group1") >= 1).group_by(KEY).agg(contact_shares)
        features = features.join(contacts, on=KEY, how="left")

    child_aggs = [pl.len().alias(f"{f2}__n_records"), *_share_exprs(f2, cfg.child_category_shares)]
    features = features.join(p2.group_by(KEY).agg(child_aggs), on=KEY, how="left").with_columns(
        pl.col(f"{f2}__n_records").fill_null(0)
    )
    features = features.with_columns((pl.col(f"{f2}__n_records") > 0).alias(f"{f2}__source_present"))  # cobertura por tabla física

    features = features.sort(KEY)
    _check_names(features)
    return FamilyAggregate(features=features, exclusions=exclusions)


def aggregate_people(base, person_1, person_2, config: PeopleConfig | None = None) -> pl.DataFrame:
    return aggregate_people_with_exclusions(base, person_1, person_2, config).features


# ----------------------------------------------------------------- adaptadores de familia
TABLES = {"applprev": ("applprev_1", "applprev_2"), "person": ("person_1", "person_2")}  # (tabla padre, tabla hija)


def _complete_coverage(features: pl.DataFrame, base: pl.DataFrame, family: str) -> pl.DataFrame:
    """Una fila por caso de la base, como bureau y productos: sin cobertura, ``n_records = 0`` y
    ``source_present = False`` en cada tabla física; el resto de características queda nulo para D2."""
    full = base.select(KEY).sort(KEY).join(features, on=KEY, how="left", validate="1:1")
    fills = []
    for table in TABLES[family]:
        n, present = f"{table}__n_records", f"{table}__source_present"
        fills += [pl.col(n).fill_null(0), pl.col(present).fill_null(False)]
    return full.with_columns(fills)


def _base_fingerprint(base: pl.DataFrame) -> str:
    """Huella de la base de referencia: cambiar casos, semanas, fechas o partición invalida la caché."""
    ordered = base.select(KEY, "WEEK_NUM", "date_decision").sort(KEY)
    return hashlib.sha256(ordered.write_csv().encode()).hexdigest()


def _shard_fingerprint(file_split: str, sources: list[str]) -> str:
    return ";".join(sorted(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for s in sources for p in io.shard_paths(file_split, s)))


def _read(file_split: str, source: str, columns: list[str], cases: pl.LazyFrame) -> pl.DataFrame:
    """Lee una fuente con proyección y solo los casos de la base (semi-join: no multiplica filas)."""
    return io.scan_parquet_shards(file_split, source, columns=columns).join(cases, on=KEY, how="semi").collect()


def _publish(family, base, split, cutoff_week, output_dir, result, sources, file_split, columns, temporal_rule):
    definitions = contracts.describe_features(
        result.features,
        temporal_rule=temporal_rule,
        source_columns={c.lower(): c for c in columns},
    )
    return contracts.write_family_features(
        result.features,
        definitions,
        family=family,
        split=split,
        output_dir=output_dir,
        exclusions=result.exclusions,
        rules_version=RULES_VERSION,
        inputs={
            "cutoff_week": str(cutoff_week),
            "base_sha256": _base_fingerprint(base),
            "shards": _shard_fingerprint(file_split, sources),
        },
    )


def build_applprev_features(
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    config: ApplicationsConfig | None = None,
) -> contracts.FamilyFeatureArtifact:
    """Familia ``applprev`` (``applprev_1`` + ``applprev_2``) -> una fila por ``case_id`` con cobertura."""
    contracts.validate_base(base, split, cutoff_week)
    cfg = config or ApplicationsConfig()
    _validate_applications_config(cfg, CREATION)
    file_split = "test" if split == "test" else "train"
    cols1, cols2 = _applications_columns(cfg)
    cases = base.lazy().select(KEY)
    a1 = _read(file_split, "applprev_1", cols1, cases)
    a2 = _read(file_split, "applprev_2", cols2, cases)
    result = aggregate_applications_with_exclusions(base, a1, a2, cfg)
    result = FamilyAggregate(_complete_coverage(result.features, base, "applprev"), result.exclusions)
    return _publish(
        "applprev", base, split, cutoff_week, output_dir, result, ["applprev_1", "applprev_2"], file_split,
        [*cols1, *cols2], f"{CREATION} <= date_decision; hijos solo con padre válido",
    )


def build_person_features(
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    config: PeopleConfig | None = None,
) -> contracts.FamilyFeatureArtifact:
    """Familia ``person`` (``person_1`` + ``person_2``): atributos de la solicitud actual, persona principal ``num_group1 = 0``."""
    contracts.validate_base(base, split, cutoff_week)
    cfg = config or PeopleConfig()
    _validate_people_config(cfg)
    file_split = "test" if split == "test" else "train"
    cols1, cols2 = _people_columns(cfg)
    cases = base.lazy().select(KEY)
    p1 = _read(file_split, "person_1", cols1, cases)
    p2 = _read(file_split, "person_2", cols2, cases)
    result = aggregate_people_with_exclusions(base, p1, p2, cfg)
    result = FamilyAggregate(_complete_coverage(result.features, base, "person"), result.exclusions)
    return _publish(
        "person", base, split, cutoff_week, output_dir, result, ["person_1", "person_2"], file_split,
        [*cols1, *cols2], "atributos de la solicitud actual; fechas internas <= date_decision",
    )


BUILDERS = {"applprev": build_applprev_features, "person": build_person_features}
