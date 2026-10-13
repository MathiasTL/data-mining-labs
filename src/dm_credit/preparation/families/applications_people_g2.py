"""G2 — Decisión de columnas de ``applprev`` y ``person`` (propuesta B4 + reglas de G2 + filtro de fuga).

Flujo (todo con datos de entrenamiento, ``WEEK_NUM < 74``; OOT y test no intervienen):
    1. ``classify_candidates``: aplica las reglas a la selección B4 y a los datos de train.
    2. ``fit_vocabulary``: categorías frecuentes para las proporciones (ajustadas solo en train).
    3. ``configs_from_decisions``: configuración de agregación con las columnas aprobadas.
    4. ``apply_screen``: el filtro de fuga (``leakage_screen``) aprueba o rechaza cada columna.

Decisiones de G2: ``aprobada``, ``excluida`` (definitiva en el primer corte), ``bloqueada``
(exclusión preventiva: estados, mora, pagos o fechas con aspecto de calendario; se reevalúa en la
segunda iteración), ``aplazada`` (columna válida que el núcleo aún no sabe agregar) y
``pendiente_filtro`` (candidata a la espera del filtro de fuga).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from dm_credit import io
from dm_credit.preparation import leakage_screen
from dm_credit.preparation.families import applications_people as ap
from dm_credit.preparation.families.applications_people import ApplicationsConfig, PeopleConfig
from dm_credit.preparation.split import partition_base

RESULTS = Path(__file__).resolve().parents[4] / "results" / "preparation"
DEFAULT_SELECTION = RESULTS / "B4_applications_people_columns.csv"
G2_COLUMNS = RESULTS / "G2_applications_people_columns.csv"
G2_VOCABULARY = RESULTS / "G2_applications_people_vocabulary.csv"
G2_SCREEN = RESULTS / "G2_applications_people_screen.csv"

MIN_COVERAGE = 0.05  # menos de 5 % de valores en la tabla fuente: sin información suficiente en el primer corte
MIN_FREQUENCY = 0.01  # A6: categorías con al menos 1 % de las observaciones
MAX_CATEGORIES = 10
MAX_DISTINCT_FOR_SHARES = 50  # por encima, solo se cuenta el número de categorías distintas

# Reglas propias de G2 sobre lo que B4 deja como candidato o fecha transformable.
OVERRIDES = {
    ("applprev_1", "dtlastpmtallstes_3545839D"): ("bloqueada", "fecha de último pago posterior a la decisión en 5,4 % de las filas válidas: parece calendario o proyección, no un hecho ocurrido (futuro)"),
    ("applprev_1", "firstnonzeroinstldate_307D"): ("bloqueada", "fecha de primera cuota posterior a la decisión en 3,3 % de las filas: calendario contractual, no prueba disponibilidad (futuro)"),
    ("person_1", "birthdate_87D"): ("excluida", "cobertura insuficiente (99 % nulo) y duplica birth_259D"),
    ("person_1", "gender_992L"): ("excluida", "atributo sensible: excluido por defecto, decisión de equidad pendiente"),
    ("person_1", "sex_738L"): ("excluida", "atributo sensible: excluido por defecto, decisión de equidad pendiente"),
}
MAIN_ONLY = 0.05  # una columna es "de la persona principal" si menos de 5 % de los contactos la informa
PLAN_SCHEMA = {
    "family": pl.String, "column": pl.String, "dtype": pl.String, "b4_action": pl.String,
    "g2_decision": pl.String, "core_role": pl.String, "reason": pl.String,
}


def _coverage(s: pl.Series) -> float:
    return 1.0 - s.null_count() / s.len() if s.len() else 0.0


def _numeric_aggs(column: str) -> tuple[str, ...]:
    return ("count", "min", "max", "mean", "sum") if column.split("_")[-1].endswith("A") else ("count", "min", "max", "mean")


def _row(family, column, dtype, b4, decision, role, reason):
    return {"family": family, "column": column, "dtype": dtype, "b4_action": b4, "g2_decision": decision, "core_role": role, "reason": reason}


def classify_candidates(selection: pl.DataFrame, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Una decisión por columna de la selección B4, con el rol que tendrá en el núcleo de agregación."""
    rows = []
    for r in selection.iter_rows(named=True):
        family, col, dtype, b4 = r["family"], r["column"], r["dtype"], r["action"]
        key = (family, col)
        if b4 == "excluir":
            rows.append(_row(family, col, dtype, b4, "excluida", "none", "excluida desde B4: clave, cuasi-identificador o estado sin trazabilidad"))
        elif b4 == "bloquear_hasta_G2":
            rows.append(_row(family, col, dtype, b4, "bloqueada", "none", "exclusión preventiva: sin evidencia de disponibilidad al decidir; se reevalúa en la segunda iteración"))
        elif b4 == "referencia_as_of":
            rows.append(_row(family, col, dtype, b4, "aprobada", "reference", "fecha de referencia: decide qué filas existían al decidir; no es característica"))
        elif key in OVERRIDES:
            decision, reason = OVERRIDES[key]
            rows.append(_row(family, col, dtype, b4, decision, "none", reason))
        elif b4 == "transformar_as_of":
            role = "birth" if col == "birth_259D" else "employment" if col == "empl_employedfrom_271D" else "date" if family == "applprev_1" else "none"
            if role == "none":
                rows.append(_row(family, col, dtype, b4, "aplazada", "none", "el núcleo aún no agrega esta fecha"))
            else:
                rows.append(_row(family, col, dtype, b4, "pendiente_filtro", role, "fecha utilizable solo como antigüedad o recencia <= date_decision"))
        else:  # candidato_G2
            rows.append(_classify_candidate(family, col, dtype, b4, tables))
    return pl.DataFrame(rows, schema=PLAN_SCHEMA).sort("family", "column")


def _classify_candidate(family, col, dtype, b4, tables) -> dict:
    df = tables.get(family)
    if df is None or col not in df.columns:
        return _row(family, col, dtype, b4, "aplazada", "none", "sin datos de train para evaluar la columna")
    if family == "applprev_1":
        cov = _coverage(df[col])
        if cov < MIN_COVERAGE:
            return _row(family, col, dtype, b4, "excluida", "none", f"cobertura insuficiente ({100 * cov:.1f} % de valores; mínimo {100 * MIN_COVERAGE:.0f} %)")
        if dtype == "Float64":
            role = "numeric"
        elif dtype == "Boolean":
            role = "bool"
        elif df[col].drop_nulls().n_unique() > MAX_DISTINCT_FOR_SHARES:
            role = "distinct"
        else:
            role = "distinct_and_shares"
        return _row(family, col, dtype, b4, "pendiente_filtro", role, "candidata: a la espera del filtro de fuga")
    if family == "person_1":
        main = df.filter(pl.col("num_group1") == 0)[col]
        contacts = df.filter(pl.col("num_group1") >= 1)[col]
        cov_main, cov_contacts = _coverage(main), _coverage(contacts)
        if max(cov_main, cov_contacts) < MIN_COVERAGE:
            return _row(family, col, dtype, b4, "excluida", "none", f"cobertura insuficiente (principal {100 * cov_main:.1f} %, contactos {100 * cov_contacts:.1f} %)")
        if cov_main >= MIN_COVERAGE:
            role = {"Float64": "main_numeric", "String": "main_categorical"}.get(dtype)
        else:
            role = "contact_shares" if dtype == "String" else None
        if role is None:
            return _row(family, col, dtype, b4, "aplazada", "none", "el núcleo aún no agrega este tipo de columna de persona")
        return _row(family, col, dtype, b4, "pendiente_filtro", role, "candidata: a la espera del filtro de fuga")
    return _row(family, col, dtype, b4, "aplazada", "none", "tabla hija: sin agregación definida en el primer corte")


VOCAB_SCHEMA = {"family": pl.String, "column": pl.String, "category": pl.String, "frequency": pl.Float64}


def fit_vocabulary(plan: pl.DataFrame, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Categorías frecuentes (>= 1 %, hasta 10) de las columnas con proporciones, ajustadas solo con train."""
    rows = []
    for r in plan.filter(pl.col("core_role").is_in(["distinct_and_shares", "contact_shares"]) & (pl.col("g2_decision") != "excluida")).iter_rows(named=True):
        df = tables[r["family"]]
        if r["core_role"] == "contact_shares":
            df = df.filter(pl.col("num_group1") >= 1)
        vc = df[r["column"]].drop_nulls().value_counts(sort=True)
        total = vc["count"].sum()
        if not total:
            continue
        top = vc.with_columns((pl.col("count") / total).alias("frequency")).filter(pl.col("frequency") >= MIN_FREQUENCY).sort(["frequency", r["column"]], descending=[True, False]).head(MAX_CATEGORIES)
        rows += [{"family": r["family"], "column": r["column"], "category": str(c), "frequency": float(f)} for c, f in zip(top[r["column"]], top["frequency"])]
    return pl.DataFrame(rows, schema=VOCAB_SCHEMA).sort(["family", "column", "frequency", "category"], descending=[False, False, True, False])


def configs_from_decisions(plan: pl.DataFrame, vocabulary: pl.DataFrame) -> tuple[ApplicationsConfig, PeopleConfig]:
    """Configuración de agregación con solo las columnas ``aprobada``; el vocabulario fija las proporciones."""
    approved = plan.filter(pl.col("g2_decision") == "aprobada")

    def cols(family: str, *roles: str) -> list[str]:
        return approved.filter((pl.col("family") == family) & pl.col("core_role").is_in(list(roles)))["column"].to_list()

    def vocab(family: str, columns: list[str]) -> dict[str, tuple[str, ...]]:
        out = {}
        for c in columns:
            cats = vocabulary.filter((pl.col("family") == family) & (pl.col("column") == c))["category"].to_list()
            if cats:
                out[c] = tuple(cats)
        return out

    shares = cols("applprev_1", "distinct_and_shares")
    app = ApplicationsConfig(
        numeric={c: _numeric_aggs(c) for c in cols("applprev_1", "numeric")},
        categorical_distinct=tuple(cols("applprev_1", "distinct", "distinct_and_shares")),
        category_shares=vocab("applprev_1", shares),
        bool_cols=tuple(cols("applprev_1", "bool")),
        date_cols=tuple(cols("applprev_1", "date")),
        children_category_shares={},
    )
    birth, employment = cols("person_1", "birth"), cols("person_1", "employment")
    people = PeopleConfig(
        main_numeric=tuple(cols("person_1", "main_numeric")),
        main_categorical=tuple(cols("person_1", "main_categorical")),
        birth_col=birth[0] if birth else None,
        employment_col=employment[0] if employment else None,
        contact_category_shares=vocab("person_1", cols("person_1", "contact_shares")),
        child_category_shares={},
    )
    return app, people


def apply_screen(plan: pl.DataFrame, screen: pl.DataFrame) -> pl.DataFrame:
    """Aprueba las candidatas sin banderas y rechaza las que tienen alguna característica sospechosa o inestable."""
    out = []
    for r in plan.iter_rows(named=True):
        if r["g2_decision"] != "pendiente_filtro":
            out.append(r)
            continue
        token = f"__{r['column'].lower()}__"
        mine = screen.filter(pl.col("feature").str.contains(token, literal=True)) if screen.height else screen
        if mine.is_empty():
            out.append(r)  # sin resultados del filtro: sigue pendiente, nunca se aprueba por omisión
            continue
        bad = mine.filter(pl.col("flag").is_in(["leak_suspect", "unstable"]))
        flags = set(mine["flag"])
        if not bad.is_empty():
            worst = bad.sort(pl.col("gini_all").abs(), descending=True, nulls_last=True).row(0, named=True)
            out.append({**r, "g2_decision": "excluida", "reason": f"rechazada por el filtro de fuga: {worst['feature']} es {worst['flag']} (Gini {worst['gini_all']})"})
        elif "ok" in flags:
            peak = mine["gini_all"].drop_nulls().abs().max()
            out.append({**r, "g2_decision": "aprobada", "reason": f"pasa el filtro de fuga en {mine.height} características (|Gini| máximo {peak:.3f})"})
        elif flags == {"not_screened"}:
            out.append({**r, "g2_decision": "aprobada", "reason": f"pasa el filtro de fuga en {mine.height} características (texto: el Gini no se midió; se aprueba por regla, sin evidencia empírica)"})
        else:
            out.append(r)  # solo "insufficient": sin evidencia no se aprueba, sigue pendiente
    return pl.DataFrame(out, schema=PLAN_SCHEMA).sort("family", "column")


def load_g2_configs(columns_path: Path | None = None, vocabulary_path: Path | None = None) -> tuple[ApplicationsConfig, PeopleConfig]:
    """Configuración con las columnas aprobadas en G2. Si falta la decisión es un error: no hay valores por defecto sin aprobar."""
    columns_path, vocabulary_path = columns_path or G2_COLUMNS, vocabulary_path or G2_VOCABULARY
    for path in (columns_path, vocabulary_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found: the G2 decision must exist before building features (run applications_people_g2)")
    return configs_from_decisions(pl.read_csv(columns_path, schema=PLAN_SCHEMA), pl.read_csv(vocabulary_path, schema=VOCAB_SCHEMA))


def _candidate_columns(selection: pl.DataFrame, family: str) -> list[str]:
    return ["case_id", "num_group1", *selection.filter(pl.col("family") == family)["column"].to_list()]


def run_g2(
    *,
    selection_path: Path = DEFAULT_SELECTION,
    output_dir: Path,
    columns_path: Path = G2_COLUMNS,
    vocabulary_path: Path = G2_VOCABULARY,
    screen_path: Path = G2_SCREEN,
    cutoff_week: int = 74,
    leak_threshold: float = 0.5,
    min_rows: int = 1000,
) -> pl.DataFrame:
    """Decide las columnas con datos de entrenamiento y escribe la decisión, el vocabulario y el filtro de fuga."""
    selection = pl.read_csv(selection_path)
    parts = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=cutoff_week)
    train = parts.train
    cases = train.lazy().select("case_id")

    tables = {}
    for family in ("applprev_1", "person_1"):
        cols = [c for c in dict.fromkeys(_candidate_columns(selection, family))]
        lf = io.scan_parquet_shards("train", family)
        cols = [c for c in cols if c in lf.collect_schema().names()]
        tables[family] = lf.select(cols).join(cases, on="case_id", how="semi").collect()

    plan = classify_candidates(selection, tables)
    vocabulary = fit_vocabulary(plan, tables)
    screening = plan.with_columns(g2_decision=pl.when(pl.col("g2_decision") == "pendiente_filtro").then(pl.lit("aprobada")).otherwise(pl.col("g2_decision")))
    app_cfg, people_cfg = configs_from_decisions(screening, vocabulary)

    labeled = train.select("case_id", "WEEK_NUM", "target") if "target" in train.columns else None
    if labeled is None:
        raise ValueError("train base must include target to run the leakage screen")
    screen_frames = []
    for build, cfg in ((ap.build_applprev_features, app_cfg), (ap.build_person_features, people_cfg)):
        art = build(train.select("case_id", "WEEK_NUM", "date_decision"), "train", cutoff_week=cutoff_week, output_dir=Path(output_dir) / build.__name__, config=cfg)
        screen_frames.append(leakage_screen.screen_features(pl.read_parquet(art.data_path), labeled, leak_threshold=leak_threshold, min_rows=min_rows, cutoff_week=cutoff_week))
    screen = pl.concat(screen_frames).sort("feature")

    final = apply_screen(plan, screen)
    for path, frame in ((columns_path, final), (vocabulary_path, vocabulary), (screen_path, screen)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(path)
    return final


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    ap_.add_argument("--output-dir", type=Path, default=io.data_dir() / "artifacts" / "preparation" / "G2-screen")
    ap_.add_argument("--cutoff-week", type=int, default=74)
    ap_.add_argument("--leak-threshold", type=float, default=0.5)
    args = ap_.parse_args()
    final = run_g2(selection_path=args.selection, output_dir=args.output_dir, cutoff_week=args.cutoff_week, leak_threshold=args.leak_threshold)
    print(final.group_by("g2_decision").len().sort("g2_decision"))
    print(f"decision: {G2_COLUMNS}\nvocabulary: {G2_VOCABULARY}\nscreen: {G2_SCREEN}")


if __name__ == "__main__":
    main()
