import numpy as np
import polars as pl
import pytest

from dm_credit.preparation.families import applications_people as ap
from dm_credit.preparation.families import applications_people_g2 as g2


def selection(rows):
    return pl.DataFrame(rows, schema={"family": pl.String, "column": pl.String, "dtype": pl.String, "action": pl.String, "reason": pl.String, "note": pl.String}, orient="row")


def sel_row(family, column, dtype, action="candidato_G2"):
    return (family, column, dtype, action, "", "")


def tables(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a1 = pl.DataFrame({
        "case_id": np.arange(n), "num_group1": np.zeros(n, dtype=int),
        "credamount_590A": rng.normal(size=n), "tenor_203L": rng.normal(size=n),
        "rare_97A": pl.Series([1.0 if i % 40 == 0 else None for i in range(n)], dtype=pl.Float64),  # 2,5 % de cobertura
        "credtype_587L": rng.choice(["CAL", "REL", "XXX"], size=n, p=[0.6, 0.39, 0.01]),
        "profession_152M": [f"p{i}" for i in range(n)],  # alta cardinalidad
        "isbidproduct_390L": rng.random(n) < 0.5,
        "approvaldate_319D": ["2020-01-01"] * n,
    })
    def alt(main, contact, dtype):
        """Valores alternados persona principal / contacto, con tipo explícito."""
        return pl.Series([main if i % 2 == 0 else contact for i in range(2 * n)], dtype=dtype)

    p1 = pl.DataFrame({
        "case_id": np.repeat(np.arange(n), 2), "num_group1": np.tile([0, 1], n),
        "mainoccupationinc_384A": alt(1.0, None, pl.Float64),
        "incometype_1044T": alt("SALARIED", None, pl.String),
        "role_1084L": alt("AA", "BB", pl.String),
        "relationshiptoclient_415T": alt(None, "SPOUSE", pl.String),
        "safeguarantyflag_411L": alt(True, None, pl.Boolean),
        "sex_738L": alt("M", None, pl.String),
        "birth_259D": alt("1980-01-01", None, pl.String),
    })
    return {"applprev_1": a1, "person_1": p1}


SEL = selection([
    sel_row("applprev_1", "credamount_590A", "Float64"),
    sel_row("applprev_1", "tenor_203L", "Float64"),
    sel_row("applprev_1", "rare_97A", "Float64"),
    sel_row("applprev_1", "credtype_587L", "String"),
    sel_row("applprev_1", "profession_152M", "String"),
    sel_row("applprev_1", "isbidproduct_390L", "Boolean"),
    sel_row("applprev_1", "approvaldate_319D", "String", "transformar_as_of"),
    sel_row("applprev_1", "dtlastpmtallstes_3545839D", "String", "transformar_as_of"),
    sel_row("applprev_1", "status_219L", "String", "bloquear_hasta_G2"),
    sel_row("applprev_1", "case_id", "Int64", "excluir"),
    sel_row("applprev_1", "creationdate_885D", "String", "referencia_as_of"),
    sel_row("person_1", "mainoccupationinc_384A", "Float64"),
    sel_row("person_1", "incometype_1044T", "String"),
    sel_row("person_1", "role_1084L", "String"),
    sel_row("person_1", "relationshiptoclient_415T", "String"),
    sel_row("person_1", "safeguarantyflag_411L", "Boolean"),
    sel_row("person_1", "sex_738L", "String"),
    sel_row("person_1", "birth_259D", "String", "transformar_as_of"),
])


@pytest.fixture
def plan():
    return g2.classify_candidates(SEL, tables())


def decision(plan, family, column):
    return plan.filter((pl.col("family") == family) & (pl.col("column") == column)).row(0, named=True)


def test_every_selection_row_gets_one_decision(plan):
    assert plan.height == SEL.height
    assert plan.select(["family", "column"]).is_duplicated().sum() == 0
    assert set(plan["g2_decision"]) <= {"aprobada", "excluida", "bloqueada", "aplazada", "pendiente_filtro"}


def test_b4_exclusions_and_blocks_are_kept(plan):
    assert decision(plan, "applprev_1", "case_id")["g2_decision"] == "excluida"
    assert decision(plan, "applprev_1", "status_219L")["g2_decision"] == "bloqueada"


def test_g2_overrides_block_future_looking_dates_and_exclude_sensitive(plan):
    r = decision(plan, "applprev_1", "dtlastpmtallstes_3545839D")
    assert r["g2_decision"] == "bloqueada" and "futuro" in r["reason"]
    s = decision(plan, "person_1", "sex_738L")
    assert s["g2_decision"] == "excluida" and "sensible" in s["reason"]


def test_low_coverage_columns_are_rejected(plan):
    r = decision(plan, "applprev_1", "rare_97A")
    assert r["g2_decision"] == "excluida" and "cobertura" in r["reason"]


def test_core_roles_follow_the_dtype(plan):
    assert decision(plan, "applprev_1", "credamount_590A")["core_role"] == "numeric"
    assert decision(plan, "applprev_1", "credtype_587L")["core_role"] == "distinct_and_shares"
    assert decision(plan, "applprev_1", "profession_152M")["core_role"] == "distinct"  # alta cardinalidad: sin proporciones
    assert decision(plan, "applprev_1", "isbidproduct_390L")["core_role"] == "bool"
    assert decision(plan, "applprev_1", "approvaldate_319D")["core_role"] == "date"
    assert decision(plan, "applprev_1", "creationdate_885D")["core_role"] == "reference"


def test_person_columns_are_split_into_main_and_contact_attributes(plan):
    assert decision(plan, "person_1", "mainoccupationinc_384A")["core_role"] == "main_numeric"
    assert decision(plan, "person_1", "incometype_1044T")["core_role"] == "main_categorical"
    assert decision(plan, "person_1", "relationshiptoclient_415T")["core_role"] == "contact_shares"
    assert decision(plan, "person_1", "birth_259D")["core_role"] == "birth"


def test_unsupported_roles_are_deferred_not_silently_dropped(plan):
    r = decision(plan, "person_1", "safeguarantyflag_411L")
    assert r["g2_decision"] == "aplazada" and r["core_role"] == "none"


def test_candidates_start_as_pending_the_leakage_screen(plan):
    assert decision(plan, "applprev_1", "credamount_590A")["g2_decision"] == "pendiente_filtro"
    assert decision(plan, "person_1", "mainoccupationinc_384A")["g2_decision"] == "pendiente_filtro"


def test_vocabulary_uses_frequent_categories_only_and_is_deterministic():
    plan = g2.classify_candidates(SEL, tables())
    v = g2.fit_vocabulary(plan, tables())
    cats = v.filter((pl.col("family") == "applprev_1") & (pl.col("column") == "credtype_587L"))["category"].to_list()
    assert cats == ["CAL", "REL"]  # XXX (1 %) no alcanza la frecuencia mínima ni es de las más frecuentes
    assert v.equals(g2.fit_vocabulary(plan, tables()))
    assert not v.filter(pl.col("column") == "profession_152M").height


def test_vocabulary_for_contact_columns_uses_contact_rows_only(plan):
    v = g2.fit_vocabulary(plan, tables())
    cats = v.filter(pl.col("column") == "relationshiptoclient_415T")["category"].to_list()
    assert cats == ["SPOUSE"]


def test_configs_are_built_from_the_plan_and_the_vocabulary(plan):
    vocab = g2.fit_vocabulary(plan, tables())
    approved = plan.with_columns(g2_decision=pl.when(pl.col("g2_decision") == "pendiente_filtro").then(pl.lit("aprobada")).otherwise(pl.col("g2_decision")))
    app, people = g2.configs_from_decisions(approved, vocab)
    assert isinstance(app, ap.ApplicationsConfig) and isinstance(people, ap.PeopleConfig)
    assert set(app.numeric) == {"credamount_590A", "tenor_203L"}
    assert app.numeric["credamount_590A"] == ("count", "min", "max", "mean", "sum")  # importe: con suma
    assert app.numeric["tenor_203L"] == ("count", "min", "max", "mean")
    assert set(app.categorical_distinct) == {"credtype_587L", "profession_152M"}
    assert app.category_shares == {"credtype_587L": ("CAL", "REL")}
    assert app.bool_cols == ("isbidproduct_390L",) and app.date_cols == ("approvaldate_319D",)
    assert people.main_numeric == ("mainoccupationinc_384A",) and people.main_categorical == ("incometype_1044T", "role_1084L")
    assert people.contact_category_shares == {"relationshiptoclient_415T": ("SPOUSE",)}
    assert people.birth_col == "birth_259D"


def test_only_approved_rows_reach_the_configs(plan):
    vocab = g2.fit_vocabulary(plan, tables())
    app, people = g2.configs_from_decisions(plan, vocab)  # nada aprobado todavía
    assert not app.numeric and not app.categorical_distinct and not app.bool_cols and not app.date_cols
    assert people.main_numeric == () and people.birth_col is None


def test_screen_results_reject_flagged_columns_and_approve_the_rest(plan):
    screen = pl.DataFrame({
        "feature": ["applprev_1__credamount_590a__mean__all", "applprev_1__tenor_203l__sum__all", "applprev_1__tenor_203l__mean__all"],
        "flag": ["ok", "leak_suspect", "ok"], "gini_all": [0.1, 0.8, 0.05],
    })
    out = g2.apply_screen(plan, screen)
    assert decision(out, "applprev_1", "credamount_590A")["g2_decision"] == "aprobada"
    r = decision(out, "applprev_1", "tenor_203L")
    assert r["g2_decision"] == "excluida" and "filtro de fuga" in r["reason"] and "0.8" in r["reason"]


def test_unscreened_candidates_stay_pending_not_approved(plan):
    out = g2.apply_screen(plan, pl.DataFrame({"feature": [], "flag": [], "gini_all": []}, schema={"feature": pl.String, "flag": pl.String, "gini_all": pl.Float64}))
    assert decision(out, "applprev_1", "credamount_590A")["g2_decision"] == "pendiente_filtro"


# ------------------------------------------------------------------ ejecutor y configuración aprobada
import preparation_artifacts as pa


@pytest.fixture
def sample_data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    pa.write_sample_data(root)
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    return root


SAMPLE_SELECTION = selection([
    sel_row("applprev_1", "credamount_590A", "Float64"),
    sel_row("applprev_1", "credtype_587L", "String"),
    sel_row("applprev_1", "isbidproduct_390L", "Boolean"),
    sel_row("applprev_1", "approvaldate_319D", "String", "transformar_as_of"),
    sel_row("applprev_1", "creationdate_885D", "String", "referencia_as_of"),
    sel_row("applprev_1", "status_219L", "String", "bloquear_hasta_G2"),
    sel_row("person_1", "mainoccupationinc_384A", "Float64"),
    sel_row("person_1", "birth_259D", "String", "transformar_as_of"),
    sel_row("person_1", "empl_employedfrom_271D", "String", "transformar_as_of"),
])


def run(tmp_path, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sel_path = tmp_path / "sel.csv"
    SAMPLE_SELECTION.write_csv(sel_path)
    out = tmp_path / "out"
    plan = g2.run_g2(
        selection_path=sel_path, output_dir=out / "art", columns_path=out / "columns.csv",
        vocabulary_path=out / "vocab.csv", screen_path=out / "screen.csv", cutoff_week=pa.CUTOFF, min_rows=2, **kw,
    )
    return plan, out


def test_run_g2_writes_the_three_reports(sample_data, tmp_path):
    plan, out = run(tmp_path)
    cols = pl.read_csv(out / "columns.csv")
    assert cols.columns == list(g2.PLAN_SCHEMA) and cols.height == SAMPLE_SELECTION.height
    assert set(cols["g2_decision"]) <= {"aprobada", "excluida", "bloqueada", "aplazada", "pendiente_filtro"}
    assert (out / "vocab.csv").exists() and pl.read_csv(out / "screen.csv").columns[0] == "feature"
    assert cols.filter(pl.col("column") == "status_219L")["g2_decision"].item() == "bloqueada"


def test_run_g2_is_deterministic(sample_data, tmp_path):
    run(tmp_path / "a")
    run(tmp_path / "b")
    for f in ("columns.csv", "vocab.csv", "screen.csv"):
        assert (tmp_path / "a" / "out" / f).read_text() == (tmp_path / "b" / "out" / f).read_text()


def test_run_g2_only_screens_the_train_partition(sample_data, tmp_path):
    _, out = run(tmp_path)
    screen = pl.read_csv(out / "screen.csv")
    assert screen["n_rows"].unique().to_list() == [2]  # casos 1 y 2 (train con corte 11); OOT y test no entran


def test_load_g2_configs_round_trip(sample_data, tmp_path):
    plan, out = run(tmp_path)
    app, people = g2.load_g2_configs(out / "columns.csv", out / "vocab.csv")
    approved = plan.filter(pl.col("g2_decision") == "aprobada")
    assert set(app.numeric) == set(approved.filter(pl.col("core_role") == "numeric")["column"])
    assert set(people.main_numeric) == set(approved.filter(pl.col("core_role") == "main_numeric")["column"])


def test_missing_decision_file_is_an_error_not_a_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="G2"):
        g2.load_g2_configs(tmp_path / "nope.csv", tmp_path / "nope2.csv")


def test_text_features_without_gini_are_approved_with_an_honest_reason(plan):
    screen = pl.DataFrame({"feature": ["person_1__incometype_1044t__main__all"], "flag": ["not_screened"], "gini_all": [None]}, schema={"feature": pl.String, "flag": pl.String, "gini_all": pl.Float64})
    r = decision(g2.apply_screen(plan, screen), "person_1", "incometype_1044T")
    assert r["g2_decision"] == "aprobada" and "texto" in r["reason"] and "no se midió" in r["reason"]


def test_dispatch_without_config_uses_only_the_g2_approved_columns(sample_data, tmp_path, monkeypatch):
    from dm_credit.preparation import build

    plan, out = run(tmp_path)
    monkeypatch.setattr(g2, "G2_COLUMNS", out / "columns.csv")
    monkeypatch.setattr(g2, "G2_VOCABULARY", out / "vocab.csv")
    parts = pa_partitions()
    art = build.build_family_features("applprev", parts["train"], "train", cutoff_week=pa.CUTOFF, output_dir=tmp_path / "built")
    approved = set(plan.filter((pl.col("family") == "applprev_1") & (pl.col("g2_decision") == "aprobada") & (pl.col("core_role") == "numeric"))["column"])
    names = {f.name for f in art.feature_schema}
    assert {f"applprev_1__{c.lower()}__count__all" for c in approved} <= names
    assert not any("status_219l" in n for n in names)  # bloqueada: jamás se construye


def test_dispatch_without_g2_files_fails_loudly_instead_of_using_unapproved_defaults(sample_data, tmp_path, monkeypatch):
    from dm_credit.preparation import build

    monkeypatch.setattr(g2, "G2_COLUMNS", tmp_path / "missing.csv")
    monkeypatch.setattr(g2, "G2_VOCABULARY", tmp_path / "missing2.csv")
    with pytest.raises(FileNotFoundError, match="G2"):
        build.build_family_features("person", pa_partitions()["train"], "train", cutoff_week=pa.CUTOFF, output_dir=tmp_path / "built")


def pa_partitions():
    from dm_credit import io
    from dm_credit.preparation.split import partition_base

    p = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=pa.CUTOFF)
    return {"train": p.train, "oot": p.oot, "test": p.test}


def test_numeric_column_with_only_insufficient_features_stays_pending(plan):
    screen = pl.DataFrame({"feature": ["applprev_1__tenor_203l__mean__all"], "flag": ["insufficient"], "gini_all": [None]}, schema={"feature": pl.String, "flag": pl.String, "gini_all": pl.Float64})
    r = decision(g2.apply_screen(plan, screen), "applprev_1", "tenor_203L")
    assert r["g2_decision"] == "pendiente_filtro"


def test_mixed_ok_and_insufficient_is_approved_without_calling_it_text(plan):
    screen = pl.DataFrame(
        {"feature": ["applprev_1__tenor_203l__mean__all", "applprev_1__tenor_203l__max__all"], "flag": ["ok", "insufficient"], "gini_all": [0.1, None]},
        schema={"feature": pl.String, "flag": pl.String, "gini_all": pl.Float64},
    )
    r = decision(g2.apply_screen(plan, screen), "applprev_1", "tenor_203L")
    assert r["g2_decision"] == "aprobada" and "texto" not in r["reason"]


def test_dispatch_with_explicit_none_config_still_uses_g2(sample_data, tmp_path, monkeypatch):
    from dm_credit.preparation import build

    monkeypatch.setattr(g2, "G2_COLUMNS", tmp_path / "missing.csv")
    monkeypatch.setattr(g2, "G2_VOCABULARY", tmp_path / "missing2.csv")
    with pytest.raises(FileNotFoundError, match="G2"):
        build.build_family_features("person", pa_partitions()["train"], "train", cutoff_week=pa.CUTOFF, output_dir=tmp_path / "built", config=None)
