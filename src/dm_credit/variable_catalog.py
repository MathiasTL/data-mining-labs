"""Genera un catálogo funcional de variables a partir de perfiles existentes.

No vuelve a leer los Parquet: combina ``results/profile_train_*/columns.csv`` con
``data/feature_definitions.csv`` y añade clasificaciones conservadoras para orientar
la revisión humana de identificadores, temporalidad y posibles fugas.

Uso:
    uv run -m dm_credit.variable_catalog
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

STRUCTURAL_COLUMNS = {"case_id", "num_group1", "num_group2"}
BASE_TEMPORAL_COLUMNS = {"date_decision", "MONTH", "WEEK_NUM"}
FAMILY_PATTERN = re.compile(
    r"^(?:train|test)_(applprev_[12]|base|credit_bureau_[ab]_[12]|debitcard_1|"
    r"deposit_1|other_1|person_[12]|static(?:_cb)?_0|tax_registry_[abc]_1)"
    r"(?:_\d+)?\.parquet$"
)
IDENTIFIER_TERMS = (
    "email",
    "mobile",
    "phone",
    "zip",
    "district",
    "address",
    "employername",
    "financialinstitution",
    "name_",
)
AVAILABILITY_TERMS = (
    "approval",
    "cancel",
    "debt",
    "dpd",
    "last",
    "overdue",
    "payment",
    "reject",
    "repayment",
    "status",
    "termination",
)


def _text(name: str, description: str | None) -> str:
    return f"{name} {description or ''}".lower()


def infer_function(name: str, dtype: str, description: str | None) -> str:
    """Infer a broad role; this is a navigation aid, not domain truth."""
    text = _text(name, description)
    if name in STRUCTURAL_COLUMNS:
        return "clave/índice estructural"
    if name == "target":
        return "objetivo"
    if name in BASE_TEMPORAL_COLUMNS or re.search(r"_\d+D$", name):
        return "temporal"
    if any(term in text for term in ("dpd", "overdue", "delinquen", "past due")):
        return "mora/incumplimiento histórico"
    if any(
        term in text
        for term in (
            "amount",
            "balance",
            "debt",
            "income",
            "instal",
            "limit",
            "turnover",
        )
    ):
        return "importe/capacidad financiera"
    if any(
        term in text
        for term in ("birth", "education", "famil", "gender", "person", "role")
    ):
        return "persona/demografía"
    if any(term in text for term in ("employ", "occupation", "salary", "tax")):
        return "empleo/ingresos"
    if any(
        term in text
        for term in ("contract", "credit", "product", "application", "card", "deposit")
    ):
        return "producto/contrato"
    if dtype in {"String", "Boolean", "Categorical"} or re.search(r"_\d+M$", name):
        return "atributo categórico"
    return "medida/atributo numérico"


def infer_temporal_relevance(name: str, description: str | None) -> str:
    text = _text(name, description)
    if name in STRUCTURAL_COLUMNS or name == "target":
        return "no aplica"
    if name in BASE_TEMPORAL_COLUMNS:
        return "tiempo de decisión/partición"
    if re.search(r"_\d+D$", name) or any(
        term in text for term in ("date", "day", "month", "year", "time")
    ):
        return "referencia temporal explícita"
    if any(term in text for term in ("previous", "histor", "last", "dpd", "overdue")):
        return "histórico implícito"
    return "sin referencia temporal explícita"


def infer_identifier_risk(name: str, description: str | None) -> str:
    text = _text(name, description)
    if name in STRUCTURAL_COLUMNS:
        return "identificador técnico"
    if any(term in text for term in IDENTIFIER_TERMS):
        return "posible cuasi-identificador/alta cardinalidad"
    return "no señalado"


def infer_leakage_review(name: str, description: str | None) -> str:
    text = _text(name, description)
    if name == "target":
        return "excluir de predictores"
    if name in STRUCTURAL_COLUMNS:
        return "excluir de predictores"
    if name in BASE_TEMPORAL_COLUMNS:
        return "reservar para tiempo/validación"
    if re.search(r"_\d+D$", name):
        return "revisar fecha contra date_decision"
    if any(term in text for term in AVAILABILITY_TERMS):
        return "revisar disponibilidad al decidir"
    return "sin señal automática; revisar dominio"


def inventory_columns(
    inventory: pl.DataFrame, excluded_families: set[str]
) -> list[dict[str, object]]:
    """Extract one logical train schema per family when no profile is available."""
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for item in inventory.filter(pl.col("split") == "train").iter_rows(named=True):
        match = FAMILY_PATTERN.match(str(item["file"]))
        if not match or match.group(1) in excluded_families:
            continue
        family = match.group(1)
        for field in str(item["types"]).split("; "):
            name, dtype = field.split(":", maxsplit=1)
            if (family, name) not in seen:
                rows.append(
                    {
                        "family": family,
                        "column": name,
                        "dtype": dtype,
                        "profile_source": "inventory_schema",
                    }
                )
                seen.add((family, name))
    return rows


def build_catalog(
    definitions: pl.DataFrame,
    profile_paths: list[Path],
    inventory: pl.DataFrame | None = None,
) -> pl.DataFrame:
    definitions_by_name = {
        row["Variable"]: row["Description"] for row in definitions.iter_rows(named=True)
    }
    source_rows: list[dict[str, object]] = []
    profiled_families: set[str] = set()
    for path in sorted(profile_paths):
        family = path.parent.name.removeprefix("profile_train_")
        profiled_families.add(family)
        for column in pl.read_csv(path).iter_rows(named=True):
            source_rows.append(
                {
                    "family": family,
                    "column": column["column"],
                    "dtype": column["dtype"],
                    "profile_source": "column_profile",
                    "pct_null": column.get("pct_null"),
                    "n_unique": column.get("n_unique", column.get("n_unique_approx")),
                }
            )
    if inventory is not None:
        source_rows.extend(inventory_columns(inventory, profiled_families))

    rows: list[dict[str, object]] = []
    for column in source_rows:
        name = str(column["column"])
        dtype = str(column["dtype"])
        description = definitions_by_name.get(name)
        rows.append(
            {
                "family": column["family"],
                "column": name,
                "dtype": dtype,
                "profile_source": column["profile_source"],
                "description": description,
                "definition_status": "official"
                if description
                else "not_in_feature_definitions",
                "function": infer_function(name, dtype, description),
                "temporal_relevance": infer_temporal_relevance(name, description),
                "identifier_risk": infer_identifier_risk(name, description),
                "leakage_review": infer_leakage_review(name, description),
                "pct_null": column.get("pct_null"),
                "n_unique": column.get("n_unique"),
            }
        )
    return pl.DataFrame(rows).sort("family", "column")


def summarize_catalog(catalog: pl.DataFrame) -> pl.DataFrame:
    return (
        catalog.group_by("family")
        .agg(
            pl.len().alias("columns"),
            (pl.col("definition_status") == "official")
            .sum()
            .alias("with_official_definition"),
            (pl.col("definition_status") != "official")
            .sum()
            .alias("without_official_definition"),
            (pl.col("temporal_relevance") != "sin referencia temporal explícita")
            .sum()
            .alias("temporal_or_structural"),
            (pl.col("identifier_risk") != "no señalado")
            .sum()
            .alias("identifier_review"),
            pl.col("leakage_review")
            .str.starts_with("revisar")
            .sum()
            .alias("availability_review"),
        )
        .sort("family")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--definitions", type=Path, default=Path("data/feature_definitions.csv")
    )
    parser.add_argument("--profiles", type=Path, default=Path("results"))
    parser.add_argument(
        "--inventory", type=Path, default=Path("results/parquet_inventory.csv")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/variable_catalog.csv")
    )
    args = parser.parse_args()

    paths = sorted(args.profiles.glob("profile_train_*/columns.csv"))
    if not paths:
        raise FileNotFoundError(f"No profile columns found below {args.profiles}")
    definitions = pl.read_csv(args.definitions)
    inventory = pl.read_csv(args.inventory) if args.inventory.is_file() else None
    catalog = build_catalog(definitions, paths, inventory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_csv(args.out)
    summary_path = args.out.with_name(f"{args.out.stem}_summary.csv")
    summarize_catalog(catalog).write_csv(summary_path)
    print(
        f"rows={catalog.height} families={catalog['family'].n_unique()} "
        f"official_definitions={(catalog['definition_status'] == 'official').sum()} "
        f"catalog={args.out} summary={summary_path}"
    )


if __name__ == "__main__":
    main()
