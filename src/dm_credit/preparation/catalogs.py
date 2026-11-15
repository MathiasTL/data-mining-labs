"""Regenera los catálogos de característica (por grupo de familias y C8) desde las piezas E4 publicadas.

    POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.catalogs --artifacts <E4> --results-dir results/preparation

Es el paso ``catalogs`` de ``build --from-scratch``: tras reconstruir E4, los catálogos que el auditor
E4 compara con las piezas se derivan de ellas. Sin ``--from-scratch`` el auditor compara contra los
catálogos existentes, así que detecta deriva en vez de ocultarla.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from dm_credit.preparation import contracts
from dm_credit.preparation.c8_catalog import write_catalog
from dm_credit.preparation.families import applications_people as ap
from dm_credit.preparation.families import bureau, depth0
from dm_credit.preparation.families import products_registry as products
from dm_credit.preparation.validation import load_published

# nombre del catálogo (``<nombre>_feature_catalog.csv``) -> familias que contiene; mismos grupos que e4.family_specs
GROUPS = {
    "depth0": tuple(depth0.SOURCES),
    "applications_people": tuple(ap.TABLES),
    "bureau": tuple(bureau.FAMILIES),
    "products": tuple(products.SOURCES),
}


def regenerate_catalogs(e4_dir: Path, results_dir: Path) -> pl.DataFrame:
    """Escribe los cuatro catálogos de grupo y el C8 consolidado; devuelve el C8."""
    published = load_published([Path(e4_dir)])
    artifacts = [a for split in published.values() for a in split]
    selected = {name: [a for a in artifacts if a.family in families] for name, families in GROUPS.items()}
    empty = sorted(name for name, group in selected.items() if not group)
    if empty:  # nada se escribe si falta algún grupo: no dejar catálogos a medias
        raise ValueError(f"no E4 pieces in {e4_dir} for catalog group(s): {empty}")
    results_dir = Path(results_dir)
    paths = [contracts.write_feature_catalog(group, results_dir / f"{name}_feature_catalog.csv") for name, group in selected.items()]
    return write_catalog(paths, results_dir / "C8_feature_catalog.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    c8 = regenerate_catalogs(args.artifacts, args.results_dir)
    print(f"catálogos regenerados; C8: {c8.height} características")


if __name__ == "__main__":
    main()
