"""Contratos compartidos de preparación (decisión G1, secciones 3 y 4).

    - ``FeatureDefinition``: una fila del catálogo de características.
    - ``FamilyFeatureArtifact``: salida de una familia (Parquet + manifiesto JSON).
    - ``FeatureMatrixArtifact``: salida del ensamblado.
    - ``write_family_features``: valida todo **antes** de escribir y publica de forma atómica.

Gramática de nombres: ``<fuente>__<variable_o_metrica>__<agregacion>__<ventana>``; la cobertura
usa ``<fuente>__n_records`` y ``<fuente>__source_present``; ``__missing`` solo lo crea D2.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping

import polars as pl

from dm_credit.baseline import RESERVED
from dm_credit.preparation.coercion import LOGICAL_TYPES

KEY = "case_id"
FORMAT_VERSION = 1
SPLITS = ("train", "oot", "test")
ROLES = ("predictor", "cobertura")
WINDOWS = ("all", "3m", "6m", "12m", "source_defined")
COVERAGE = ("n_records", "source_present")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(__[a-z0-9_]+){1,4}$")
DTYPE_TO_LOGICAL = {
    pl.Boolean: "bool",
    pl.String: "string",
    pl.Date: "date",
    pl.Float32: "float",
    pl.Float64: "float",
}


def parse_feature_name(name: str) -> dict[str, str]:
    """Descompone un nombre de característica según la gramática de G1."""
    if not NAME_RE.match(name):
        raise ValueError(f"invalid feature name {name!r}: does not follow the G1 grammar")
    parts = name.split("__")
    source, rest = parts[0], parts[1:]
    if rest[-1] == "missing" and len(rest) >= 3:  # sufijo de ausencia (D2)
        rest = rest[:-1]
    if len(rest) == 1:
        if rest[0] not in COVERAGE:
            raise ValueError(f"invalid feature name {name!r}: two-segment names must be coverage ({list(COVERAGE)})")
        return {"source_table": source, "variable": "", "aggregation": rest[0], "window": "all"}
    if rest[-1] not in WINDOWS:
        raise ValueError(f"invalid feature name {name!r}: last segment must be a window {list(WINDOWS)}")
    if len(rest) == 2:  # fuente__metrica__ventana
        return {"source_table": source, "variable": "", "aggregation": rest[0], "window": rest[1]}
    if len(rest) == 3:
        return {"source_table": source, "variable": rest[0], "aggregation": rest[1], "window": rest[2]}
    raise ValueError(f"invalid feature name {name!r}: too many segments")


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    role: Literal["predictor", "cobertura"]
    logical_type: Literal["date", "int", "float", "bool", "string"]
    source_table: str
    source_columns: tuple[str, ...]
    aggregation: str
    window: str
    temporal_rule: str
    description: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {list(ROLES)}, got {self.role!r}")
        if self.logical_type not in LOGICAL_TYPES:
            raise ValueError(f"logical_type must be one of {sorted(LOGICAL_TYPES)}, got {self.logical_type!r}")
        parse_feature_name(self.name)


@dataclass(frozen=True)
class FamilyFeatureArtifact:
    family: str
    split: Literal["train", "oot", "test"]
    data_path: Path
    manifest_path: Path
    row_count: int
    case_count: int
    feature_schema: tuple[FeatureDefinition, ...]


@dataclass(frozen=True)
class FeatureMatrixArtifact:
    split: Literal["train", "oot", "test"]
    data_path: Path | str
    manifest_path: Path | str
    row_count: int
    control_columns: tuple[str, ...]
    feature_schema: tuple[FeatureDefinition, ...]
    input_manifests: tuple[Path | str, ...]


def _logical_type(dtype: pl.DataType, name: str) -> str:
    if dtype.is_integer():
        return "int"
    if dtype in DTYPE_TO_LOGICAL:
        return DTYPE_TO_LOGICAL[dtype]
    raise ValueError(f"unsupported dtype {dtype} for feature {name!r}")


def describe_features(
    frame: pl.DataFrame,
    *,
    temporal_rule: str,
    source_columns: Mapping[str, str] | None = None,
) -> tuple[FeatureDefinition, ...]:
    """Una ``FeatureDefinition`` por columna (sin ``case_id``), en el orden de la tabla."""
    source_columns = source_columns or {}
    out = []
    for name in (c for c in frame.columns if c != KEY):
        p = parse_feature_name(name)
        is_coverage = p["aggregation"] in COVERAGE or name.endswith("__missing")
        cols = (source_columns.get(p["variable"], p["variable"]),) if p["variable"] else ()
        what = f"{p['aggregation']} de {p['variable']}" if p["variable"] else p["aggregation"]
        out.append(
            FeatureDefinition(
                name=name,
                role="cobertura" if is_coverage else "predictor",
                logical_type=_logical_type(frame.schema[name], name),
                source_table=p["source_table"],
                source_columns=cols,
                aggregation=p["aggregation"],
                window=p["window"],
                temporal_rule=temporal_rule,
                description=f"{what} en {p['source_table']} (ventana {p['window']})",
            )
        )
    return tuple(out)


def _validate(features: pl.DataFrame, definitions: tuple[FeatureDefinition, ...], split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if KEY not in features.columns:
        raise ValueError(f"missing required column {KEY}")
    if features[KEY].null_count():
        raise ValueError(f"{KEY} contains nulls")
    if features[KEY].n_unique() != features.height:
        raise ValueError(f"{KEY} is not unique: an aggregate must have one row per case")
    reserved = [c for c in features.columns if c in RESERVED and c != KEY]
    if reserved:
        raise ValueError(f"reserved columns cannot be features: {reserved}")
    if features.columns != [KEY, *(d.name for d in definitions)]:
        raise ValueError("schema mismatch: columns must be case_id followed by the definitions in the same order")
    if not features[KEY].is_sorted():
        raise ValueError(f"{KEY} must be sorted ascending (G1: ordered aggregates)")


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_family_features(
    features: pl.DataFrame,
    definitions: tuple[FeatureDefinition, ...],
    *,
    family: str,
    split: str,
    output_dir: Path,
    exclusions: Mapping[str, int] | None = None,
    rules_version: str = "unversioned",
    inputs: Mapping[str, str] | None = None,
) -> FamilyFeatureArtifact:
    """Valida y publica un agregado y su manifiesto. Si falla una validación no se escribe nada."""
    definitions = tuple(definitions)
    _validate(features, definitions, split)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"{family}__{split}.parquet"
    manifest_path = output_dir / f"{family}__{split}.manifest.json"
    manifest = {
        "format_version": FORMAT_VERSION,
        "rules_version": rules_version,
        "family": family,
        "split": split,
        "data_path": data_path.name,
        "row_count": features.height,
        "case_count": features[KEY].n_unique(),
        "features": [asdict(d) for d in definitions],
        "exclusions": dict(exclusions or {}),
        "inputs": dict(inputs or {}),
        "validations": [
            "case_id no nulo",
            "case_id único",
            "sin columnas reservadas",
            "esquema igual a las definiciones y en el mismo orden",
            "ordenado por case_id",
        ],
    }
    _atomic_write(data_path, lambda p: features.write_parquet(p))
    _atomic_write(manifest_path, lambda p: p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)))
    return FamilyFeatureArtifact(
        family=family,
        split=split,  # type: ignore[arg-type]
        data_path=data_path,
        manifest_path=manifest_path,
        row_count=features.height,
        case_count=manifest["case_count"],
        feature_schema=definitions,
    )


def load_family_artifact(manifest_path: Path) -> FamilyFeatureArtifact:
    """Reconstruye el artefacto desde su manifiesto; falla si está ausente o malformado."""
    manifest_path = Path(manifest_path)
    try:
        m = json.loads(manifest_path.read_text())
        schema = tuple(FeatureDefinition(**{**f, "source_columns": tuple(f["source_columns"])}) for f in m["features"])
        return FamilyFeatureArtifact(
            family=m["family"],
            split=m["split"],
            data_path=manifest_path.parent / m["data_path"],
            manifest_path=manifest_path,
            row_count=m["row_count"],
            case_count=m["case_count"],
            feature_schema=schema,
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest {manifest_path} is missing or malformed: {exc}") from exc
