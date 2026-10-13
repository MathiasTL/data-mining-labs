"""Despacho por familia (decisión G1, punto 1) y comando único de reproducción (E7).

    POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.build --config configs/preparation.json [--from-scratch]

Sin ``--from-scratch`` audita las piezas de E4 existentes y regenera E5, E6 y la validación. Con
``--from-scratch`` además reconstruye las 36 piezas de E4 desde los Parquet originales. Cada paso
corre en su propio proceso, en serie; al final escribe el manifiesto final (``E7_manifest.json``),
``run_metadata.csv`` y ``final_schema.csv``.

``build_family_features(family, base, split, *, cutoff_week, output_dir)`` delega en el constructor
registrado de cada familia. Las familias se registran en ``BUILDERS``; P1 amplía este registro
con depth 0, productos y bureau. Hoy: applprev/person, bureau (C4-C5) y productos/registros (C6-C7).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import polars as pl

from dm_credit.preparation.contracts import FamilyFeatureArtifact
from dm_credit.preparation.families import (
    applications_people,
    applications_people_g2,
    bureau,
    depth0,
    products_registry,
)

_REGISTRIES = (applications_people.BUILDERS, bureau.BUILDERS, products_registry.BUILDERS, depth0.BUILDERS)
_names = [name for registry in _REGISTRIES for name in registry]
if len(_names) != len(set(_names)):
    raise RuntimeError(f"family name registered twice: {sorted(n for n in set(_names) if _names.count(n) > 1)}")
BUILDERS = {name: builder for registry in _REGISTRIES for name, builder in registry.items()}
G2_CONFIG_INDEX = {"applprev": 0, "person": 1}  # posición de la familia en load_g2_configs()


def build_family_features(
    family: str,
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    **options,
) -> FamilyFeatureArtifact:
    if family not in BUILDERS:
        raise ValueError(f"unknown family {family!r}; available: {sorted(BUILDERS)}")
    if options.get("config") is None and family in G2_CONFIG_INDEX:
        # sin configuración explícita solo se usan las columnas aprobadas en G2; si falta la decisión, error
        options["config"] = applications_people_g2.load_g2_configs()[G2_CONFIG_INDEX[family]]
    return BUILDERS[family](base, split, cutoff_week=cutoff_week, output_dir=output_dir, **options)



# ------------------------------------------------------------------------------- E7
ROOT = Path(__file__).resolve().parents[3]
REQUIRED_CONFIG = ("cutoff_week", "polars_max_threads", "artifact_root", "versions", "e4", "catalog", "decisions", "results_dir")
CONTROLS = ("case_id", "WEEK_NUM", "date_decision", "MONTH", "target")


def load_config(path: Path) -> dict:
    """Configuración versionada de la preparación; una clave ausente es un error."""
    cfg = json.loads(Path(path).read_text())
    missing = [k for k in REQUIRED_CONFIG if k not in cfg]
    if missing:
        raise ValueError(f"config {path}: missing keys {missing}")
    if set(cfg["versions"]) != {"e4", "e5", "e6"}:
        raise ValueError("config: versions must define e4, e5 and e6")
    return cfg


def _dirs(cfg: dict) -> dict[str, Path]:
    from dm_credit import io

    root = io.data_dir() / cfg["artifact_root"]
    return {k: root / v for k, v in cfg["versions"].items()}


def merge_checkpoints(e4_dir: Path) -> None:
    """Junta los checkpoints de bureau de cada salida en ``<E4>/checkpoints`` (lo que lee su auditor)."""
    target = Path(e4_dir) / "checkpoints"
    for cache in sorted(Path(e4_dir).glob("*/*/_partial_cache")):
        shutil.copytree(cache, target, dirs_exist_ok=True)
        shutil.rmtree(cache)


@dataclass
class Step:
    name: str
    module: str | None
    args: list[str]
    env: dict[str, str]
    action: Callable[[], None] | None = field(default=None, repr=False)

    @property
    def command(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.args] if self.module else [f"<interno: {self.name}>"]


def plan_steps(cfg: dict, *, from_scratch: bool) -> list[Step]:
    d = _dirs(cfg)
    env = {"POLARS_MAX_THREADS": str(cfg["polars_max_threads"])}
    steps = []
    if from_scratch:
        steps.append(Step("e4_rebuild", "dm_credit.preparation.e4_rebuild",
                          ["--all", "--max-rss-mib", str(cfg["e4"]["max_rss_mib"]), "--timeout-seconds", str(cfg["e4"]["timeout_seconds"])], env))
        steps.append(Step("e4_checkpoints", None, [], env, action=lambda: merge_checkpoints(d["e4"])))
    steps += [
        Step("e4_audit", "dm_credit.preparation.e4", [], env),
        Step("e4_checkpoint_audit", "dm_credit.preparation.e4_shards", ["--checkpoint-dir", str(d["e4"] / "checkpoints")], env),
        Step("e5", "dm_credit.preparation.e5", ["--artifacts", str(d["e4"]), "--output-dir", str(d["e5"])], env),
        Step("e6", "dm_credit.preparation.e6", ["--masters", str(d["e5"]), "--output-dir", str(d["e6"])], env),
        Step("validation", "dm_credit.preparation.validation", ["--artifacts", str(d["e4"]), "--matrix-dir", str(d["e5"])], env),
    ]
    return steps


def subprocess_runner(step: Step, log: Path) -> int:
    if step.action is not None:
        step.action()
        log.write_text("ok\n")
        return 0
    with log.open("w") as stream:
        return subprocess.run(step.command, cwd=ROOT, env={**os.environ, **step.env}, stdout=stream, stderr=subprocess.STDOUT).returncode


def run_pipeline(cfg: dict, *, from_scratch: bool, runner=subprocess_runner, log_dir: Path) -> list[dict]:
    """Ejecuta los pasos en serie y se detiene en el primer fallo."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if from_scratch and runner is subprocess_runner:
        for path in _dirs(cfg).values():  # E4 --all exige carpetas sin manifiestos previos
            if path.exists():
                shutil.rmtree(path)
    records = []
    for step in plan_steps(cfg, from_scratch=from_scratch):
        log = log_dir / f"{step.name}.log"
        started = time.perf_counter()
        code = runner(step, log)
        records.append({"name": step.name, "command": step.command, "returncode": code,
                        "seconds": round(time.perf_counter() - started, 2), "log": str(log)})
        print(f"{step.name}: {'ok' if code == 0 else 'falló'} ({records[-1]['seconds']} s)", flush=True)
        if code != 0:
            raise RuntimeError(f"step {step.name} failed with exit code {code}; see {log}")
    return records


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def final_schema(train_matrix: pl.DataFrame, catalog: pl.DataFrame) -> pl.DataFrame:
    """Esquema final ordenado: controles y características con su rol y tipo lógico."""
    from dm_credit.preparation.contracts import logical_type

    info = {r["name"]: r for r in catalog.iter_rows(named=True)}
    rows = []
    for i, col in enumerate(train_matrix.columns):
        if col in CONTROLS:
            role, logical = "control", logical_type(train_matrix.schema[col], col)
        else:
            role, logical = info[col]["role"], info[col]["logical_type"]
        rows.append({"position": i, "column": col, "dtype": str(train_matrix.schema[col]), "role": role, "logical_type": logical})
    return pl.DataFrame(rows)


def final_manifest(*, config_path: Path, bases, e4_artifacts, e5_report: pl.DataFrame, e6_report: pl.DataFrame, e6_dir: Path, steps: list[dict]) -> dict:
    """Manifiesto final de la preparación: configuración, particiones y sumas de verificación de cada salida."""
    from dm_credit.preparation.families.applications_people import _base_fingerprint

    cfg = load_config(config_path)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        commit = None
    partitions = {}
    for split, base in bases.items():
        partitions[split] = {
            "cases": base.height,
            "positives": int(base["target"].sum()) if "target" in base.columns else None,
            "week_min": int(base["WEEK_NUM"].min()) if base.height else None,
            "week_max": int(base["WEEK_NUM"].max()) if base.height else None,
            "base_sha256": _base_fingerprint(base),
        }
    e4 = []
    for split in ("train", "oot", "test"):
        for art in sorted(e4_artifacts[split], key=lambda a: a.family):
            m = json.loads(Path(art.manifest_path).read_text())
            e4.append({"family": art.family, "split": split, "manifest": _rel(art.manifest_path),
                       "manifest_sha256": _sha(art.manifest_path), "data_sha256": m["data_sha256"], "rows": m["row_count"]})
    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "python": platform.python_version(),
        "polars": pl.__version__,
        "config": _rel(config_path),
        "config_sha256": _sha(config_path),
        "cutoff_week": cfg["cutoff_week"],
        "inputs": {path: _sha(ROOT / path) for path in [cfg["catalog"], *cfg["decisions"]]},
        "partitions": partitions,
        "artifacts": {
            "e4": e4,
            "e5": [{k: r[k] for k in ("split", "rows", "columns", "features", "data_sha256")} | {"data_path": _rel(r["data_path"])} for r in e5_report.iter_rows(named=True)],
            "e6": {
                "outputs": [{k: r[k] for k in ("view", "split", "rows", "predictors", "predictor_nulls", "data_sha256")} | {"data_path": _rel(r["data_path"])} for r in e6_report.iter_rows(named=True)],
                "state_sha256": _sha(Path(e6_dir) / "matrix_views_state.json"),
                "spec_sha256": _sha(Path(e6_dir) / "matrix_views_spec.csv"),
            },
        },
        "steps": steps,
        "regenerate": "POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.build --config configs/preparation.json --from-scratch",
    }


def _run_metadata(manifest: dict) -> pl.DataFrame:
    rows = [("config", manifest["config"]), ("config_sha256", manifest["config_sha256"]), ("git_commit", str(manifest["git_commit"])),
            ("polars", manifest["polars"]), ("cutoff_week", str(manifest["cutoff_week"])), ("regenerate", manifest["regenerate"])]
    for split, p in manifest["partitions"].items():
        rows += [(f"{split}.cases", str(p["cases"])), (f"{split}.positives", str(p["positives"])), (f"{split}.base_sha256", p["base_sha256"])]
    for r in manifest["artifacts"]["e5"]:
        rows += [(f"e5.{r['split']}.path", r["data_path"]), (f"e5.{r['split']}.sha256", r["data_sha256"])]
    for r in manifest["artifacts"]["e6"]["outputs"]:
        rows += [(f"e6.{r['view']}.{r['split']}.path", r["data_path"]), (f"e6.{r['view']}.{r['split']}.sha256", r["data_sha256"])]
    rows += [("e6.state_sha256", manifest["artifacts"]["e6"]["state_sha256"]), ("e4.artifacts", str(len(manifest["artifacts"]["e4"])))]
    rows += [(f"step.{s['name']}.seconds", str(s["seconds"])) for s in manifest["steps"]]
    return pl.DataFrame(rows, schema=["key", "value"], orient="row")


def main() -> None:
    from dm_credit import io
    from dm_credit.preparation.split import partition_base
    from dm_credit.preparation.validation import load_published

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "preparation.json")
    ap.add_argument("--from-scratch", action="store_true", help="reconstruir también las 36 piezas de E4")
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = ROOT / cfg["results_dir"]
    steps = run_pipeline(cfg, from_scratch=args.from_scratch, log_dir=io.data_dir() / cfg["artifact_root"] / "E7-logs")

    d = _dirs(cfg)
    parts = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=cfg["cutoff_week"])
    bases = {"train": parts.train, "oot": parts.oot, "test": parts.test}
    manifest = final_manifest(
        config_path=args.config, bases=bases, e4_artifacts=load_published([d["e4"]]),
        e5_report=pl.read_csv(results / "E5_master_tables.csv"), e6_report=pl.read_csv(results / "E6_views.csv"),
        e6_dir=d["e6"], steps=steps,
    )
    (results / "E7_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    _run_metadata(manifest).write_csv(results / "run_metadata.csv")
    train = pl.read_parquet(d["e5"] / "feature_matrix__train.parquet")
    final_schema(train, pl.read_csv(ROOT / cfg["catalog"])).write_csv(results / "final_schema.csv")
    print(f"manifiesto final: {results / 'E7_manifest.json'}")


if __name__ == "__main__":
    main()
