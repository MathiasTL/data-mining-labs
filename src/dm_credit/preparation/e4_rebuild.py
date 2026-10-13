"""Regeneración E4 selectiva de C2/C3 obsoletos, serial y con presupuesto explícito.

Solo repara el caso observado: fingerprints de shards distintos. Otras causas o
familias requieren revisar su presupuesto, no habilitan una reconstrucción ciega.
Publica en E4-v1; conserva todos los artefactos anteriores. Nunca ejecuta E5/E6.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from dm_credit import io
from dm_credit.preparation.build import build_family_features
from dm_credit.preparation.e4 import EXPECTED_ROWS, ROOT, require, sha256_file
from dm_credit.preparation.families import applications_people as ap
from dm_credit.preparation.split import partition_base


def process_tree_rss(pid: int) -> int:
    """RSS externo del proceso y descendientes en KiB (Linux)."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        rss = next(
            (
                int(line.split()[1])
                for line in status.splitlines()
                if line.startswith("VmRSS:")
            ),
            0,
        )
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        return rss + sum(process_tree_rss(int(child)) for child in children)
    except (FileNotFoundError, ProcessLookupError):
        return 0


def available_kib() -> int:
    return next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )


def select_jobs(audit: dict) -> list[tuple[str, str]]:
    failures = [c for c in audit["checks"] if c["status"] != "ok"]
    jobs = []
    for run in audit["runs"]:
        if run["status"] in ("reused", "generated_validated"):
            continue
        scope = f"{run['family']}/{run['split']}"
        relevant = [
            c for c in failures if c["scope"] in ("global", run["family"], scope)
        ]
        require(
            run["family"] in ap.TABLES,
            f"Presupuesto no aprobado para regenerar {scope}",
        )
        require(
            bool(relevant)
            and all(
                c["check"] == "metadata_current_inputs"
                and "shards actuales:" in c["error"]
                for c in relevant
            ),
            f"Revisión manual necesaria: {scope}: {relevant}",
        )
        jobs.append((run["family"], run["split"]))
    # Primero la prueba local pequeña, después train/OOT por familia; nunca paralelo.
    return sorted(
        jobs,
        key=lambda job: (
            job[1] != "test",
            list(ap.TABLES).index(job[0]),
            list(EXPECTED_ROWS).index(job[1]),
        ),
    )


def build_one(family: str, split: str, output: Path) -> None:
    require(
        family in ap.TABLES and split in EXPECTED_ROWS, "Solo C2/C3 y splits conocidos"
    )
    parts = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=74)
    base = getattr(parts, split)
    require(base.height == EXPECTED_ROWS[split], "Conteo base incorrecto")
    artifact = build_family_features(
        family, base, split, cutoff_week=74, output_dir=output
    )
    require(artifact.case_count == base.height, "El builder perdió casos")


def rebuild(audit_path: Path, max_rss_mib: int, timeout_seconds: int) -> list[dict]:
    require(
        os.environ.get("POLARS_MAX_THREADS") == "1" and pl.thread_pool_size() == 1,
        "POLARS_MAX_THREADS=1 obligatorio",
    )
    require(max_rss_mib > 0 and timeout_seconds > 0, "Presupuestos deben ser positivos")
    require(
        Path("/usr/bin/time").is_file(),
        "Falta GNU time; no ejecutar sin medición externa",
    )
    audit = json.loads(audit_path.read_text())
    jobs = select_jobs(audit)
    results = ROOT / "results/preparation"
    log_root = io.data_dir() / "artifacts/preparation/E4-v1/logs"
    log_root.mkdir(parents=True, exist_ok=True)
    records_path = results / "E4_builds.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    for family, split in jobs:
        prior = [r for r in records if (r["family"], r["split"]) == (family, split)]
        if prior and prior[-1]["status"] == "blocked":
            print(
                f"{family}/{split}: bloqueo anterior conservado; no reintentar",
                flush=True,
            )
            continue
        output = io.data_dir() / "artifacts/preparation/E4-v1" / family / split
        manifest = output / f"{family}__{split}.manifest.json"
        # Una segunda ejecución debe auditar la salida publicada, no sobrescribirla.
        require(
            not manifest.exists(),
            f"Ya existe {manifest}; ejecutar auditor E4 antes de regenerar",
        )
        record = dict(
            family=family,
            split=split,
            status="blocked",
            manifest=str(manifest),
            max_rss_mib=max_rss_mib,
            timeout_seconds=timeout_seconds,
        )
        records.append(record)
        try:
            require(
                available_kib() >= (max_rss_mib + 1536) * 1024,
                "MemAvailable insuficiente para presupuesto + reserva 1536 MiB",
            )
            file_split = "test" if split == "test" else "train"
            inputs = [
                p
                for table in ap.TABLES[family]
                for p in io.shard_paths(file_split, table)
            ]
            inputs += list((ROOT / "src/dm_credit/preparation").rglob("*.py"))
            inputs += [ROOT / "src/dm_credit/io.py", ROOT / "uv.lock"]
            inputs += list(results.glob("G2_applications_people*.csv"))
            record["input_sha256"] = {str(p): sha256_file(p) for p in inputs}
            metrics = log_root / f"{family}__{split}.time.json"
            log = log_root / f"{family}__{split}.log"
            command = [
                "/usr/bin/time",
                "-f",
                '{"seconds":%e,"peak_rss_kib":%M}',
                "-o",
                str(metrics),
                sys.executable,
                "-m",
                "dm_credit.preparation.e4_rebuild",
                "--worker",
                family,
                split,
                "--output",
                str(output),
            ]
            record.update(
                command=command, log=str(log), metrics=str(metrics), status="running"
            )
            records_path.write_text(json.dumps(records, indent=2) + "\n")
            started = time.monotonic()
            with log.open("w") as stream:
                process = subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    while process.poll() is None:
                        record["observed_seconds"] = time.monotonic() - started
                        record["sampled_tree_peak_rss_kib"] = max(
                            record.get("sampled_tree_peak_rss_kib", 0),
                            process_tree_rss(process.pid),
                        )
                        require(
                            time.monotonic() - started < timeout_seconds,
                            "Tiempo límite excedido",
                        )
                        require(
                            record["sampled_tree_peak_rss_kib"] <= max_rss_mib * 1024,
                            "Presupuesto RSS excedido",
                        )
                        require(
                            available_kib() >= 1024 * 1024,
                            "Reserva de memoria del sistema inferior a 1 GiB",
                        )
                        time.sleep(0.1)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            record["returncode"] = process.returncode
            record["external_resources"] = json.loads(metrics.read_text())
            require(process.returncode == 0, f"Builder falló; ver {log}")
            require(
                all(
                    sha256_file(Path(p)) == digest
                    for p, digest in record["input_sha256"].items()
                ),
                "Inputs cambiaron durante construcción",
            )
            record.update(
                status="generated_pending_audit", manifest_sha256=sha256_file(manifest)
            )
            print(
                f"{family}/{split}: generado; {record['external_resources']}",
                flush=True,
            )
        except Exception as exc:
            record.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
            print(f"{family}/{split}: bloqueado: {exc}", flush=True)
        finally:
            records_path.write_text(json.dumps(records, indent=2) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", type=Path, default=ROOT / "results/preparation/E4_audit.json"
    )
    parser.add_argument("--max-rss-mib", type=int, default=5120)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--worker", nargs=2, metavar=("FAMILY", "SPLIT"), help=argparse.SUPPRESS
    )
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(
        os.environ.get("POLARS_MAX_THREADS") == "1", "POLARS_MAX_THREADS=1 obligatorio"
    )
    if args.worker:
        require(args.output is not None, "worker requiere output")
        build_one(*args.worker, args.output)
    else:
        records = rebuild(args.audit, args.max_rss_mib, args.timeout_seconds)
        if any(r["status"] == "blocked" for r in records):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
