"""Regeneración E4 serial y con presupuesto explícito.

Por defecto solo repara C2/C3 obsoletos por huella de shards. Con ``--all`` reconstruye las 12
familias en el orden del backlog (§7): la prueba pequeña de cada familia, profundidad 0, productos,
solicitudes y personas, bureau B, registros fiscales y bureau A al final. Funciona en Linux (``/proc``
y GNU time) y en macOS (``ps``, ``vm_stat`` y ``/usr/bin/time -l``). Publica en E4-v1. Nunca ejecuta E5/E6.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from dm_credit import io
from dm_credit.preparation.build import BUILDERS, build_family_features
from dm_credit.preparation.e4 import EXPECTED_ROWS, ROOT, require, sha256_file
from dm_credit.preparation.families import applications_people as ap
from dm_credit.preparation.families import bureau
from dm_credit.preparation.split import partition_base


A2 = "credit_bureau_a_2"
# Orden de ejecución real del backlog §7: de lo pequeño a lo pesado, bureau A al final.
FAMILY_ORDER = (
    "static_0", "static_cb_0", "debitcard_1", "deposit_1", "other_1", "applprev", "person",
    "credit_bureau_b", "tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1", "credit_bureau_a",
)
HAS_PROC = Path("/proc/meminfo").exists()


def family_tables(family: str) -> tuple[str, ...]:
    """Tablas físicas que lee una familia (sin ``credit_bureau_a_2``, que G2 deja fuera)."""
    if family in ap.TABLES:
        return ap.TABLES[family]
    if family in bureau.FAMILIES:
        return tuple(t for t in bureau.FAMILIES[family] if t != A2)
    return (family,)


def all_jobs() -> list[tuple[str, str]]:
    """Las 36 construcciones: primero test de cada familia, luego OOT y train en el orden del backlog."""
    missing = sorted(set(BUILDERS) - set(FAMILY_ORDER))
    require(not missing, f"familias sin lugar en el orden E4: {missing}")
    small = [(f, "test") for f in FAMILY_ORDER]
    heavy = [(f, s) for f in FAMILY_ORDER for s in ("oot", "train")]
    return small + heavy


def parse_bsd_time(log: str) -> dict:
    """Lee el resumen de ``/usr/bin/time -l`` (macOS): segundos reales y máximo RSS en bytes."""
    real = re.search(r"([\d.]+)\s+real", log)
    rss = re.search(r"(\d+)\s+maximum resident set size", log)
    if not (real and rss):
        raise ValueError("no se encontraron las métricas de /usr/bin/time -l en el log")
    return {"seconds": float(real.group(1)), "peak_rss_kib": int(rss.group(1)) // 1024}


def _ps_tree_rss(pid: int) -> int:
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    rows = [tuple(int(x) for x in line.split()) for line in out.splitlines() if line.strip()]
    children: dict[int, list[int]] = {}
    rss = {}
    for p, pp, r in rows:
        children.setdefault(pp, []).append(p)
        rss[p] = r
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack += children.get(p, [])
    return total


def process_tree_rss(pid: int) -> int:
    """RSS externo del proceso y descendientes en KiB (``/proc`` en Linux, ``ps`` en otros sistemas)."""
    if not HAS_PROC:
        return _ps_tree_rss(pid)
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
    if not HAS_PROC:
        # macOS: porcentaje libre del sistema según memory_pressure (incluye memoria recuperable)
        pressure = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
        match = re.search(r"free percentage:\s+(\d+)%", pressure)
        if match:
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
            return total * int(match.group(1)) // 100 // 1024
        # sin memory_pressure: páginas libres, inactivas y especulativas de vm_stat (estimación conservadora)
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+) bytes", out).group(1))
        pages = sum(int(m) for m in re.findall(r"Pages (?:free|inactive|speculative):\s+(\d+)", out))
        return pages * page // 1024
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
    require(family in BUILDERS and split in EXPECTED_ROWS, "Familia o split desconocidos")
    parts = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=74)
    base = getattr(parts, split)
    require(base.height == EXPECTED_ROWS[split], "Conteo base incorrecto")
    artifact = build_family_features(
        family, base, split, cutoff_week=74, output_dir=output
    )
    require(artifact.case_count == base.height, "El builder perdió casos")


def rebuild(audit_path: Path, max_rss_mib: int, timeout_seconds: int, *, everything: bool = False) -> list[dict]:
    require(
        os.environ.get("POLARS_MAX_THREADS") == "1" and pl.thread_pool_size() == 1,
        "POLARS_MAX_THREADS=1 obligatorio",
    )
    require(max_rss_mib > 0 and timeout_seconds > 0, "Presupuestos deben ser positivos")
    require(
        Path("/usr/bin/time").is_file(),
        "Falta GNU time; no ejecutar sin medición externa",
    )
    jobs = all_jobs() if everything else select_jobs(json.loads(audit_path.read_text()))
    results = ROOT / "results/preparation"
    log_root = io.data_dir() / "artifacts/preparation/E4-v1/logs"
    log_root.mkdir(parents=True, exist_ok=True)
    records_path = results / "E4_builds.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    for family, split in jobs:
        prior = [r for r in records if (r["family"], r["split"]) == (family, split)]
        if prior and prior[-1]["status"] == "blocked" and not everything:
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
                for table in family_tables(family)
                for p in io.shard_paths(file_split, table)
            ]
            inputs += list((ROOT / "src/dm_credit/preparation").rglob("*.py"))
            inputs += [ROOT / "src/dm_credit/io.py", ROOT / "uv.lock"]
            inputs += list(results.glob("G2_applications_people*.csv"))
            record["input_sha256"] = {str(p): sha256_file(p) for p in inputs}
            metrics = log_root / f"{family}__{split}.time.json"
            log = log_root / f"{family}__{split}.log"
            timer = (
                ["/usr/bin/time", "-f", '{"seconds":%e,"peak_rss_kib":%M}', "-o", str(metrics)]
                if HAS_PROC
                else ["/usr/bin/time", "-l"]  # macOS: el resumen va al final del log
            )
            command = [
                *timer,
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
            if not HAS_PROC:
                metrics.write_text(json.dumps(parse_bsd_time(log.read_text()), separators=(",", ":")))
                record["platform"] = "macOS: /usr/bin/time -l, ps y vm_stat"
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


def any_blocked(records: list[dict]) -> bool:
    """Solo cuenta el último registro de cada pieza: un bloqueo viejo ya reconstruido no es un fallo."""
    latest = {(r["family"], r["split"]): r for r in records}
    return any(r["status"] == "blocked" for r in latest.values())


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
    parser.add_argument("--all", action="store_true", help="reconstruir las 36 piezas en el orden del backlog")
    args = parser.parse_args()
    require(
        os.environ.get("POLARS_MAX_THREADS") == "1", "POLARS_MAX_THREADS=1 obligatorio"
    )
    if args.worker:
        require(args.output is not None, "worker requiere output")
        build_one(*args.worker, args.output)
    else:
        records = rebuild(args.audit, args.max_rss_mib, args.timeout_seconds, everything=args.all)
        if any_blocked(records):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
