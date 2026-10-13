"""Auditor serial de checkpoints bureau; no lee raw ni reconstruye artefactos.

CLI: POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e4_shards
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path

import polars as pl

from dm_credit import io
from dm_credit.preparation import contracts
from dm_credit.preparation.e4 import ( ROOT, Checks, check_equal, require, shard_fingerprint, )
from dm_credit.preparation.families import bureau


def run_audit(*, root: Path = ROOT, checkpoint_dir: Path | None = None) -> dict:
    """Publica CSV/JSON incluso con fallos; lanza ValueError si la auditoría falla."""
    require( os.environ.get("POLARS_MAX_THREADS") == "1", "POLARS_MAX_THREADS=1 obligatorio" )
    check_equal(pl.thread_pool_size(), 1, "pool Polars (iniciar un proceso nuevo)")
    started = time.perf_counter()
    root = Path(root)
    output = root / "results/preparation"
    artifacts = root / "data/artifacts/preparation/C4-C5-v2"
    cache = ( Path(checkpoint_dir) if checkpoint_dir is not None else artifacts / "checkpoints" )
    checks, rows, expected, totals = Checks(), [], {}, {}
    report = { "checkpoint_root": str(cache.resolve()), "cutoff_week": 74, "limitations": [ "No hay recursos históricos verificables del build: duración y ru_maxrss son null.", "Raw rows leídas durante el build no disponibles; el auditor no lee raw ni A2.", "ru_maxrss es el máximo acumulado del proceso auditor, en KiB Linux.", "Base/split se confían a current_base_sha256 de E4; no se releen bases.", "Manifiestos sin sumas n_records: filas/casos se contrastan con coverage E4; " "exclusiones y procedencia con manifiestos actuales, sin releer Parquet finales.", "Fingerprint de fuente: nombre, tamaño y SHA-256 del contenido (io.file_fingerprint), portable entre computadoras.", ], }

    def prepare():
        require(cache.is_dir(), f"No existe directorio de checkpoints: {cache}")
        audit = json.loads((output / "E4_audit.json").read_text())
        check_equal(audit["cutoff_week"], 74, "cutoff E4")
        bases = audit["current_base_sha256"]
        require( set(bases) == set(contracts.SPLITS) and len(set(bases.values())) == 3, "current_base_sha256 debe identificar unívocamente los tres splits", )
        report["current_base_sha256"] = bases
        resolved = bureau.resolve_columns( bureau.load_selection(output / "B5_bureau_columns.csv"), bureau.load_g2_decisions(output / "G2_bureau_columns.csv"), )
        for family in bureau.FAMILIES:
            plan = bureau.plan_from_decisions(family, resolved)
            require( all(t.table != "credit_bureau_a_2" for t in plan.tables), "A2 no permitido", )
            for split in contracts.SPLITS:
                fps = { t.table: shard_fingerprint(split, (t.table,), ordered=True) for t in plan.tables }
                totals[family, split] = { "plan": plan, "fps": fps, "exclusions": Counter(), "tables": {}, }
                for tp in plan.tables:
                    totals[family, split]["tables"][tp.table] = { "rows": 0, "cases": set(), }
                    paths = io.shard_paths( "test" if split == "test" else "train", tp.table )
                    for path, fp in zip(paths, fps[tp.table].split(";"), strict=True):
                        key = dict( rules_version=bureau.RULES_VERSION, plan=plan.fingerprint(), shard=fp, parent_shards=fps[tp.parent] if tp.parent else "", base_sha256=bases[split], cutoff_week="74", )
                        expected[tp.table, path.name, bases[split]] = ( family, split, tp, key, )
        report["e4_coverage"] = audit["coverage"]

    checks.run("setup", "metadata y planes vigentes", prepare)
    seen = Counter()
    for path in sorted(cache.rglob("*.json")) if expected else []:
        begin = time.perf_counter()
        row = dict( checkpoint=str(path), table=path.parent.name, shard=None, split=None, status="invalid", error=None, base_sha256=None, rules_version=None, plan=None, current_plan=None, source_fingerprint=None, current_source_fingerprint=None, parent_shards=None, current_parent_shards=None, cutoff_week=None, partial_sha256=None, keys_sha256=None, rows=None, cases=None, exclusions=None, build_seconds=None, build_ru_maxrss_kib=None, raw_rows_read=None, )
        rows.append(row)

        def inspect():
            meta = json.loads(path.read_text())
            key = meta["key"]
            row.update( { k: key.get(k) for k in ( "base_sha256", "rules_version", "plan", "parent_shards", "cutoff_week", ) } )
            row.update( source_fingerprint=key["shard"], shard=key["shard"].split(":")[0], partial_sha256=meta.get("partial_sha256"), keys_sha256=meta.get("keys_sha256"), exclusions=meta.get("exclusions"), )
            row["split"] = next( ( s for s, h in report["current_base_sha256"].items() if h == key["base_sha256"] ), None, )
            identity = (row["table"], row["shard"], key["base_sha256"])
            item = expected.get(identity)
            if item is None:
                row.update( status="obsolete", error="No pertenece al inventario/base vigente; no utilizado", )
                return
            family, split, tp, current = item
            row.update( current_plan=current["plan"], current_source_fingerprint=current["shard"], current_parent_shards=current["parent_shards"], )
            if key != current:
                row.update( status="obsolete", error="Clave distinta de la vigente; no utilizado", )
                return
            seen[identity] += 1
            check_equal(seen[identity], 1, "checkpoint vigente duplicado")
            for kind in ("partial", "keys"):
                digest = meta[f"{kind}_sha256"]
                if kind == "keys":
                    check_equal( digest is not None, tp.has_children, "presencia de parent keys" )
                if digest is not None:
                    payload = path.with_suffix(f".{kind}.parquet")
                    require( payload.resolve().is_relative_to(cache.resolve()), "payload fuera de checkpoints", )
                    check_equal(contracts._sha256(payload), digest, f"hash {kind}")
            partial = pl.read_parquet( path.with_suffix(".partial.parquet"), columns=["case_id", "n_rows"] )
            contracts.validate_base_keys(partial)
            counts = partial["n_rows"]
            require( counts.dtype.is_integer() and not counts.null_count() and (counts > 0).all(), "n_rows inválido", )
            exclusions = meta["exclusions"]
            require( isinstance(exclusions, dict) and all(type(v) is int and v >= 0 for v in exclusions.values()), "exclusiones inválidas", )
            row.update( rows=int(counts.sum()), cases=partial["case_id"].n_unique(), status="ok" )
            total = totals[family, split]
            total["tables"][tp.table]["rows"] += row["rows"]
            total["tables"][tp.table]["cases"].update(partial["case_id"].to_list())
            total["exclusions"].update(exclusions)

        checks.run(str(path), "checkpoint", inspect)
        if checks.rows[-1]["status"] != "ok":
            row.update(status="invalid", error=checks.rows[-1]["error"])
        row.update( audit_seconds=time.perf_counter() - begin, audit_process_ru_maxrss_kib=resource.getrusage( resource.RUSAGE_SELF ).ru_maxrss, )

    def reconcile(family, split, total):
        plan = total["plan"]
        for identity, (f, s, _, _) in expected.items():
            if (f, s) == (family, split):
                check_equal( seen[identity], 1, f"checkpoint vigente requerido {identity}" )
        manifest = artifacts / family / split / f"{family}__{split}.manifest.json"
        replacement = artifacts.parent / "E4-v1" / family / split / manifest.name
        if replacement.exists():  # igual que el auditor E4: el reemplazo vigente tiene prioridad
            manifest = replacement
        m = json.loads(manifest.read_text())
        for name, value in dict( family=family, split=split, rules_version=bureau.RULES_VERSION ).items():
            check_equal(m[name], value, name)
        for name, value in dict( base_sha256=report["current_base_sha256"][split], cutoff_week="74", plan_sha256=plan.fingerprint(), g2_mode="approved", shards=";".join(total["fps"].values()), ).items():
            check_equal(m["inputs"][name], value, name)
        check_equal(dict(total["exclusions"]), m["exclusions"], "suma exclusiones")
        for table, values in total["tables"].items():
            coverage = [ r for r in report["e4_coverage"] if (r["family"], r["split"], r["table"]) == (family, split, table) ]
            require(bool(coverage), f"Falta cobertura E4 {table}/{split}")
            for c in coverage:
                check_equal(c["status"], "ok", "cobertura E4 válida")
                check_equal(values["rows"], c["rows"], "suma n_rows")
                check_equal( len(values["cases"]), c["cases"], "casos únicos entre shards" )
                check_equal( m["case_count"], c["base_cases"], "casos base manifiesto/E4" )
            check_equal(m["row_count"], m["case_count"], "filas finales por caso")

    for (family, split), total in totals.items():
        checks.run( f"{family}/{split}", "conciliación", lambda: reconcile(family, split, total) )
    report.pop("e4_coverage", None)
    report.update( ok=all(c["status"] == "ok" for c in checks.rows), checks=checks.rows, checkpoints=rows, audit_seconds=time.perf_counter() - started, audit_process_ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "E4_shards.csv").open("w", newline="") as stream:
        writer = csv.DictWriter( stream, fieldnames=list(rows[0]) if rows else ["checkpoint", "status", "error"], )
        writer.writeheader()
        writer.writerows( { k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in row.items() } for row in rows )
    (output / "E4_shards.json").write_text( json.dumps(report, indent=2, ensure_ascii=False) )
    require(report["ok"], f"Auditoría inválida; consultar {output / 'E4_shards.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint-dir", type=Path)
    run_audit(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
