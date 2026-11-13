"""Auditoría de inventario de solo lectura sobre fuentes; escribe en esta carpeta."""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(name, records, fields):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    evidence = []
    paths = sorted((ROOT / "results").rglob("*"))
    paths += sorted((ROOT / "development").glob("*.md"))
    paths += sorted((ROOT / "configs").glob("*"))
    for path in paths:
        if path.is_file():
            evidence.append(
                dict(
                    path=str(path.relative_to(ROOT)),
                    bytes=path.stat().st_size,
                    sha256=digest(path),
                )
            )
    save("inventario-evidencias.csv", evidence, ["path", "bytes", "sha256"])

    prep = ROOT / "results/preparation"
    catalog = rows(prep / "C8_feature_catalog.csv")
    save("inventario-caracteristicas.csv", catalog, list(catalog[0]))
    checks = rows(prep / "F3_final_checks.csv")
    artifacts = []
    for report in ["E5_master_tables.csv", "E6_views.csv"]:
        for row in rows(prep / report):
            original = row["data_path"]
            # Las rutas archivadas proceden de otra máquina: conservar el sufijo del proyecto.
            marker = "/data-mining-labs/"
            relative = original.split(marker, 1)[1] if marker in original else original
            local = ROOT / relative
            actual = digest(local) if local.is_file() else ""
            artifacts.append(
                dict(
                    report=report,
                    split=row["split"],
                    view=row.get("view", "master"),
                    archived_path=original,
                    local_path=relative,
                    exists=local.is_file(),
                    expected_sha256=row["data_sha256"],
                    actual_sha256=actual,
                    status=("coincide" if actual == row["data_sha256"] else "diferente")
                    if actual
                    else "ausente",
                )
            )
    save("disponibilidad-matrices.csv", artifacts, list(artifacts[0]))

    manifest_checks = []
    for name in ["E7_manifest.json", "F3_final_run_manifest.json"]:
        manifest = json.loads((prep / name).read_text())
        for relative, expected in manifest.get("inputs", {}).items():
            path = ROOT / relative
            actual = digest(path) if path.is_file() else ""
            manifest_checks.append(
                dict(
                    manifest=name,
                    path=relative,
                    expected_sha256=expected,
                    actual_sha256=actual,
                    status=("coincide" if actual == expected else "diferente")
                    if actual
                    else "ausente",
                )
            )
    save(
        "verificacion-entradas-manifiestos.csv",
        manifest_checks,
        ["manifest", "path", "expected_sha256", "actual_sha256", "status"],
    )
    names = [row["name"] for row in catalog]
    required = [
        "source_table",
        "source_columns",
        "aggregation",
        "window",
        "temporal_rule",
        "description",
    ]
    summary = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="Inventario, integridad documental y SHA-256; sin entrenar ni regenerar matrices",
        evidence_files=len(evidence),
        features=len(catalog),
        roles=dict(Counter(row["role"] for row in catalog)),
        sources=dict(Counter(row["source_table"] for row in catalog)),
        duplicate_feature_names=[k for k, n in Counter(names).items() if n > 1],
        empty_catalog_fields={
            key: sum(not row.get(key, "").strip() for row in catalog)
            for key in required
        },
        archived_checks=dict(Counter(row["status"] for row in checks)),
        matrix_availability=dict(Counter(row["status"] for row in artifacts)),
        manifest_inputs=dict(Counter(row["status"] for row in manifest_checks)),
    )
    (OUT / "resumen-inventario.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
