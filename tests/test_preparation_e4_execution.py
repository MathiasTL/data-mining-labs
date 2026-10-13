"""E4 execution guards with synthetic metadata; never launch real builders."""

import json
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dm_credit.preparation import e4_rebuild as rebuild
from dm_credit.preparation import e4_shards as shards


def forbidden(*args, **kwargs):
    pytest.fail("Real data access or process execution is forbidden")


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    for name in ("load_base", "shard_paths", "scan_parquet_shards"):
        monkeypatch.setattr(rebuild.io, name, forbidden)
    monkeypatch.setattr(rebuild, "build_family_features", forbidden)
    monkeypatch.setattr(rebuild.subprocess, "Popen", forbidden)
    monkeypatch.setattr(rebuild.os, "killpg", forbidden)
    monkeypatch.setattr(shards.pl, "read_parquet", forbidden)


def audit_for(jobs):
    return {
        "runs": [dict(family=f, split=s, status="invalid") for f, s in jobs],
        "checks": [
            dict(
                scope=f"{f}/{s}",
                status="invalid",
                check="metadata_current_inputs",
                error="shards actuales: distintos",
            )
            for f, s in jobs
        ],
    }


@pytest.mark.parametrize("problem", ["global", "family", "check", "error", "missing"])
def test_select_jobs_rejects_unapproved_failures(problem):
    audit = audit_for([("applprev", "train")])
    if problem == "global":
        audit["checks"].append(
            dict(
                scope="global", status="invalid", check="base", error="base modificada"
            )
        )
    elif problem == "family":
        audit = audit_for([("credit_bureau_a", "train")])
    elif problem == "check":
        audit["checks"][0]["check"] = "schema"
    elif problem == "error":
        audit["checks"][0]["error"] = "plan modificado"
    else:
        audit["checks"] = []
    with pytest.raises(ValueError, match="Presupuesto no aprobado|Revisión manual"):
        rebuild.select_jobs(audit)


def test_select_jobs_skips_valid_outputs_and_orders_test_first():
    expected = [(f, "test") for f in ("applprev", "person")] + [
        (f, s) for f in ("applprev", "person") for s in ("train", "oot")
    ]

    audit = audit_for(list(reversed(expected)))
    audit["runs"] += [
        dict(family="other", split="test", status=status)
        for status in ("reused", "generated_validated")
    ]
    audit["checks"].append(dict(scope="global", status="ok", check="base"))
    assert rebuild.select_jobs(audit) == expected
    assert rebuild.select_jobs({"runs": audit["runs"][-2:], "checks": []}) == []


@pytest.mark.parametrize("gone", [FileNotFoundError, ProcessLookupError])
def test_process_tree_rss_sums_descendants_and_handles_disappearing_processes(
    monkeypatch, gone
):
    files = {
        "/proc/10/status": "Name:\tworker\nVmRSS:\t100 kB\n",
        "/proc/10/task/10/children": "11 12 13",
        "/proc/11/status": "VmRSS:\t20 kB\n",
        "/proc/11/task/11/children": "14",
        "/proc/12/status": "Name:\tno-rss\n",
        "/proc/12/task/12/children": "",
        "/proc/14/status": "VmRSS:\t3 kB\n",
        "/proc/14/task/14/children": "",
    }

    def read(path, *args, **kwargs):
        if str(path) == "/proc/13/status":
            raise gone()
        return files[str(path)]

    monkeypatch.setattr(Path, "read_text", read)
    assert rebuild.process_tree_rss(10) == 123
    assert rebuild.process_tree_rss(13) == 0


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    monkeypatch.setenv("POLARS_MAX_THREADS", "1")
    monkeypatch.setattr(rebuild.pl, "thread_pool_size", lambda: 1)
    monkeypatch.setattr(rebuild, "ROOT", tmp_path)
    monkeypatch.setattr(rebuild.io, "data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(rebuild.io, "shard_paths", lambda *args: [])
    monkeypatch.setattr(rebuild, "sha256_file", lambda path: "synthetic-digest")
    is_file = Path.is_file
    monkeypatch.setattr(
        Path, "is_file", lambda p: p == Path("/usr/bin/time") or is_file(p)
    )
    results = tmp_path / "results/preparation"
    results.mkdir(parents=True)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(audit_for([("applprev", "train"), ("person", "test")])))
    monkeypatch.setattr(rebuild, "available_kib", lambda: 10 * 1024**2)
    monkeypatch.setattr(rebuild, "process_tree_rss", lambda pid: 0)
    monkeypatch.setattr(rebuild.time, "monotonic", lambda: 0)
    monkeypatch.setattr(rebuild.time, "sleep", forbidden)
    return audit, results / "E4_builds.json"


@pytest.mark.parametrize(
    "reason,error",
    [
        ("rss", "Presupuesto RSS excedido"),
        ("timeout", "Tiempo límite excedido"),
        ("reserve", "Reserva de memoria"),
        ("preflight", "MemAvailable insuficiente"),
    ],
)
def test_supervisor_blocks_within_budget_and_never_overlaps_workers(
    supervisor, monkeypatch, reason, error
):
    audit, records_path = supervisor
    events = []
    process = Mock(pid=12345, returncode=-signal.SIGKILL)
    process.poll.return_value = None
    process.wait.side_effect = lambda: events.append("wait")

    def launch(command, **kwargs):
        assert not events or events[-1] == "wait"
        assert kwargs["start_new_session"] is True
        events.append(command[command.index("--worker") + 2])
        return process

    popen = Mock(side_effect=launch)
    monkeypatch.setattr(rebuild.subprocess, "Popen", popen)
    kill = Mock(side_effect=lambda *args: events.append("kill"))
    monkeypatch.setattr(rebuild.os, "killpg", kill)
    if reason == "rss":
        monkeypatch.setattr(rebuild, "process_tree_rss", lambda pid: 5120 * 1024 + 1)
    elif reason == "timeout":
        clock = iter([0, 120, 120] * 2)
        monkeypatch.setattr(rebuild.time, "monotonic", lambda: next(clock))
    else:
        memory = iter([10 * 1024**2, 1024**2 - 1] * 2)
        monkeypatch.setattr(
            rebuild,
            "available_kib",
            lambda: 0 if reason == "preflight" else next(memory),
        )
    records = rebuild.rebuild(audit, 5120, 120)
    assert [(r["family"], r["split"]) for r in records] == [
        ("person", "test"),
        ("applprev", "train"),
    ]
    assert all(r["status"] == "blocked" and error in r["error"] for r in records)
    assert json.loads(records_path.read_text()) == records
    if reason == "preflight":
        popen.assert_not_called()
        kill.assert_not_called()
    else:
        assert events == ["test", "kill", "wait", "train", "kill", "wait"]
        assert kill.call_count == process.wait.call_count == 2
        kill.assert_called_with(12345, signal.SIGKILL)
    popen.reset_mock()
    assert rebuild.rebuild(audit, 5120, 120) == records
    popen.assert_not_called()  # Persisted blocks must not retry the aborted train.


@pytest.mark.parametrize(
    "env,pool,error",
    [
        (None, 1, "POLARS_MAX_THREADS=1"),
        ("2", 1, "POLARS_MAX_THREADS=1"),
        ("1", 2, "pool Polars"),
    ],
)
def test_shards_requires_single_thread_before_any_output(
    tmp_path, monkeypatch, env, pool, error
):
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    if env is not None:
        monkeypatch.setenv("POLARS_MAX_THREADS", env)
    monkeypatch.setattr(shards.pl, "thread_pool_size", lambda: pool)
    with pytest.raises(ValueError, match=error):
        shards.run_audit(root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_shards_rejects_a2_plan_before_fingerprints_and_publishes_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("POLARS_MAX_THREADS", "1")
    monkeypatch.setattr(shards.pl, "thread_pool_size", lambda: 1)
    output = tmp_path / "results/preparation"
    output.mkdir(parents=True)
    cache = tmp_path / "checkpoints"
    cache.mkdir()
    (output / "E4_audit.json").write_text(
        json.dumps(
            {
                "cutoff_week": 74,
                "current_base_sha256": {
                    s: f"synthetic-{s}" for s in shards.contracts.SPLITS
                },
            }
        )
    )
    for name in ("load_selection", "load_g2_decisions", "resolve_columns"):
        monkeypatch.setattr(shards.bureau, name, lambda *args: [])
    plan = SimpleNamespace(tables=[SimpleNamespace(table="credit_bureau_a_2")])
    monkeypatch.setattr(shards.bureau, "plan_from_decisions", lambda *args: plan)
    monkeypatch.setattr(shards, "shard_fingerprint", forbidden)
    with pytest.raises(ValueError, match="Auditoría inválida"):
        shards.run_audit(root=tmp_path, checkpoint_dir=cache)
    report = json.loads((output / "E4_shards.json").read_text())
    assert report["ok"] is False
    assert report["checkpoints"] == []
    assert len(report["checks"]) == 1
    assert "A2 no permitido" in report["checks"][0]["error"]
    assert (output / "E4_shards.csv").read_text().splitlines() == [
        "checkpoint,status,error"
    ]
