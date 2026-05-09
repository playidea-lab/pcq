"""Public JSON contract regression tests for agent-facing pcq outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pcq
from pcq.agent import (
    get_json_contracts,
    validate_json_contract,
)
from pcq.agent.compare import compare_runs
from pcq.agent.describe import describe_run
from pcq.agent.validate_run import validate_run
from pcq.cli import main as cli_main


def _setup_cfg(out_dir: Path, **extra) -> Path:
    cfg = {
        "output_dir": str(out_dir),
        "seed": 42,
        "monitor": "eval_acc",
        "mode": "max",
        "_metrics_declared": [
            {"name": "epoch", "mode": "max"},
            {"name": "eval_acc", "mode": "max"},
        ],
    }
    cfg.update(extra)
    p = out_dir.parent / f"{out_dir.name}_cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _make_run(out_dir: Path, monkeypatch, value: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))
    history = [{"epoch": 0, "eval_acc": value}]
    pcq.save_metrics(history)
    pcq.save_run_summary(history=history, status="completed")
    pcq.save_manifest()
    pcq.finalize_run(history=history)


def _assert_contract(name: str, payload: dict) -> None:
    errors = validate_json_contract(name, payload)
    assert errors == []


def test_json_contract_registry_is_json_serializable():
    contracts = get_json_contracts()
    assert "pcq.run.envelope" in contracts
    assert "pcq.run.event" in contracts
    assert "pcq.describe_run.record" in contracts
    assert "pcq.compare_runs.diff" in contracts
    assert "pcq.validation_report" in contracts
    assert "pcq.agent_install.result" in contracts
    assert "pcq.agent_status.result" in contracts
    json.dumps(contracts)


def test_validate_json_contract_reports_missing_required_fields():
    errors = validate_json_contract("pcq.validation_report", {"schema_version": 1})
    assert "missing required field: status" in errors
    assert "missing required field: checks" in errors


def test_pcq_run_json_envelope_contract_success_and_error(
    tmp_path: Path, capfd
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "cq.yaml").write_text(
        "name: run-contract\n"
        f"cmd: {sys.executable} train.py\n"
        "configs:\n"
        "  seed: 42\n",
        encoding="utf-8",
    )
    (project / "train.py").write_text("print('contract-ok')\n", encoding="utf-8")

    rc = cli_main(["run", "--path", str(project), "--json"])
    captured = capfd.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    _assert_contract("pcq.run.envelope", payload)
    assert payload["status"] == "completed"
    assert payload["stdout_tail"] == "contract-ok\n"

    missing = tmp_path / "missing_cq_yaml"
    missing.mkdir()
    rc = cli_main(["run", "--path", str(missing), "--json"])
    captured = capfd.readouterr()
    assert rc == 1
    payload = json.loads(captured.out)
    _assert_contract("pcq.run.envelope", payload)
    assert payload["status"] == "error"
    assert "cq.yaml" in payload["error"]


def test_pcq_run_jsonl_event_contract(tmp_path: Path, capfd):
    project = tmp_path / "project"
    project.mkdir()
    (project / "cq.yaml").write_text(
        "name: run-event-contract\n"
        f"cmd: {sys.executable} train.py\n",
        encoding="utf-8",
    )
    (project / "train.py").write_text(
        "import pcq\n"
        "pcq.log(epoch=1, eval_acc=0.9)\n",
        encoding="utf-8",
    )

    rc = cli_main(["run", "--path", str(project), "--jsonl"])
    captured = capfd.readouterr()
    assert rc == 0
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert events
    for event in events:
        _assert_contract("pcq.run.event", event)
    assert events[0]["event"] == "run.started"
    assert any(event["event"] == "metric" for event in events)
    assert events[-1]["event"] == "run.completed"


def test_describe_compare_and_validate_run_json_contracts(tmp_path, monkeypatch):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_run(a, monkeypatch, 0.6)
    _make_run(b, monkeypatch, 0.8)

    desc = describe_run(b).to_dict()
    _assert_contract("pcq.describe_run.record", desc)
    assert desc["decision_facts"]["run_completed"] is True

    diff = compare_runs(a, b).to_dict()
    _assert_contract("pcq.compare_runs.diff", diff)
    assert diff["decision_facts"]["comparable"] is True

    report = validate_run(b, strictness=2).to_dict()
    _assert_contract("pcq.validation_report", report)
    assert report["strictness"] == 2
