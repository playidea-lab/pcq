"""Framework-neutral contract script examples."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from pcq.agent import validate_json_contract
from pcq.cli import main as cli_main


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_json_stdout(captured) -> dict:
    payload = json.loads(captured.out)
    assert isinstance(payload, dict)
    return payload


def _assert_contract(name: str, payload: dict) -> None:
    assert validate_json_contract(name, payload) == []


def test_numpy_contract_script_runs_without_framework_adapter(tmp_path, capfd):
    project = tmp_path / "numpy_contract"
    project.mkdir()
    shutil.copy(REPO_ROOT / "examples" / "contract_numpy.py", project / "train.py")
    (project / "cq.yaml").write_text(
        "name: numpy-contract-test\n"
        f"cmd: {sys.executable} train.py\n"
        "configs:\n"
        "  output_dir: runs/numpy\n"
        "  seed: 42\n"
        "  epochs: 2\n"
        "  lr: 0.2\n"
        "  monitor: eval_acc\n"
        "  mode: max\n"
        "metrics:\n"
        "  epoch:\n"
        "    mode: max\n"
        "  train_loss:\n"
        "    mode: min\n"
        "  eval_acc:\n"
        "    mode: max\n"
        "artifacts:\n"
        "  - runs/numpy/\n"
        "inputs:\n"
        "  synthetic:\n"
        "    opaque: true\n"
        "    reason: generated in test\n",
        encoding="utf-8",
    )

    rc = cli_main(["run", "--path", str(project), "--json"])
    run_payload = _read_json_stdout(capfd.readouterr())
    assert rc == 0
    _assert_contract("pcq.run.envelope", run_payload)
    assert run_payload["status"] == "completed"

    out = project / "runs" / "numpy"
    assert (out / "model.npz").exists()
    assert (out / "framework_result.json").exists()

    rc = cli_main(["validate-run", str(out), "--strictness", "2", "--json"])
    validation = _read_json_stdout(capfd.readouterr())
    assert rc == 0
    _assert_contract("pcq.validation_report", validation)
    assert validation["status"] == "pass"

    rc = cli_main(["describe-run", str(out), "--json"])
    desc = _read_json_stdout(capfd.readouterr())
    assert rc == 0
    _assert_contract("pcq.describe_run.record", desc)
    assert desc["target_metric"] == "eval_acc"
    assert desc["decision_facts"]["run_completed"] is True
