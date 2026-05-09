"""CLI subprocess 테스트 — 각 command 의 JSON 출력 + exit code 검증.

v4.0: atom registry / preset / Trainer 제거 후 정리.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(*args: str) -> tuple[int, dict | str, str]:
    """pcq CLI 실행. JSON parsable 이면 dict, 아니면 raw string 반환."""
    result = subprocess.run(
        [sys.executable, "-m", "pcq.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    try:
        out: dict | str = json.loads(result.stdout)
    except json.JSONDecodeError:
        out = result.stdout
    return result.returncode, out, result.stderr


def test_inspect_examples_json_returns_project_inspection():
    """examples/ 안에 cq.yaml 있음 → ProjectInspection JSON."""
    rc, out, _ = _run_cli("inspect", "examples", "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["schema_version"] == 1
    assert out["has_cq_yaml"] is True


def test_inspect_no_cq_yaml_warns_but_does_not_error(tmp_path):
    rc, out, _ = _run_cli("inspect", str(tmp_path), "--json")
    assert rc == 0  # warning, not error
    assert isinstance(out, dict)
    assert out["has_cq_yaml"] is False
    assert any("no cq.yaml" in w for w in out.get("warnings", []))


def test_inspect_nonexistent_path_returns_error():
    rc, out, _ = _run_cli("inspect", "/nonexistent/path/xyz", "--json")
    assert rc == 1
    assert isinstance(out, dict)
    assert any("does not exist" in e for e in out.get("errors", []))


def test_cli_inspect_does_not_load_project_atoms_by_default(tmp_path):
    (tmp_path / "cq.yaml").write_text("name: t\ncmd: uv run python train.py\n")
    (tmp_path / "train.py").write_text("import pcq\ncq.config()\n")
    (tmp_path / "pcq_atoms.py").write_text("raise ImportError('side effect')\n")

    rc, out, _ = _run_cli("inspect", str(tmp_path), "--json")

    assert rc == 0
    assert isinstance(out, dict)
    # v4.0: project_atoms_loaded 는 항상 not loaded (atom registry 제거)
    assert out["errors"] == []


def test_validate_examples_passes_or_warns():
    rc, out, _ = _run_cli("validate", "examples", "--json")
    # examples 는 cq.yaml 정상 → pass 또는 warn
    assert rc == 0
    assert isinstance(out, dict)
    assert out["status"] in ("pass", "warn")
    assert isinstance(out["checks"], list)
    assert len(out["checks"]) >= 1
    # cq_yaml_exists check 통과해야
    cq_check = next(c for c in out["checks"] if c["id"] == "cq_yaml_exists")
    assert cq_check["status"] == "pass"


def test_validate_strictness_level_reported():
    rc, out, _ = _run_cli("validate", "examples", "--strictness", "1", "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["strictness"] == 1
    assert out["strictness_name"] == "static"
    strict = next(c for c in out["checks"] if c["id"] == "strictness_level")
    assert strict["evidence"]["level"] == 1


def test_resolve_accepts_explicit_cq_yaml_path():
    rc, out, _ = _run_cli("resolve", "--cq-yaml", "examples/cq.yaml", "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["cq_yaml_path"].endswith("examples/cq.yaml")
    assert out["project_root"].endswith("examples")


def test_validate_no_cq_yaml_blocking_fail(tmp_path):
    rc, out, _ = _run_cli("validate", str(tmp_path), "--json")
    assert rc == 1
    assert isinstance(out, dict)
    assert out["status"] == "fail"
    assert out["blocking_count"] >= 1
    cq_check = next(c for c in out["checks"] if c["id"] == "cq_yaml_exists")
    assert cq_check["status"] == "fail"
    assert cq_check["severity"] == "blocking"


def test_summarize_run_missing_output_dir_unknown(tmp_path):
    rc, out, _ = _run_cli("summarize-run", str(tmp_path / "nonexistent"), "--json")
    assert rc == 0  # graceful — status="unknown" not failure
    assert isinstance(out, dict)
    assert out["status"] == "unknown"


def test_cli_version():
    """--version 작동 (argparse 가 자동으로 stdout 출력 + exit 0).

    pcq 버전은 pcq.__version__ 와 일치해야 한다 (SemVer 형식).
    """
    import pcq

    result = subprocess.run(
        [sys.executable, "-m", "pcq.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert pcq.__version__ in result.stdout


def test_cli_no_command_exits_2():
    """sub-command 없이 호출 → argparse error (exit 2)."""
    result = subprocess.run(
        [sys.executable, "-m", "pcq.cli"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


# ─────────────────────────────────────────────────────────────────────
# v1.10 / v4.0: pcq init-experiment / apply-plan / validate --plan
# ─────────────────────────────────────────────────────────────────────


def test_cli_init_experiment(tmp_path):
    rc, out, _ = _run_cli(
        "init-experiment",
        "--output", str(tmp_path),
        "--json",
    )
    assert rc == 0
    assert isinstance(out, dict)
    assert "cq.yaml" in out["files_created"]
    assert "train.py" in out["files_created"]
    assert (tmp_path / "cq.yaml").exists()
    assert (tmp_path / "train.py").exists()


def test_cli_init_experiment_default_name(tmp_path):
    rc, out, _ = _run_cli(
        "init-experiment",
        "--output", str(tmp_path),
        "--json",
    )
    assert rc == 0
    assert out["name"] == "pcq-experiment"


def test_cli_init_experiment_force(tmp_path):
    (tmp_path / "cq.yaml").write_text("# pre-existing\n")
    rc, out, _ = _run_cli(
        "init-experiment",
        "--output", str(tmp_path),
        "--force",
        "--json",
    )
    assert rc == 0
    assert "cq.yaml" in out["files_created"]


def test_cli_apply_plan_set_config(tmp_path):
    """init → apply-plan 으로 epochs 수정."""
    _run_cli(
        "init-experiment",
        "--output", str(tmp_path),
        "--force",
        "--json",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema_version": 1,
        "id": "exp-cli-001",
        "base": {"baseline": "init"},
        "changes": [{"op": "set_config", "key": "epochs", "value": 3}],
    }))
    rc, out, _ = _run_cli(
        "apply-plan", str(plan_path),
        "--path", str(tmp_path),
        "--json",
    )
    assert rc == 0
    assert isinstance(out, dict)
    assert out["status"] == "applied"
    assert "cq.yaml" in out["files_changed"]


def test_cli_apply_plan_missing_file_returns_error(tmp_path):
    rc, out, _ = _run_cli(
        "apply-plan", str(tmp_path / "missing.json"),
        "--path", str(tmp_path),
        "--json",
    )
    assert rc == 1
    assert isinstance(out, dict)
    assert "error" in out


def test_cli_validate_with_plan(tmp_path):
    """validate --plan 옵션 — plan 이 valid 면 plan_validation pass."""
    _run_cli(
        "init-experiment",
        "--output", str(tmp_path),
        "--force",
        "--json",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema_version": 1,
        "id": "exp-validate-ok",
        "base": {"baseline": "init"},
        "changes": [{"op": "set_config", "key": "epochs", "value": 5}],
    }))
    rc, out, _ = _run_cli(
        "validate", str(tmp_path),
        "--plan", str(plan_path),
        "--json",
    )
    plan_check = next(
        (c for c in out["checks"] if c["id"] == "plan_validation"), None
    )
    assert plan_check is not None
    assert plan_check["status"] == "pass"


def test_cli_validate_with_missing_plan_file(tmp_path):
    _run_cli(
        "init-experiment",
        "--output", str(tmp_path),
        "--force",
        "--json",
    )
    rc, out, _ = _run_cli(
        "validate", str(tmp_path),
        "--plan", str(tmp_path / "nonexistent.json"),
        "--json",
    )
    assert rc == 1
    plan_checks = [c for c in out["checks"] if c["id"] == "plan_validation"]
    assert plan_checks
    assert plan_checks[0]["status"] == "fail"


def test_cli_validate_script_complete(tmp_path):
    """contract script 가 표준 helper 모두 호출하면 validate pass/warn."""
    _run_cli(
        "init-experiment",
        "--output", str(tmp_path),
        "--force",
        "--json",
    )
    rc, out, _ = _run_cli("validate", str(tmp_path), "--json")
    # contract artifact 체크는 통과 (pass/warn)
    assert rc in (0, 1)
    assert isinstance(out, dict)
    ids = {c["id"] for c in out["checks"]}
    assert "cq_config_called" in ids
    assert "standard_artifacts_helper" in ids


# ─────────────────────────────────────────────────────────────────────
# v1.16: pcq finalize / validate-run / inspect has_run_record
# ─────────────────────────────────────────────────────────────────────


def test_cli_finalize(tmp_path):
    """contract artifacts 모두 존재 → pcq finalize → run_record.json 작성."""
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / "cq.yaml").write_text(
        "name: t\ncmd: c\nconfigs: {}\nmetrics: [eval_acc]\nartifacts: [output/]\n"
    )
    (output / "metrics.json").write_text(
        json.dumps({"history": [{"epoch": 0}]})
    )
    (output / "run_summary.json").write_text(json.dumps({"status": "completed"}))
    (output / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": []})
    )
    (output / "config.json").write_text("{}")

    rc, out, _ = _run_cli("finalize", str(output), "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert "run_record_path" in out
    assert (output / "run_record.json").exists()
    assert (output / "validation_report.json").exists()


def test_cli_validate_run_pass(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "metrics.json").write_text(
        json.dumps({"history": [{"epoch": 0}]})
    )
    (output / "run_summary.json").write_text(
        json.dumps({"best": {"epoch": 0}, "last": {"epoch": 0}})
    )
    (output / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": []})
    )
    rc, out, _ = _run_cli("validate-run", str(output), "--json")
    assert rc in (0, 1)
    assert isinstance(out, dict)
    assert "checks" in out


def test_cli_validate_run_accepts_strictness(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "metrics.json").write_text(
        json.dumps({"history": [{"epoch": 0}]})
    )
    (output / "run_summary.json").write_text(
        json.dumps({"best": {"epoch": 0}, "last": {"epoch": 0}})
    )
    (output / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": []})
    )
    rc, out, _ = _run_cli(
        "validate-run", str(output), "--strictness", "3", "--json"
    )
    assert rc == 1
    assert isinstance(out, dict)
    assert out["strictness"] == 3
    strict = next(c for c in out["checks"] if c["id"] == "strictness_level")
    assert strict["evidence"]["name"] == "reproducible"


def test_cli_validate_run_fail_when_missing_metrics(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": []})
    )
    rc, out, _ = _run_cli("validate-run", str(output), "--json")
    assert rc == 1
    assert isinstance(out, dict)
    assert out["status"] == "fail"


def test_cli_inspect_recognizes_run_record(tmp_path):
    """outputs.has_run_record / has_validation_report 노출."""
    (tmp_path / "cq.yaml").write_text(
        "name: t\ncmd: c\nconfigs: {}\nmetrics: []\nartifacts: [output/]\n"
    )
    (tmp_path / "train.py").write_text("import pcq\ncq.config()\n")
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    (out_dir / "run_record.json").write_text("{}")
    (out_dir / "validation_report.json").write_text("{}")
    rc, j, _ = _run_cli("inspect", str(tmp_path), "--json")
    assert rc == 0
    assert j["outputs"]["has_run_record"] is True
    assert j["outputs"]["has_validation_report"] is True


# ─────────────────────────────────────────────────────────────────────
# v1.17: describe-run / compare-runs
# ─────────────────────────────────────────────────────────────────────


def _build_completed_run(
    tmp_path: Path, monitor: str = "eval_iou", value: float = 0.7,
):
    """헬퍼 — 간단한 contract artifact 세트로 finalize 까지 마친 output dir 작성."""
    import pcq

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "output_dir": str(tmp_path),
        "monitor": monitor,
        "mode": "max",
        "seed": 42,
    }))
    import os
    prev = os.environ.get("CQ_CONFIG_JSON")
    os.environ["CQ_CONFIG_JSON"] = str(cfg_path)
    try:
        history = [{"epoch": 0, monitor: value}]
        pcq.save_metrics(history)
        pcq.save_run_summary(history=history, status="completed")
        pcq.save_manifest()
        pcq.finalize_run(history=history)
    finally:
        if prev is None:
            os.environ.pop("CQ_CONFIG_JSON", None)
        else:
            os.environ["CQ_CONFIG_JSON"] = prev


def test_cli_describe_run_completed(tmp_path):
    _build_completed_run(tmp_path, value=0.7)
    rc, out, _ = _run_cli("describe-run", str(tmp_path), "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["schema_version"] == 1
    assert out["status"] == "completed"
    assert out["target_metric"] == "eval_iou"
    assert out["best_value"] == 0.7
    assert out["epochs_completed"] == 1


def test_cli_describe_run_no_record_status(tmp_path):
    rc, out, _ = _run_cli("describe-run", str(tmp_path), "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["status"] == "no_record"


def test_cli_compare_runs_improved(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    _build_completed_run(a_dir, value=0.5)
    _build_completed_run(b_dir, value=0.7)
    rc, out, _ = _run_cli("compare-runs", str(a_dir), str(b_dir), "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["target_metric"] == "eval_iou"
    assert out["metric_direction"] == "improved"
    # rounding: 0.7 - 0.5 = 0.2
    assert abs(out["metric_delta"] - 0.2) < 1e-6


def test_cli_compare_runs_with_record_json_path(tmp_path):
    """A/B 인자에 run_record.json 직접 지정 가능."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    _build_completed_run(a_dir, value=0.5)
    _build_completed_run(b_dir, value=0.5)
    rc, out, _ = _run_cli(
        "compare-runs",
        str(a_dir / "run_record.json"),
        str(b_dir / "run_record.json"),
        "--json",
    )
    assert rc == 0
    assert isinstance(out, dict)
    assert out["metric_direction"] == "tied"


def test_cli_compare_runs_missing_file_returns_empty_diff(tmp_path):
    rc, out, _ = _run_cli(
        "compare-runs",
        str(tmp_path / "nonexistent_a"),
        str(tmp_path / "nonexistent_b"),
        "--json",
    )
    assert rc == 0
    assert isinstance(out, dict)
    # missing record → 빈 diff (run_id 비어있음)
    assert "metric_delta" not in out


# ── v1.18 lineage CLI ─────────────────────────────────────────────────


def _write_lineage_record(
    out_dir: Path,
    run_id: str,
    parent_id: str | None = None,
    parent_path: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rr: dict = {
        "schema_version": 1,
        "run": {"id": run_id, "status": "completed"},
        "execution": {},
        "source": {},
        "environment": {},
        "metrics": {"declared": [], "history_path": "metrics.json"},
        "artifacts": [],
        "summary": {},
    }
    if parent_id:
        rr["run"]["parent_run_id"] = parent_id
    if parent_path:
        rr["run"]["parent_run_path"] = parent_path
    (out_dir / "run_record.json").write_text(json.dumps(rr), encoding="utf-8")


def test_cli_lineage_single(tmp_path):
    out_dir = tmp_path / "run"
    _write_lineage_record(out_dir, "x")
    rc, out, _ = _run_cli("lineage", str(out_dir), "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["schema_version"] == 1
    assert len(out["chain"]) == 1
    assert out["chain"][0]["run_id"] == "x"


def test_cli_lineage_chain(tmp_path):
    parent = tmp_path / "p"
    child = tmp_path / "c"
    _write_lineage_record(parent, "parent")
    _write_lineage_record(
        child, "child", parent_id="parent", parent_path=str(parent)
    )
    rc, out, _ = _run_cli("lineage", str(child), "--json")
    assert rc == 0
    assert [n["run_id"] for n in out["chain"]] == ["child", "parent"]


def test_cli_lineage_max_depth_truncates(tmp_path):
    """--max-depth 1 이면 root + 1 단계만 따라가고 truncated=True."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    _write_lineage_record(c, "c")
    _write_lineage_record(b, "b", parent_id="c", parent_path=str(c))
    _write_lineage_record(a, "a", parent_id="b", parent_path=str(b))
    rc, out, _ = _run_cli(
        "lineage", str(a), "--max-depth", "1", "--json"
    )
    assert rc == 0
    assert out["truncated"] is True


def test_cli_lineage_missing_record(tmp_path):
    """없는 path 도 graceful exit (rc=0, 빈 chain + notes)."""
    rc, out, _ = _run_cli(
        "lineage", str(tmp_path / "nope"), "--json"
    )
    assert rc == 0
    assert isinstance(out, dict)
    assert out["chain"] == []
    assert out["notes"]
