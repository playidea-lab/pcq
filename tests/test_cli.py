"""CLI subprocess 테스트 — 각 command 의 JSON 출력 + exit code 검증."""
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
    assert out["cq_yaml"]["name"] == "cq-python-smoke"
    assert "epoch" in out["cq_yaml"]["declared_metrics"]


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
    assert out["project_atoms_loaded"]["loaded"] is False
    assert out["errors"] == []


def test_cli_inspect_load_project_atoms_opt_in_reports_errors(tmp_path):
    (tmp_path / "cq.yaml").write_text("name: t\ncmd: uv run python train.py\n")
    (tmp_path / "train.py").write_text("import pcq\ncq.config()\n")
    (tmp_path / "pcq_atoms.py").write_text("raise ImportError('side effect')\n")

    rc, out, _ = _run_cli(
        "inspect", str(tmp_path), "--load-project-atoms", "--json"
    )

    assert rc == 1
    assert isinstance(out, dict)
    assert out["project_atoms_loaded"]["errors"]
    assert any("side effect" in e for e in out["errors"])


def test_recipe_meta_json_known_recipe():
    rc, out, _ = _run_cli("recipe-meta", "vision/fake_smoke", "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["schema_version"] == 1
    assert out["name"] == "vision/fake_smoke"
    assert out["task"] == "classification"


def test_recipe_meta_unknown_recipe_fails_exit_1():
    rc, _, _ = _run_cli("recipe-meta", "does/not/exist", "--json")
    assert rc == 1


def test_validate_examples_passes_or_warns():
    rc, out, _ = _run_cli("validate", "examples", "--json")
    # examples 는 cq.yaml 정상 → pass 또는 warn (recipe metric mismatch 등)
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


def test_summarize_run_after_fit_returns_completed(tmp_path):
    """실제 1-epoch fit() 후 summarize-run JSON."""
    import pcq

    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    pcq.Trainer(task="classification", dataset="fake", model="mlp", cfg=cfg).fit()

    rc, out, _ = _run_cli("summarize-run", str(tmp_path), "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["schema_version"] == 1
    assert out["status"] == "completed"
    assert out["last"]["epoch"] == 0
    # eval_loss monitor → best 존재
    assert out["best"] is not None


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


def test_dry_run_no_preset_in_examples_returns_graceful():
    """examples/train.py 는 atom-only (preset literal 없음) → graceful 처리 (v1.13).

    v1.13 부터 trainer 검출됐으나 preset literal 없을 때 rc=0 + detail 메시지.
    """
    rc, out, _ = _run_cli("dry-run", "examples", "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert "detail" in out
    assert out["kind"] == "trainer"


# ─────────────────────────────────────────────────────────────────────
# v1.8: pcq atoms list / show / validate-ref
# ─────────────────────────────────────────────────────────────────────


def test_cli_atoms_list_json_includes_all_kinds():
    rc, out, _ = _run_cli("atoms", "list", "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["schema_version"] == 1
    assert "atoms" in out
    expected_kinds = {"model", "dataset", "loss", "optim", "sched", "metric"}
    assert expected_kinds.issubset(set(out["atoms"].keys()))


def test_cli_atoms_list_filter_kind():
    rc, out, _ = _run_cli("atoms", "list", "--kind", "loss", "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert list(out["atoms"].keys()) == ["loss"]
    # cross_entropy 가 explicit 으로 표시
    ce = next(a for a in out["atoms"]["loss"] if a["name"] == "cross_entropy")
    assert ce["metadata_status"] == "explicit"
    assert "classification" in ce["tasks"]


def test_cli_atoms_show_explicit_meta():
    rc, out, _ = _run_cli("atoms", "show", "loss", "cross_entropy", "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["name"] == "cross_entropy"
    assert out["metadata_status"] == "explicit"
    assert "ignore_index" in out["params"]
    assert out["params"]["ignore_index"]["default"] == -100
    assert out["label_contract"]["ignore_index_param"] == "ignore_index"


def test_cli_atoms_show_explicit_meta_for_mlp():
    """v1.9 에서 모든 built-in atom 이 explicit. mlp 도 params 와 contract 보유."""
    rc, out, _ = _run_cli("atoms", "show", "model", "mlp", "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["metadata_status"] == "explicit"
    assert "in_dim" in out["params"]
    assert "out_dim" in out["params"]


def test_cli_atoms_show_unknown_returns_error():
    rc, out, _ = _run_cli("atoms", "show", "loss", "does_not_exist", "--json")
    assert rc == 1
    assert isinstance(out, dict)
    assert "error" in out


def test_cli_atoms_validate_ref_valid(tmp_path):
    """JSON ref file → validate-ref 통과."""
    ref_file = tmp_path / "ref.json"
    ref_file.write_text(
        '{"kind": "loss", "name": "cross_entropy", '
        '"params": {"ignore_index": -1}}'
    )
    rc, out, _ = _run_cli("atoms", "validate-ref", str(ref_file), "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["valid"] is True
    assert out["errors"] == []


def test_cli_atoms_validate_ref_unknown_param(tmp_path):
    ref_file = tmp_path / "ref.json"
    ref_file.write_text(
        '{"kind": "loss", "name": "cross_entropy", '
        '"params": {"bogus_param": 0}}'
    )
    rc, out, _ = _run_cli("atoms", "validate-ref", str(ref_file), "--json")
    assert rc == 1
    assert isinstance(out, dict)
    assert out["valid"] is False
    assert any("unknown" in e for e in out["errors"])


def test_cli_atoms_validate_ref_unknown_atom(tmp_path):
    ref_file = tmp_path / "ref.json"
    ref_file.write_text(
        '{"kind": "loss", "name": "no_such_loss_xyz", "params": {}}'
    )
    rc, out, _ = _run_cli("atoms", "validate-ref", str(ref_file), "--json")
    assert rc == 1
    assert isinstance(out, dict)
    assert out["valid"] is False


def test_cli_atoms_validate_ref_kind_mismatch(tmp_path):
    """kind 가 잘못 지정된 경우 — kind=loss 인 cross_entropy 를 model 로 표기."""
    ref_file = tmp_path / "ref.json"
    ref_file.write_text(
        '{"kind": "model", "name": "cross_entropy", "params": {}}'
    )
    rc, out, _ = _run_cli("atoms", "validate-ref", str(ref_file), "--json")
    # cross_entropy 는 model registry 에 없음 → unknown
    assert rc == 1
    assert isinstance(out, dict)
    assert out["valid"] is False


# ─────────────────────────────────────────────────────────────────────
# v1.10: pcq init-experiment / apply-plan / validate --plan
# ─────────────────────────────────────────────────────────────────────


def test_cli_init_experiment(tmp_path):
    rc, out, _ = _run_cli(
        "init-experiment",
        "--preset", "vision/fake_smoke",
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
        "--preset", "vision/fake_smoke",
        "--output", str(tmp_path),
        "--json",
    )
    assert rc == 0
    assert out["name"] == "vision-fake_smoke"


def test_cli_init_experiment_force(tmp_path):
    (tmp_path / "cq.yaml").write_text("# pre-existing\n")
    rc, out, _ = _run_cli(
        "init-experiment",
        "--preset", "vision/fake_smoke",
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
        "--preset", "vision/fake_smoke",
        "--output", str(tmp_path),
        "--force",
        "--json",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema_version": 1,
        "id": "exp-cli-001",
        "base": {"preset": "vision/fake_smoke"},
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


def test_cli_apply_plan_rejected_unknown_atom(tmp_path):
    _run_cli(
        "init-experiment",
        "--preset", "vision/fake_smoke",
        "--output", str(tmp_path),
        "--force",
        "--json",
    )
    plan_path = tmp_path / "bad.json"
    plan_path.write_text(json.dumps({
        "schema_version": 1,
        "id": "exp-cli-bad",
        "base": {"preset": "vision/fake_smoke"},
        "changes": [
            {"op": "set_atom", "atom": "loss", "name": "no_such_loss"},
        ],
    }))
    rc, out, _ = _run_cli(
        "apply-plan", str(plan_path),
        "--path", str(tmp_path),
        "--json",
    )
    assert rc == 1
    assert isinstance(out, dict)
    assert out["status"] == "rejected"


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
        "--preset", "vision/fake_smoke",
        "--output", str(tmp_path),
        "--force",
        "--json",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema_version": 1,
        "id": "exp-validate-ok",
        "base": {"preset": "vision/fake_smoke"},
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


def test_cli_validate_with_invalid_plan_atom_fails(tmp_path):
    _run_cli(
        "init-experiment",
        "--preset", "vision/fake_smoke",
        "--output", str(tmp_path),
        "--force",
        "--json",
    )
    plan_path = tmp_path / "bad_plan.json"
    plan_path.write_text(json.dumps({
        "schema_version": 1,
        "id": "exp-bad-validate",
        "base": {"preset": "vision/fake_smoke"},
        "changes": [
            {"op": "set_atom", "atom": "loss", "name": "nonexistent"},
        ],
    }))
    rc, out, _ = _run_cli(
        "validate", str(tmp_path),
        "--plan", str(plan_path),
        "--json",
    )
    assert rc == 1
    plan_checks = [c for c in out["checks"] if c["id"] == "plan_validation"]
    assert plan_checks
    assert any(c["status"] == "fail" for c in plan_checks)


def test_cli_validate_with_missing_plan_file(tmp_path):
    _run_cli(
        "init-experiment",
        "--preset", "vision/fake_smoke",
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


# ─────────────────────────────────────────────────────────────────────
# v1.12: atoms list --source / scaffold / validate-local / smoke
# ─────────────────────────────────────────────────────────────────────


def test_cli_atoms_list_with_source_filter():
    rc, out, _ = _run_cli("atoms", "list", "--source", "builtin", "--json")
    assert rc == 0
    assert isinstance(out, dict)
    for kind, atoms in out["atoms"].items():
        for a in atoms:
            assert a["source"] == "builtin"


def test_cli_atoms_list_includes_source_and_module_field():
    rc, out, _ = _run_cli("atoms", "list", "--json")
    assert rc == 0
    assert isinstance(out, dict)
    samples = out["atoms"]["loss"]
    assert samples
    sample = samples[0]
    assert "source" in sample
    assert "module" in sample


def test_cli_atoms_list_load_project(tmp_path):
    (tmp_path / "pcq_atoms.py").write_text('''
import pcq
pcq.register_loss(
    "cli_load_proj_v12",
    factory=lambda: __import__("torch").nn.CrossEntropyLoss(),
    meta={"tasks": ["classification"]},
)
''', encoding="utf-8")
    rc, out, _ = _run_cli(
        "atoms", "list",
        "--source", "project",
        "--load-project", str(tmp_path),
        "--kind", "loss",
        "--json",
    )
    assert rc == 0
    names = [a["name"] for a in out["atoms"]["loss"]]
    assert "cli_load_proj_v12" in names


def test_cli_atoms_scaffold(tmp_path):
    rc, out, _ = _run_cli(
        "atoms", "scaffold", "model", "scaff_test_model_v12",
        "--path", str(tmp_path),
        "--json",
    )
    assert rc == 0
    assert isinstance(out, dict)
    assert out["status"] == "created"
    assert (tmp_path / "atoms" / "models.py").exists()


def test_cli_atoms_scaffold_invalid_kind_fails(tmp_path):
    rc, out, _ = _run_cli(
        "atoms", "scaffold", "bogus", "x_v12",
        "--path", str(tmp_path),
        "--json",
    )
    # argparse choices 검증으로 rc=2 (argparse error)
    assert rc == 2


def test_cli_atoms_validate_local_ok(tmp_path):
    (tmp_path / "pcq_atoms.py").write_text('''
import pcq
pcq.register_loss(
    "cli_test_loss_v12",
    factory=lambda: __import__("torch").nn.CrossEntropyLoss(),
    meta={
        "tasks": ["classification"],
        "input_contract": {"logits": ["B", "C"], "target": ["B"]},
    },
)
''', encoding="utf-8")
    rc, out, _ = _run_cli(
        "atoms", "validate-local", str(tmp_path), "--json",
    )
    # pass 또는 warn → rc=0
    assert rc == 0
    assert isinstance(out, dict)
    assert out["status"] in ("pass", "warn")


def test_cli_atoms_validate_local_fail(tmp_path):
    (tmp_path / "pcq_atoms.py").write_text(
        "raise ImportError('cli_oops_v12')\n", encoding="utf-8",
    )
    rc, out, _ = _run_cli(
        "atoms", "validate-local", str(tmp_path), "--json",
    )
    assert rc == 1
    assert out["status"] == "fail"


def test_cli_atoms_smoke_builtin():
    rc, out, _ = _run_cli("atoms", "smoke", "model", "mlp", "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["passed"] is True


def test_cli_atoms_smoke_unknown_atom():
    rc, out, _ = _run_cli(
        "atoms", "smoke", "model", "no_such_atom_v12", "--json",
    )
    assert rc == 1
    assert out["passed"] is False


def test_cli_atoms_smoke_with_load_project(tmp_path):
    (tmp_path / "pcq_atoms.py").write_text('''
import pcq
import torch.nn as nn

pcq.register_model(
    "cli_smoke_proj_model_v12",
    factory=lambda: nn.Linear(8, 2),
    meta={
        "tasks": ["classification"],
        "input_contract": {"x": ["B", "8"]},
        "output_contract": {"logits": ["B", "2"]},
    },
)
''', encoding="utf-8")
    rc, out, _ = _run_cli(
        "atoms", "smoke", "model", "cli_smoke_proj_model_v12",
        "--load-project", str(tmp_path),
        "--json",
    )
    assert rc == 0
    assert out["passed"] is True


def test_cli_init_creates_cq_atoms_py(tmp_path):
    rc, out, _ = _run_cli(
        "init-experiment",
        "--preset", "vision/fake_smoke",
        "--output", str(tmp_path),
        "--force",
        "--json",
    )
    assert rc == 0
    assert "pcq_atoms.py" in out["files_created"]
    assert (tmp_path / "pcq_atoms.py").exists()
    assert (tmp_path / "atoms" / "__init__.py").exists()


def test_cli_init_script_style_without_preset(tmp_path):
    rc, out, _ = _run_cli(
        "init-experiment",
        "--style", "script",
        "--output", str(tmp_path),
        "--json",
    )
    assert rc == 0
    assert out["style"] == "script"
    assert out["preset"] == ""
    assert "train.py" in out["files_created"]
    assert "pcq_atoms.py" not in out["files_created"]
    assert "eval_acc" in (tmp_path / "cq.yaml").read_text()


def test_cli_init_experiment_style_without_preset(tmp_path):
    rc, out, _ = _run_cli(
        "init-experiment",
        "--style", "experiment",
        "--output", str(tmp_path),
        "--json",
    )
    assert rc == 0
    assert out["style"] == "experiment"
    assert "pcq_atoms.py" in out["files_created"]
    text = (tmp_path / "cq.yaml").read_text()
    assert "train_loss" in text
    assert "preset:" not in text


def test_cli_dry_run_script_graceful(tmp_path):
    """script project 에서 dry-run 은 rc=0 + detail 메시지."""
    _run_cli(
        "init-experiment",
        "--style", "script",
        "--output", str(tmp_path),
        "--force",
        "--json",
    )
    rc, out, _ = _run_cli("dry-run", str(tmp_path), "--json")
    assert rc == 0
    assert isinstance(out, dict)
    assert out["kind"] == "script"
    assert "detail" in out
    assert "expected_artifacts" in out


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


def _build_completed_run(tmp_path: Path, monitor: str = "eval_iou", value: float = 0.7):
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


def test_cli_validate_script_complete(tmp_path):
    """script project 의 train.py 가 contract 를 모두 갖추면 validate pass/warn."""
    _run_cli(
        "init-experiment",
        "--style", "script",
        "--output", str(tmp_path),
        "--force",
        "--json",
    )
    rc, out, _ = _run_cli("validate", str(tmp_path), "--json")
    assert rc == 0
    assert isinstance(out, dict)
    # script-aware gate 들이 존재
    ids = {c["id"] for c in out["checks"]}
    assert "cq_config_called" in ids
    assert "standard_artifacts_helper" in ids


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
