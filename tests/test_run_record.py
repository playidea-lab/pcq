"""run_record.json schema + finalize_run (v1.16+)."""
import json
import subprocess
from pathlib import Path

import pcq
from pcq.agent.run_record import RunRecord


def _setup_cfg(tmp_path: Path, **extra) -> Path:
    cfg = {"output_dir": str(tmp_path), "seed": 42}
    cfg.update(extra)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def _git(project: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )


def test_finalize_run_writes_run_record(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    history = [{"epoch": 0, "eval_loss": 0.5}, {"epoch": 1, "eval_loss": 0.3}]
    pcq.save_metrics(history)
    pcq.save_run_summary(history=history, status="completed")
    pcq.save_manifest()
    rr_path = pcq.finalize_run(history=history)
    assert rr_path.exists()
    assert rr_path.name == "run_record.json"


def test_run_record_has_required_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    pcq.save_metrics([{"epoch": 0, "eval_loss": 0.5}])
    pcq.save_run_summary(history=[{"epoch": 0, "eval_loss": 0.5}])
    pcq.save_manifest()
    pcq.finalize_run(history=[{"epoch": 0, "eval_loss": 0.5}])
    rr = json.loads((tmp_path / "run_record.json").read_text())
    for k in (
        "schema_version",
        "run",
        "execution",
        "source",
        "environment",
        "metrics",
        "artifacts",
    ):
        assert k in rr, f"missing key: {k}"


def test_run_record_environment_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    pcq.save_metrics([])
    pcq.save_run_summary(history=[])
    pcq.save_manifest()
    pcq.finalize_run(history=[])
    rr = json.loads((tmp_path / "run_record.json").read_text())
    assert rr["environment"]["python"]
    assert "platform" in rr["environment"]
    assert rr["environment"]["pcq_version"]
    assert rr["environment"]["device"]


def test_run_record_inputs_from_cq_yaml(tmp_path, monkeypatch):
    """cq.yaml의 inputs section이 run_record.inputs로 그대로 복사."""
    (tmp_path / "cq.yaml").write_text("""
name: t
cmd: c
configs: {}
metrics: [eval_acc]
artifacts: [output/]
inputs:
  dataset:
    name: dental
    uri: cq://datasets/dental/v12
""")
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))
    pcq.save_metrics([])
    pcq.save_run_summary(history=[])
    pcq.save_manifest()
    pcq.finalize_run(history=[])
    rr = json.loads((out_dir / "run_record.json").read_text())
    assert rr["inputs"]["dataset"]["name"] == "dental"
    assert rr["inputs"]["dataset"]["uri"] == "cq://datasets/dental/v12"
    assert rr["input_summary"]["count"] == 1
    assert rr["input_summary"]["identity"]["dataset"]["has_uri"] is True


def test_run_record_config_evidence_from_cq_yaml(tmp_path, monkeypatch):
    """cq.yaml identity and seed are recorded in RunRecord.config/source."""
    (tmp_path / "cq.yaml").write_text(
        """
name: t
cmd: c
configs:
  output_dir: out
  seed: 123
metrics:
  eval_acc:
    mode: max
artifacts: [out/]
inputs: {}
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    pcq.save_all(history=[{"epoch": 0, "eval_acc": 0.5}])

    rr = json.loads((tmp_path / "out" / "run_record.json").read_text())
    assert rr["config"]["seed"] == 123
    assert rr["config"]["cq_yaml_path"] == "cq.yaml"
    assert rr["config"]["cq_yaml_sha256"]
    assert rr["source"]["cq_yaml_path"] == "cq.yaml"
    assert rr["source"]["cq_yaml_sha256"] == rr["config"]["cq_yaml_sha256"]


def test_save_all_finalize_default_true(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    paths = pcq.save_all(history=[{"epoch": 0, "eval_acc": 0.5}])
    assert "run_record" in paths
    assert paths["run_record"].exists()


def test_save_all_finalize_false(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    paths = pcq.save_all(
        history=[{"epoch": 0, "eval_acc": 0.5}], finalize=False
    )
    assert "run_record" not in paths
    assert not (tmp_path / "run_record.json").exists()


def test_run_record_validation_status_in_record(tmp_path, monkeypatch):
    """finalize 후 validation_report.json도 작성 + run_record.validation 갱신."""
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    pcq.save_metrics([{"epoch": 0, "loss": 0.1}])
    pcq.save_run_summary(history=[{"epoch": 0, "loss": 0.1}])
    pcq.save_manifest()
    pcq.finalize_run(history=[{"epoch": 0, "loss": 0.1}])
    assert (tmp_path / "validation_report.json").exists()
    rr = json.loads((tmp_path / "run_record.json").read_text())
    assert rr["validation"]["status"] in ("pass", "warn", "fail")


def test_record_patch_opt_in(tmp_path, monkeypatch):
    """cfg.record_patch=true에서 dirty repo면 patch_sha256/changed_files 기록."""
    monkeypatch.setenv(
        "CQ_CONFIG_JSON", str(_setup_cfg(tmp_path, record_patch=True))
    )
    pcq.save_metrics([])
    pcq.save_run_summary(history=[])
    pcq.save_manifest()
    pcq.finalize_run(history=[])
    rr = json.loads((tmp_path / "run_record.json").read_text())
    # dirty면 patch_sha256 기록, clean이면 안 함 — 둘 다 OK
    assert "git_sha" in rr["source"]
    assert "dirty" in rr["source"]


def test_run_record_dataclass_to_dict():
    """RunRecord dataclass to_dict 직접 검증."""
    record = RunRecord()
    d = record.to_dict()
    assert d["schema_version"] == 1
    assert "run" in d
    assert "environment" in d
    assert "validation" in d


def test_finalize_run_summary_includes_target_metric(tmp_path, monkeypatch):
    """run_summary 의 monitor.name 이 run_record.summary.target_metric 으로 흘러 들어감."""
    monkeypatch.setenv(
        "CQ_CONFIG_JSON",
        str(_setup_cfg(tmp_path, monitor="eval_iou", mode="max")),
    )
    history = [{"epoch": 0, "eval_iou": 0.5}, {"epoch": 1, "eval_iou": 0.7}]
    pcq.save_metrics(history)
    pcq.save_run_summary(history=history, status="completed")
    pcq.save_manifest()
    pcq.finalize_run(history=history)
    rr = json.loads((tmp_path / "run_record.json").read_text())
    assert rr["summary"].get("target_metric") == "eval_iou"


def test_finalize_run_artifacts_from_manifest(tmp_path, monkeypatch):
    """artifacts 는 manifest.json files entry 를 그대로 가져온다."""
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    (tmp_path / "model.pt").write_bytes(b"fake")
    pcq.save_metrics([{"epoch": 0}])
    pcq.save_run_summary(history=[{"epoch": 0}])
    pcq.save_manifest()
    pcq.finalize_run(history=[{"epoch": 0}])
    rr = json.loads((tmp_path / "run_record.json").read_text())
    paths_in_artifacts = {a.get("path") for a in rr["artifacts"]}
    assert "model.pt" in paths_in_artifacts


def test_finalize_run_passes_plan_id_through(tmp_path, monkeypatch):
    """plan_id, intent 가 agent.* 로 흘러간다."""
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    pcq.save_metrics([])
    pcq.save_run_summary(history=[])
    pcq.save_manifest()
    pcq.finalize_run(
        history=[],
        plan_id="exp-plan-001",
        intent="baseline",
    )
    rr = json.loads((tmp_path / "run_record.json").read_text())
    assert rr["agent"].get("plan_id") == "exp-plan-001"
    assert rr["agent"].get("intent") == "baseline"


def test_save_all_strictness3_run_passes_with_complete_evidence(
    tmp_path, monkeypatch
):
    """Normal pcq run can pass validate-run strictness 3 after PR2."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\nname='strict3-demo'\n", encoding="utf-8"
    )
    (project / "uv.lock").write_text("# fake lock\n", encoding="utf-8")
    (project / "train.py").write_text("import pcq\n", encoding="utf-8")
    (project / "cq.yaml").write_text(
        """
name: strict3-demo
cmd: uv run python train.py
configs:
  output_dir: out
  seed: 42
  strictness: 3
  monitor: eval_acc
  mode: max
metrics:
  eval_acc:
    mode: max
artifacts: [out/]
inputs: {}
""",
        encoding="utf-8",
    )
    _git(project, "init")
    _git(project, "config", "user.email", "cq@example.test")
    _git(project, "config", "user.name", "cq test")
    _git(project, "add", "pyproject.toml", "uv.lock", "train.py", "cq.yaml")
    _git(project, "commit", "-m", "init")

    monkeypatch.chdir(project)
    paths = pcq.save_all(history=[{"epoch": 0, "eval_acc": 0.75}])

    out_dir = project / "out"
    assert paths["validation_report"] == out_dir / "validation_report.json"
    report = json.loads((out_dir / "validation_report.json").read_text())
    rr = json.loads((out_dir / "run_record.json").read_text())
    assert report["strictness"] == 3
    assert report["status"] == "pass"
    assert rr["validation"]["status"] == "pass"
    assert rr["source"]["git_sha"]
    assert rr["source"]["changed_files"]
    assert rr["environment"]["lockfile"] == "uv.lock"
    assert rr["environment"]["lockfile_sha256"]
    assert rr["config"]["seed"] == 42


def test_finalize_run_finds_cq_yaml_from_subdirectory(tmp_path, monkeypatch):
    """v2.2 audit-fix: finalize_run의 cq.yaml read가 cwd-relative였음 —
    sub-directory에서 학습 시 inputs/metrics_schema 누락. resolver의 ancestor
    walk-up으로 해결."""
    # project root with cq.yaml
    project = tmp_path / "myproj"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='x'\n")
    (project / "cq.yaml").write_text("""
name: t
cmd: c
configs: {}
metrics: [eval_acc]
inputs:
  dataset:
    name: dental
    uri: cq://datasets/dental/v12
""")
    # nested cwd
    deep = project / "scripts" / "exp"
    deep.mkdir(parents=True)
    output = deep / "out"
    output.mkdir()
    cfg_path = deep / "cfg.json"
    cfg_path.write_text(json.dumps({"output_dir": str(output)}))

    monkeypatch.chdir(deep)
    monkeypatch.setenv("CQ_CONFIG_JSON", str(cfg_path))

    pcq.save_metrics([])
    pcq.save_run_summary(history=[])
    pcq.save_manifest()
    pcq.finalize_run(history=[])

    rr = json.loads((output / "run_record.json").read_text())
    # 이전 v2.1.x: inputs={} (cq.yaml을 cwd 기준 못 찾음).
    # v2.2: resolver walk-up으로 발견 → inputs.dataset 보존.
    assert rr["inputs"]["dataset"]["name"] == "dental"
    assert rr["inputs"]["dataset"]["uri"] == "cq://datasets/dental/v12"


def test_lockfile_walks_up_from_subdirectory(tmp_path, monkeypatch):
    """v2.0.1 fix: cwd가 lockfile 보유 디렉토리의 하위여도 walk-up으로 발견."""
    import json
    # project root with uv.lock
    project = tmp_path / "myproj"
    project.mkdir()
    (project / "uv.lock").write_text("# fake lockfile")
    (project / "pyproject.toml").write_text("[project]\nname='x'")
    # nested cwd 2 levels deep
    deep = project / "a" / "b"
    deep.mkdir(parents=True)
    output = deep / "out"
    output.mkdir()
    cfg_path = deep / "cfg.json"
    cfg_path.write_text(json.dumps({"output_dir": str(output)}))

    monkeypatch.chdir(deep)
    monkeypatch.setenv("CQ_CONFIG_JSON", str(cfg_path))

    import pcq
    pcq.save_metrics([])
    pcq.save_run_summary(history=[])
    pcq.save_manifest()
    pcq.finalize_run(history=[])

    rr = json.loads((output / "run_record.json").read_text())
    assert rr["environment"]["lockfile"] == "uv.lock"
    assert rr["environment"]["lockfile_sha256"]


def test_lockfile_stops_at_project_root_marker(tmp_path, monkeypatch):
    """v2.0.1: pyproject.toml 만나면 ascent 중단 — nested project가 부모 lockfile 못 가져감."""
    import json
    # parent project with uv.lock
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "uv.lock").write_text("parent lock")
    # nested child project (own pyproject, no lockfile)
    child = parent / "child"
    child.mkdir()
    (child / "pyproject.toml").write_text("[project]\nname='child'")
    output = child / "out"
    output.mkdir()
    cfg_path = child / "cfg.json"
    cfg_path.write_text(json.dumps({"output_dir": str(output)}))

    monkeypatch.chdir(child)
    monkeypatch.setenv("CQ_CONFIG_JSON", str(cfg_path))

    import pcq
    pcq.save_metrics([])
    pcq.save_run_summary(history=[])
    pcq.save_manifest()
    pcq.finalize_run(history=[])

    rr = json.loads((output / "run_record.json").read_text())
    # child has pyproject but no lockfile → no parent walk-up
    assert "lockfile" not in rr["environment"]
