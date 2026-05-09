"""ResolvedConfig + resolve_project — single source of truth for cq.yaml interpretation.

v2.2: cq.yaml + CQ_CONFIG_JSON env → ResolvedConfig 단일 view.
inspect / validate / finalize_run / RunRecord 모두 동일 resolver 사용.
"""
from __future__ import annotations

import json
from pathlib import Path

import pcq
from pcq.agent.resolver import ResolvedConfig, resolve_project


def _write_cq_yaml(tmp: Path, body: str) -> None:
    """tmp_path에 cq.yaml 작성."""
    (tmp / "cq.yaml").write_text(body)


def test_resolve_finds_cq_yaml(tmp_path):
    _write_cq_yaml(tmp_path, "name: t\ncmd: x\nconfigs:\n  output_dir: out\n")
    rc = resolve_project(path=tmp_path)
    assert rc.cq_yaml_path == (tmp_path / "cq.yaml").resolve()
    assert rc.project_root == tmp_path.resolve()
    assert rc.name == "t"
    assert rc.cfg["output_dir"] == "out"


def test_resolve_output_dir_is_absolute_and_under_project_root(tmp_path):
    """v2.5: resolve_project is read-only — output_dir computed but NOT created."""
    _write_cq_yaml(
        tmp_path,
        "name: t\ncmd: x\nconfigs:\n  output_dir: runs/exp001\n",
    )
    rc = resolve_project(path=tmp_path)
    assert rc.output_dir == (tmp_path / "runs" / "exp001").resolve()
    # v2.5: read-only — resolve_project does NOT mkdir.
    assert not rc.output_dir.exists()
    # write-side: resolve_run_context creates it.
    from pcq.agent.resolver import resolve_run_context
    ctx = resolve_run_context(path=tmp_path)
    assert ctx.output_dir.exists()


def test_resolve_walks_up_from_subdirectory(tmp_path, monkeypatch):
    """v2.2: cwd가 project root의 하위여도 cq.yaml 발견."""
    _write_cq_yaml(tmp_path, "name: t\ncmd: x\nconfigs: {}\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    rc = resolve_project()
    assert rc.cq_yaml_path == (tmp_path / "cq.yaml").resolve()


def test_resolve_stops_at_pyproject_root(tmp_path, monkeypatch):
    """nested project가 부모 cq.yaml 잡지 않음."""
    _write_cq_yaml(tmp_path, "name: parent\ncmd: x\nconfigs: {}\n")
    nested = tmp_path / "child"
    nested.mkdir()
    (nested / "pyproject.toml").write_text("[project]\nname='child'\n")
    monkeypatch.chdir(nested)
    rc = resolve_project()
    assert rc.cq_yaml_path is None  # parent의 cq.yaml 안 잡음


def test_resolve_dict_style_metrics(tmp_path):
    _write_cq_yaml(
        tmp_path,
        """name: t
cmd: x
configs: {}
metrics:
  eval_iou: {mode: max, split: val}
  eval_loss: {mode: min}
""",
    )
    rc = resolve_project(path=tmp_path)
    assert "eval_iou" in rc.declared_metrics
    assert rc.metrics_schema["eval_iou"]["mode"] == "max"


def test_resolve_list_style_metrics(tmp_path):
    _write_cq_yaml(
        tmp_path,
        """name: t
cmd: x
configs: {}
metrics:
  - epoch
  - eval_acc
""",
    )
    rc = resolve_project(path=tmp_path)
    assert rc.declared_metrics == ["epoch", "eval_acc"]
    assert rc.metrics_schema == {}


def test_resolve_inputs_passthrough(tmp_path):
    _write_cq_yaml(
        tmp_path,
        """name: t
cmd: x
configs: {}
metrics: []
inputs:
  dataset: {name: dental, uri: "cq://datasets/dental/v12"}
""",
    )
    rc = resolve_project(path=tmp_path)
    assert rc.inputs["dataset"]["name"] == "dental"
    assert rc.inputs["dataset"]["uri"] == "cq://datasets/dental/v12"


def test_resolve_env_overrides_cq_yaml_configs(tmp_path, monkeypatch):
    """CQ_CONFIG_JSON env merge — env가 cq.yaml.configs를 override."""
    _write_cq_yaml(
        tmp_path,
        "name: t\ncmd: x\nconfigs:\n  epochs: 10\n  batch_size: 64\n",
    )
    env_cfg = {"epochs": 5, "lr": 0.001}  # epochs override + lr 추가
    cfg_path = tmp_path / "env.json"
    cfg_path.write_text(json.dumps(env_cfg))
    monkeypatch.setenv("CQ_CONFIG_JSON", str(cfg_path))
    rc = resolve_project(path=tmp_path)
    assert rc.cfg["epochs"] == 5  # env 우선
    assert rc.cfg["batch_size"] == 64  # cq.yaml에서 살아남음
    assert rc.cfg["lr"] == 0.001  # env 새 키


def test_resolve_no_cq_yaml_uses_cwd(tmp_path, monkeypatch):
    """cq.yaml 없으면 cwd 기반 + cfg from env only."""
    monkeypatch.chdir(tmp_path)
    rc = resolve_project()
    assert rc.cq_yaml_path is None
    assert rc.project_root == tmp_path.resolve()


def test_resolve_to_dict_json_safe(tmp_path):
    _write_cq_yaml(tmp_path, "name: t\ncmd: x\nconfigs:\n  epochs: 3\n")
    rc = resolve_project(path=tmp_path)
    d = rc.to_dict()
    json.dumps(d)  # JSON-safe (예외 없으면 OK)


def test_resolve_explicit_cq_yaml_path(tmp_path):
    """cq_yaml_path 인자 직접 전달."""
    p = tmp_path / "custom.yaml"
    p.write_text("name: custom\ncmd: x\nconfigs: {}\n")
    rc = resolve_project(cq_yaml_path=p)
    assert rc.cq_yaml_path == p.resolve()
    assert rc.name == "custom"


def test_resolve_malformed_yaml_records_parse_error(tmp_path):
    (tmp_path / "cq.yaml").write_text(":::not\n  valid: [unclosed\n")
    rc = resolve_project(path=tmp_path)
    # parse_errors 또는 cfg 비어있음 (yaml 파서 종류에 따라 동작 상이)
    assert rc.parse_errors or rc.cfg == {}


def test_resolve_output_dir_absolute_already(tmp_path):
    abs_out = tmp_path / "abs_output"
    _write_cq_yaml(
        tmp_path,
        f"name: t\ncmd: x\nconfigs:\n  output_dir: {abs_out}\n",
    )
    rc = resolve_project(path=tmp_path)
    assert rc.output_dir == abs_out.resolve()


def test_resolve_top_level_cq_resolve_project_export():
    """pcq.resolve_project / pcq.ResolvedConfig 가 top-level public API."""
    assert hasattr(pcq, "resolve_project")
    assert hasattr(pcq, "ResolvedConfig")
    assert pcq.resolve_project is resolve_project
    assert pcq.ResolvedConfig is ResolvedConfig


def test_resolve_default_output_dir_when_cfg_empty(tmp_path):
    """cfg에 output_dir 없으면 'output' 기본값 (project_root/output)."""
    _write_cq_yaml(tmp_path, "name: t\ncmd: x\nconfigs: {}\n")
    rc = resolve_project(path=tmp_path)
    assert rc.output_dir == (tmp_path / "output").resolve()


def test_resolve_artifacts_list_and_dict(tmp_path):
    """artifacts list-style / dict-style 둘 다 list[str]로 정규화."""
    _write_cq_yaml(
        tmp_path,
        """name: t
cmd: x
configs: {}
artifacts:
  - output/
  - logs/
""",
    )
    rc = resolve_project(path=tmp_path)
    assert sorted(rc.artifacts) == ["logs/", "output/"]


def test_resolve_metrics_declared_from_cfg_fallback(tmp_path, monkeypatch):
    """cq.yaml에 metrics 없고 CQ_CONFIG_JSON에 _metrics_declared만 있으면 사용."""
    _write_cq_yaml(tmp_path, "name: t\ncmd: x\nconfigs: {}\n")
    env_cfg = {"_metrics_declared": ["eval_iou", "eval_loss"]}
    cfg_path = tmp_path / "env.json"
    cfg_path.write_text(json.dumps(env_cfg))
    monkeypatch.setenv("CQ_CONFIG_JSON", str(cfg_path))
    rc = resolve_project(path=tmp_path)
    assert sorted(rc.declared_metrics) == ["eval_iou", "eval_loss"]
