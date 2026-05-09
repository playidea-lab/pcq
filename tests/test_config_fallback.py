"""tests/test_config_fallback.py — Fix 1 (G7-5/G0-1).

pcq.config() 가 CQ_CONFIG_JSON env 부재 시 cq.yaml.configs 로 fallback.

RED Phase: dogfood 가 발견한 P0 — fresh user 가 `python train.py` 직접 실행 못함.
GREEN Phase: pcq.core.config() 가 resolve_project() 호출하여 cfg 반환.

우선순위:
  1. CQ_CONFIG_JSON env 명시 시 → JSON 파일 read (현재 동작 유지)
  2. env 없으면 → resolve_project() 호출. resolved.cfg 반환
  3. 둘 다 없으면 → 명시 RuntimeError
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pcq
from pcq import core


@pytest.fixture(autouse=True)
def reset_module_state():
    # 테스트 간 module-level cache 초기화.
    core._undeclared_warned.clear()
    core._undeclared_count.clear()
    core._reset_declared_cache()
    yield
    core._undeclared_warned.clear()
    core._undeclared_count.clear()
    core._reset_declared_cache()


def _write_cq_yaml(project: Path, cfg_dict: dict, name: str = "demo") -> Path:
    """cq.yaml 작성 — 테스트용 fixture 헬퍼."""
    from pcq.agent.yaml_io import write_yaml

    data = {
        "name": name,
        "cmd": "python train.py",
        "configs": cfg_dict,
    }
    p = project / "cq.yaml"
    write_yaml(data, p)
    return p


def test_config_falls_back_to_cq_yaml_when_env_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """env 없고 cq.yaml.configs 있으면 그 값 반환."""
    project = tmp_path / "proj"
    project.mkdir()
    cfg = {"epochs": 5, "lr": 0.01, "batch_size": 64}
    _write_cq_yaml(project, cfg)

    monkeypatch.chdir(project)
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)

    result = pcq.config()
    assert result["epochs"] == 5
    assert result["lr"] == 0.01
    assert result["batch_size"] == 64


def test_config_env_overrides_cq_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """env 와 cq.yaml 둘 다 있으면 env 우선 (resolver 의 merge 정책)."""
    project = tmp_path / "proj"
    project.mkdir()
    yaml_cfg = {"epochs": 5, "lr": 0.01}
    _write_cq_yaml(project, yaml_cfg)

    env_cfg_path = project / "cfg.json"
    env_cfg_path.write_text(json.dumps({"epochs": 99, "lr": 0.001}))

    monkeypatch.chdir(project)
    monkeypatch.setenv("CQ_CONFIG_JSON", str(env_cfg_path))

    result = pcq.config()
    # env 가 cq.yaml 을 override (resolver 의 표준 priority)
    assert result["epochs"] == 99
    assert result["lr"] == 0.001


def test_config_neither_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """env 도 cq.yaml 도 없으면 명확한 에러."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    # tmp_path 안에는 cq.yaml 없음

    with pytest.raises(RuntimeError) as ei:
        pcq.config()
    msg = str(ei.value)
    assert "cq.yaml" in msg or "CQ_CONFIG_JSON" in msg


def test_config_cq_yaml_missing_configs_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """cq.yaml 은 있는데 configs 섹션이 비어 있어도 OK (빈 dict 반환)."""
    from pcq.agent.yaml_io import write_yaml

    project = tmp_path / "proj"
    project.mkdir()
    write_yaml({"name": "demo", "cmd": "python train.py"}, project / "cq.yaml")

    monkeypatch.chdir(project)
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)

    result = pcq.config()
    assert result == {}


def test_config_walks_up_to_find_cq_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """cwd 가 project 의 sub-dir 여도 ancestor walk-up 으로 cq.yaml 찾음."""
    project = tmp_path / "proj"
    project.mkdir()
    _write_cq_yaml(project, {"epochs": 7})
    sub = project / "runs" / "exp0"
    sub.mkdir(parents=True)

    monkeypatch.chdir(sub)
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)

    result = pcq.config()
    assert result["epochs"] == 7
