"""tests/test_compare_runs_config.py — Fix 4 (G1-4).

compare-runs 가 두 RunRecord 의 cq.yaml.configs 차이를 detect.

dogfood gen 0→1 에서 5 axis 변경했는데 config_changes=[] 였음. 원인: compare 는
RunRecord.agent.{recipe, overrides} 만 비교했고 실제 cq.yaml.configs 는 미비교.

fix: source 또는 config 섹션의 cq_yaml_path 를 통해 cq.yaml 을 read 하고 dict diff.
fail open — cq.yaml 못 찾으면 빈 list (현재 동작 유지).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pcq.agent.compare import compare_runs
from pcq.agent.yaml_io import write_yaml


def _build_minimal_run_record(
    out_dir: Path,
    cq_yaml_relpath: str,
    cq_yaml_sha256: str,
    run_id: str = "run-x",
) -> Path:
    """직접 RunRecord 구조를 작성 — finalize_run 의 무거운 의존성 없이.

    test 가 cq.yaml diff 만 검증하므로 최소 evidence 만 채운다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rr = {
        "schema_version": 1,
        "run": {"id": run_id, "name": "test", "status": "completed"},
        "execution": {"cmd": "python train.py", "config_path": "cq.yaml"},
        "source": {
            "git_sha": "deadbeef",
            "dirty": False,
            "cq_yaml_path": cq_yaml_relpath,
            "cq_yaml_sha256": cq_yaml_sha256,
        },
        "environment": {
            "python": "3.12.0",
            "platform": "test",
            "pcq_version": "test",
        },
        "config": {
            "cq_yaml_path": cq_yaml_relpath,
            "cq_yaml_sha256": cq_yaml_sha256,
        },
        "metrics": {"declared": [], "history_path": "metrics.json"},
        "artifacts": [],
    }
    rr_path = out_dir / "run_record.json"
    rr_path.write_text(json.dumps(rr, indent=2), encoding="utf-8")
    return rr_path


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def test_compare_runs_detects_config_changes(tmp_path: Path):
    """두 RunRecord 가 다른 cq.yaml.configs 를 가리키면 diff 노출."""
    project_a = tmp_path / "proj_a"
    project_a.mkdir()
    cq_a = {"name": "demo", "cmd": "python train.py", "configs": {
        "epochs": 10, "lr": 0.001, "batch_size": 64
    }}
    write_yaml(cq_a, project_a / "cq.yaml")

    project_b = tmp_path / "proj_b"
    project_b.mkdir()
    cq_b = {"name": "demo", "cmd": "python train.py", "configs": {
        "epochs": 10, "lr": 0.0001, "batch_size": 128, "dropout": 0.5
    }}
    write_yaml(cq_b, project_b / "cq.yaml")

    # output_dir 들은 각 project 안에 둔다.
    a_out = project_a / "output"
    b_out = project_b / "output"
    a_text = (project_a / "cq.yaml").read_text(encoding="utf-8")
    b_text = (project_b / "cq.yaml").read_text(encoding="utf-8")
    _build_minimal_run_record(
        a_out, "cq.yaml", _sha256_str(a_text), run_id="run-a"
    )
    _build_minimal_run_record(
        b_out, "cq.yaml", _sha256_str(b_text), run_id="run-b"
    )

    diff = compare_runs(a_out, b_out)
    keys = {c["key"] for c in diff.config_changes}
    # lr 변경, batch_size 변경, dropout 추가 (b only) 모두 detect.
    assert "lr" in keys, f"lr not in {keys}: {diff.config_changes}"
    assert "batch_size" in keys
    assert "dropout" in keys
    # epochs 는 동일 — 등장하지 말아야 함.
    assert "epochs" not in keys


def test_compare_runs_handles_missing_cq_yaml(tmp_path: Path):
    """cq.yaml path 가 깨졌어도 graceful (빈 list, no crash)."""
    a_out = tmp_path / "a_out"
    b_out = tmp_path / "b_out"
    # cq_yaml_path 는 설정하지만 실제 file 은 없음.
    _build_minimal_run_record(
        a_out, "ghost.yaml", "0" * 64, run_id="run-a"
    )
    _build_minimal_run_record(
        b_out, "ghost.yaml", "0" * 64, run_id="run-b"
    )

    diff = compare_runs(a_out, b_out)
    # crash 없이 완료. yaml-based config_changes 는 빈 list 또는 (ancestor 비교 외)
    # 다른 source (overrides/recipe) 만 detect 가능.
    assert isinstance(diff.config_changes, list)
    # 옛 동작 (overrides/recipe) 은 여전히 유지 — 둘 다 None 이므로 변화 없음.


def test_compare_runs_via_cq_yaml_sha256_skip_when_same(tmp_path: Path):
    """두 RunRecord 가 같은 cq_yaml_sha256 이면 yaml diff skip (빈 list)."""
    project = tmp_path / "proj"
    project.mkdir()
    cq_data = {"name": "demo", "cmd": "python train.py", "configs": {
        "epochs": 5, "lr": 0.01
    }}
    write_yaml(cq_data, project / "cq.yaml")
    cq_text = (project / "cq.yaml").read_text(encoding="utf-8")
    sha = _sha256_str(cq_text)

    a_out = project / "out_a"
    b_out = project / "out_b"
    _build_minimal_run_record(a_out, "cq.yaml", sha, run_id="run-a")
    _build_minimal_run_record(b_out, "cq.yaml", sha, run_id="run-b")

    diff = compare_runs(a_out, b_out)
    # 같은 sha → yaml-based diff 결과는 비어야 (다른 source 변화도 없으니 빈).
    yaml_keys = {c["key"] for c in diff.config_changes}
    assert "lr" not in yaml_keys
    assert "epochs" not in yaml_keys
