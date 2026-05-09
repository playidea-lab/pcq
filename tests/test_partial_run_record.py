"""save_partial_run_record (v2.11) — streaming partial RunRecord."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

import pcq
from pcq.contract import _atomic_write_json


def _setup_cfg(out_dir: Path, **extra) -> Path:
    """CQ_CONFIG_JSON tmp 파일 생성."""
    cfg = {"output_dir": str(out_dir), "seed": 42}
    cfg.update(extra)
    p = out_dir.parent / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def test_partial_run_record_writes_minimal_run_record(tmp_path, monkeypatch):
    """partial dump 가 valid run_record.json 을 만든다."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))

    rr_path = pcq.save_partial_run_record(history=[])
    assert rr_path.exists()
    assert rr_path.name == "run_record.json"

    rr = json.loads(rr_path.read_text())
    assert rr["schema_version"] == 1
    assert rr["run"]["partial"] is True
    assert rr["run"]["status"] == "running"
    assert rr["run"]["last_updated_at"]
    # finished_at 은 partial 이므로 없음 (None → to_dict 가 제거)
    assert "finished_at" not in rr["run"]


def test_partial_to_final_transition_flips_partial_false(tmp_path, monkeypatch):
    """partial=true 로 시작 → finalize_run() → partial 키 제거 (False default)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))

    pcq.save_partial_run_record(history=[{"epoch": 0, "loss": 0.5}])
    mid = json.loads((out_dir / "run_record.json").read_text())
    assert mid["run"]["partial"] is True

    # 이후 finalize.
    pcq.save_metrics([{"epoch": 0, "loss": 0.5}])
    pcq.save_run_summary(history=[{"epoch": 0, "loss": 0.5}])
    pcq.save_manifest()
    pcq.finalize_run(history=[{"epoch": 0, "loss": 0.5}])
    final = json.loads((out_dir / "run_record.json").read_text())
    # partial=False default 는 to_dict 가 제거 — 키 부재 == False.
    assert "partial" not in final["run"]
    assert final["run"]["status"] == "completed"
    assert final["run"]["finished_at"]


def test_partial_status_checkpointed_allowed(tmp_path, monkeypatch):
    """checkpointed status 도 받는다 — 'running' 의 다른 의미."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))

    rr_path = pcq.save_partial_run_record(status="checkpointed")
    rr = json.loads(rr_path.read_text())
    assert rr["run"]["status"] == "checkpointed"
    assert rr["run"]["partial"] is True


def test_partial_status_rejects_finalized_states(tmp_path, monkeypatch):
    """completed/failed/partial 은 finalize_run 영역 — partial helper 거부."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))

    with pytest.raises(ValueError, match="running.*checkpointed"):
        pcq.save_partial_run_record(status="completed")
    with pytest.raises(ValueError):
        pcq.save_partial_run_record(status="failed")


def test_partial_atomic_write_always_parseable(tmp_path):
    """tmp+rename 으로 reader 가 항상 valid JSON 만 본다."""
    target = tmp_path / "run_record.json"
    payload = {"schema_version": 1, "run": {"partial": True}}
    _atomic_write_json(target, payload)
    assert target.exists()
    # 같은 디렉토리에 tmp 파일이 남지 않아야 한다 — os.replace 후 정리.
    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert leftover == [], f"unexpected tmp files: {leftover}"
    assert json.loads(target.read_text()) == payload


def test_partial_atomic_write_concurrent_readers(tmp_path):
    """동시 read 중 write — reader 가 항상 완전한 JSON 만 read."""
    target = tmp_path / "run_record.json"
    _atomic_write_json(target, {"epoch": 0})

    stop = threading.Event()
    errors: list[Exception] = []

    def reader():
        while not stop.is_set():
            try:
                data = json.loads(target.read_text())
                assert "epoch" in data
            except json.JSONDecodeError as e:
                errors.append(e)
                return

    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(50):
            _atomic_write_json(target, {"epoch": i})
    finally:
        stop.set()
        t.join(timeout=2.0)
    assert not errors, f"reader saw partial JSON: {errors}"


def test_partial_save_no_chdir_no_env_pollution(tmp_path):
    """다른 cwd 에서 호출해도 명시 output_dir 로 작성 (chdir 부작용 없음)."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "cq.yaml").write_text(
        "name: p\ncmd: c\nconfigs:\n  output_dir: out\n  seed: 1\n",
        encoding="utf-8",
    )
    out_dir = project / "out"

    # cwd 는 project 외부.
    other = tmp_path / "elsewhere"
    other.mkdir()
    cwd_before = os.getcwd()
    try:
        os.chdir(other)
        rr_path = pcq.save_partial_run_record(
            output_dir=out_dir, project_root=project
        )
    finally:
        os.chdir(cwd_before)
    assert rr_path == (out_dir / "run_record.json").resolve()
    rr = json.loads(rr_path.read_text())
    assert rr["run"]["partial"] is True


def test_validate_run_partial_skips_evidence_gates(tmp_path, monkeypatch):
    """partial=true 인 run 은 strictness>=3 evidence gates 가 skip 되고 단일
    run_finalized fail 로 대체된다."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))

    pcq.save_partial_run_record(history=[])
    from pcq.agent.validate_run import validate_run

    report = validate_run(out_dir, strictness=3)
    ids = {c.id for c in report.checks}
    # run_finalized gate 가 fail 로 등장
    assert "run_finalized" in ids
    finalized_check = next(c for c in report.checks if c.id == "run_finalized")
    assert finalized_check.status == "fail"
    # source/seed/lockfile/inputs evidence gates 는 skip — 추가되지 않음.
    assert "lockfile_evidence" not in ids
    assert "source_reproducibility" not in ids
    # manifest/run_summary missing 은 warn (downgraded) 이어야 함.
    manifest_check = next(c for c in report.checks if c.id == "manifest_present")
    assert manifest_check.status == "warn"
    rs_check = next(c for c in report.checks if c.id == "run_summary_present")
    assert rs_check.status == "warn"


def test_validate_run_strictness3_fails_on_partial_marker(tmp_path, monkeypatch):
    """strictness 3 + partial=true → run_finalized fail."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))

    pcq.save_partial_run_record()
    from pcq.agent.validate_run import validate_run

    report = validate_run(out_dir, strictness=3)
    assert report.status == "fail"
    finalized = next(c for c in report.checks if c.id == "run_finalized")
    assert "not yet finalized" in finalized.detail


def test_validate_run_finalized_passes_run_finalized(tmp_path, monkeypatch):
    """finalize 후 run_finalized gate 는 pass."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))

    history = [{"epoch": 0, "loss": 0.1}]
    pcq.save_metrics(history)
    pcq.save_run_summary(history=history)
    pcq.save_manifest()
    pcq.finalize_run(history=history)

    from pcq.agent.validate_run import validate_run

    report = validate_run(out_dir, strictness=3)
    finalized = next(
        (c for c in report.checks if c.id == "run_finalized"), None
    )
    # finalized run 은 evidence gate 들이 평가됨 — run_finalized 는 pass.
    assert finalized is not None
    assert finalized.status == "pass"


def test_partial_then_failed_finalize(tmp_path, monkeypatch):
    """partial → finalize_run(status='failed') 정상 — partial=False, status=failed."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))

    pcq.save_partial_run_record(history=[{"epoch": 0, "loss": 0.9}])
    pcq.save_metrics([{"epoch": 0, "loss": 0.9}])
    pcq.save_run_summary(history=[{"epoch": 0, "loss": 0.9}], status="failed")
    pcq.save_manifest()
    pcq.finalize_run(history=[{"epoch": 0, "loss": 0.9}], status="failed")

    rr = json.loads((out_dir / "run_record.json").read_text())
    assert rr["run"]["status"] == "failed"
    assert "partial" not in rr["run"]   # default False — to_dict 가 제거.


def test_partial_last_updated_at_advances(tmp_path, monkeypatch):
    """반복 호출 시 last_updated_at 이 갱신된다 (ISO timestamp)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))

    pcq.save_partial_run_record()
    rr1 = json.loads((out_dir / "run_record.json").read_text())
    t1 = rr1["run"]["last_updated_at"]

    # 두 번째 dump (시각 progresses; 정확히 동일이어도 string 형식 보장됨).
    pcq.save_partial_run_record()
    rr2 = json.loads((out_dir / "run_record.json").read_text())
    t2 = rr2["run"]["last_updated_at"]

    # 둘 다 ISO-Z 포맷, 두 번째가 첫 번째보다 늦거나 같음.
    assert t1.endswith("Z") and t2.endswith("Z")
    assert t2 >= t1


def test_partial_run_record_export_from_cq_namespace():
    """pcq.save_partial_run_record 가 top-level 로 export 되어 있다."""
    assert hasattr(pcq, "save_partial_run_record")
    assert callable(pcq.save_partial_run_record)
