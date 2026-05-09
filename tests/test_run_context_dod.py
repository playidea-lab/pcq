"""v2.5.0 DoD: cq.yaml 해석 수렴 회귀 테스트.

사용자가 명시한 9개 시나리오 검증.
모든 consumer (contract.save_*, finalize_run, CLI inspect/validate/finalize,
inspect/validate)가 ResolvedConfig + RunContext 단일 경로로 수렴함을 확인.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


import pcq
from pcq.agent.resolver import resolve_project, resolve_run_context


def _write_cq_yaml(tmp: Path, body: str) -> None:
    """tmp_path에 cq.yaml 작성."""
    (tmp / "cq.yaml").write_text(body)


# ─────────────────────────────────────────────────────────────────────
# DoD #1: env 없이 cq.yaml만 → save_all(finalize=True) → 모든 artifact 생성
# ─────────────────────────────────────────────────────────────────────
def test_dod1_env_없이_cq_yaml만_save_all(tmp_path, monkeypatch):
    """env 없이 cq.yaml.configs.output_dir: out 만 두고 save_all(finalize=True)
    실행 → 모든 artifact가 out/에 생성."""
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    _write_cq_yaml(
        tmp_path,
        """name: dod1
cmd: uv run python train.py
configs:
  output_dir: out
  monitor: eval_loss
  mode: min
metrics:
  - eval_loss
""",
    )
    monkeypatch.chdir(tmp_path)

    history = [{"epoch": 0, "eval_loss": 0.5}, {"epoch": 1, "eval_loss": 0.3}]
    paths = pcq.save_all(history=history, finalize=True)

    out_dir = tmp_path / "out"
    for fname in (
        "config.json",
        "metrics.json",
        "manifest.json",
        "run_summary.json",
        "run_record.json",
    ):
        assert (out_dir / fname).exists(), f"{fname} missing in {out_dir}"
    # paths returned by save_all 도 모두 out_dir 안.
    for key in ("config", "metrics", "manifest", "run_summary", "run_record"):
        assert paths[key].parent == out_dir.resolve()


# ─────────────────────────────────────────────────────────────────────
# DoD #2: nested cwd에서 parent cq.yaml 발견 → 동일 output_dir
# ─────────────────────────────────────────────────────────────────────
def test_dod2_nested_cwd에서_parent_cq_yaml(tmp_path, monkeypatch):
    """nested cwd에서 실행해도 parent cq.yaml을 찾아 동일 output_dir 사용."""
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    _write_cq_yaml(
        tmp_path,
        "name: dod2\ncmd: x\nconfigs:\n  output_dir: out\n",
    )
    nested = tmp_path / "scripts" / "sub" / "dir"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    rc = resolve_project()
    # parent cq.yaml 발견 → output_dir 은 parent의 out
    assert rc.cq_yaml_path == (tmp_path / "cq.yaml").resolve()
    assert rc.output_dir == (tmp_path / "out").resolve()
    # save_metrics 가 정확히 parent의 out 에 쓰는지 확인
    pcq.save_metrics([{"epoch": 0, "loss": 0.1}])
    assert (tmp_path / "out" / "metrics.json").exists()


# ─────────────────────────────────────────────────────────────────────
# DoD #3: pcq finalize runs/exp001 → root cq.yaml metadata 가 RunRecord 에 반영
# ─────────────────────────────────────────────────────────────────────
def test_dod3_finalize_uses_root_cq_yaml_metadata(tmp_path, monkeypatch):
    """pcq finalize runs/exp001가 project root cq.yaml의 name/cmd/inputs/metrics
    를 RunRecord에 반영."""
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    _write_cq_yaml(
        tmp_path,
        """name: dod3-experiment
cmd: uv run python train.py --foo bar
configs:
  output_dir: runs/exp001
  _cmd: uv run python train.py --foo bar
metrics:
  eval_iou: {mode: max}
inputs:
  dataset: {name: dod3-data, uri: "cq://datasets/dod3"}
""",
    )

    # runs/exp001/ 에 manifest+metrics 만 미리 작성.
    out_dir = tmp_path / "runs" / "exp001"
    out_dir.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": []})
    )
    (out_dir / "metrics.json").write_text(
        json.dumps({"history": [{"epoch": 0, "eval_iou": 0.7}]})
    )
    (out_dir / "run_summary.json").write_text(
        json.dumps({
            "schema_version": 1,
            "status": "completed",
            "monitor": {"name": "eval_iou", "mode": "max"},
            "best": {"epoch": 0, "metrics": {"eval_iou": 0.7}},
            "last": {"epoch": 0, "metrics": {"eval_iou": 0.7}},
        })
    )

    # CLI finalize 실행 — chdir/env tmp 트릭 없이도 작동해야 함.
    result = subprocess.run(
        [sys.executable, "-m", "pcq.cli", "finalize", str(out_dir), "--json"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),  # cwd=tmp_path: cq.yaml 보임
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    rr = json.loads((out_dir / "run_record.json").read_text())
    assert rr["run"]["name"] == "dod3-experiment"
    # cmd 는 cfg["_cmd"] 우선, 없으면 cq.yaml.cmd 폴백 — finalize 는 _cmd 사용.
    assert "train.py" in rr["execution"]["cmd"]
    # inputs propagated.
    assert "dataset" in rr["inputs"]
    assert rr["inputs"]["dataset"]["uri"] == "cq://datasets/dod3"
    # metrics declared 발견.
    declared_names = [m["name"] for m in rr["metrics"]["declared"]]
    assert "eval_iou" in declared_names


# ─────────────────────────────────────────────────────────────────────
# DoD #4: inspect 가 custom output_dir 의 manifest/metrics/run_record 감지
# ─────────────────────────────────────────────────────────────────────
def test_dod4_inspect_custom_output_dir(tmp_path, monkeypatch):
    """pcq inspect가 custom output_dir의 manifest/metrics/run_record를 감지."""
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    _write_cq_yaml(
        tmp_path,
        "name: t\ncmd: x\nconfigs:\n  output_dir: my_runs/v1\n",
    )
    out_dir = tmp_path / "my_runs" / "v1"
    out_dir.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "files": []})
    )
    (out_dir / "metrics.json").write_text(
        json.dumps({"history": [{"epoch": 0, "loss": 0.5}]})
    )
    (out_dir / "run_record.json").write_text(json.dumps({"run": {"id": "r1"}}))

    insp = pcq.inspect_project(tmp_path)
    assert insp.outputs is not None
    assert insp.outputs.has_manifest is True
    assert insp.outputs.has_metrics is True
    assert insp.outputs.has_run_record is True
    # output_dir reflects custom location
    assert "my_runs/v1" in (insp.outputs.output_dir or "")


# ─────────────────────────────────────────────────────────────────────
# DoD #5: validate 가 custom output_dir 의 manifest evidence 검증
# ─────────────────────────────────────────────────────────────────────
def test_dod5_validate_custom_output_dir(tmp_path, monkeypatch):
    """pcq validate가 custom output_dir의 manifest evidence를 검증."""
    import hashlib

    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    _write_cq_yaml(
        tmp_path,
        """name: t
cmd: x
configs:
  output_dir: my_runs/v1
metrics:
  - eval_acc
""",
    )
    out_dir = tmp_path / "my_runs" / "v1"
    out_dir.mkdir(parents=True)
    # 진짜 file + sha256 일치하는 manifest entry
    payload = b"fake_model_bytes"
    (out_dir / "model.pkl").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    (out_dir / "manifest.json").write_text(
        json.dumps({
            "schema_version": 2,
            "files": [{"path": "model.pkl", "kind": "model", "sha256": sha}],
        })
    )

    report = pcq.validate_project(tmp_path)
    # manifest_evidence pass 가 포함되었는지 (gate 가 custom output_dir 인식).
    evidence_checks = [c for c in report.checks if c.id == "manifest_evidence"]
    assert evidence_checks, "manifest_evidence gate not run"
    assert evidence_checks[0].status == "pass"


# ─────────────────────────────────────────────────────────────────────
# DoD #6: resolve_project 호출만으로 output dir 생성 안 됨 (read-only)
# ─────────────────────────────────────────────────────────────────────
def test_dod6_resolve_project_no_mkdir(tmp_path, monkeypatch):
    """pcq.resolve_project() 호출만으로 output dir이 생성되지 않음."""
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    _write_cq_yaml(
        tmp_path, "name: t\ncmd: x\nconfigs:\n  output_dir: out\n"
    )
    rc = resolve_project(path=tmp_path)
    assert not (tmp_path / "out").exists(), "resolve_project must NOT mkdir"
    assert rc.output_dir == (tmp_path / "out").resolve()
    # write-side: resolve_run_context 가 mkdir.
    ctx = resolve_run_context(path=tmp_path)
    assert ctx.output_dir.exists()


# ─────────────────────────────────────────────────────────────────────
# DoD #7: run_record metadata 가 cq.yaml 로부터 propagate
# ─────────────────────────────────────────────────────────────────────
def test_dod7_run_record_metadata_from_cq_yaml(tmp_path, monkeypatch):
    """run_record.run.name == cq.yaml.name, execution.cmd == cq.yaml.cmd (fallback)."""
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    _write_cq_yaml(
        tmp_path,
        """name: dod7-run
cmd: uv run python train.py --epochs 3
configs:
  output_dir: out
metrics:
  - eval_loss
""",
    )
    monkeypatch.chdir(tmp_path)

    # save_all → run_record.json 자동 생성 (finalize=True).
    history = [{"epoch": 0, "eval_loss": 0.5}]
    pcq.save_all(history=history, finalize=True)
    rr = json.loads((tmp_path / "out" / "run_record.json").read_text())
    assert rr["run"]["name"] == "dod7-run"
    assert rr["execution"]["cmd"] == "uv run python train.py --epochs 3"


# ─────────────────────────────────────────────────────────────────────
# DoD #8: env CQ_CONFIG_JSON.output_dir overrides cq.yaml.configs.output_dir
# ─────────────────────────────────────────────────────────────────────
def test_dod8_env_overrides_yaml_output_dir(tmp_path, monkeypatch):
    """CQ_CONFIG_JSON.output_dir가 cq.yaml.configs.output_dir를 override."""
    _write_cq_yaml(
        tmp_path, "name: t\ncmd: x\nconfigs:\n  output_dir: a\n"
    )
    env_cfg_path = tmp_path / "env.json"
    env_cfg_path.write_text(json.dumps({"output_dir": str(tmp_path / "b")}))
    monkeypatch.setenv("CQ_CONFIG_JSON", str(env_cfg_path))
    rc = resolve_project(path=tmp_path)
    assert rc.output_dir == (tmp_path / "b").resolve()
    assert rc.cfg["output_dir"] == str(tmp_path / "b")


# ─────────────────────────────────────────────────────────────────────
# DoD #9: 3 modes — env-only / cq.yaml only / both
# ─────────────────────────────────────────────────────────────────────
def test_dod9_three_modes_pass(tmp_path, monkeypatch):
    """env-only / cq.yaml-only / 둘 다 → 모두 정상."""
    # mode_a: env-only
    a_dir = tmp_path / "a"
    a_dir.mkdir()
    monkeypatch.chdir(a_dir)
    cfg_path = a_dir / "cfg.json"
    cfg_path.write_text(
        json.dumps({"output_dir": str(a_dir / "out_a"), "lr": 0.001})
    )
    monkeypatch.setenv("CQ_CONFIG_JSON", str(cfg_path))
    rc = resolve_project()
    assert rc.cfg["lr"] == 0.001
    assert rc.cfg["output_dir"] == str(a_dir / "out_a")

    # mode_b: cq.yaml-only (no env)
    b_dir = tmp_path / "b"
    b_dir.mkdir()
    _write_cq_yaml(
        b_dir, "name: b\ncmd: x\nconfigs:\n  output_dir: out_b\n  lr: 0.01\n"
    )
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    rc = resolve_project(path=b_dir)
    assert rc.cfg["lr"] == 0.01
    assert rc.output_dir == (b_dir / "out_b").resolve()

    # mode_c: cq.yaml + env (env wins on overlap)
    c_dir = tmp_path / "c"
    c_dir.mkdir()
    _write_cq_yaml(
        c_dir,
        "name: c\ncmd: x\nconfigs:\n  output_dir: out_c\n  epochs: 10\n  lr: 0.01\n",
    )
    cfg2 = c_dir / "cfg.json"
    cfg2.write_text(json.dumps({"epochs": 5}))  # epochs override
    monkeypatch.setenv("CQ_CONFIG_JSON", str(cfg2))
    rc = resolve_project(path=c_dir)
    assert rc.cfg["epochs"] == 5  # env wins
    assert rc.cfg["lr"] == 0.01  # cq.yaml 의 비overlap 키 보존
    assert rc.output_dir == (c_dir / "out_c").resolve()


# ─────────────────────────────────────────────────────────────────────
# DoD #10: explicit save_all(output_dir=...) is applied to every artifact
# ─────────────────────────────────────────────────────────────────────
def test_dod10_save_all_explicit_output_dir_single_location(tmp_path, monkeypatch):
    """save_all(output_dir=...) 명시 시 config/metrics/summary/manifest/finalize
    모두 같은 output_dir 에 생성."""
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    monkeypatch.chdir(tmp_path)

    explicit = tmp_path / "explicit_out"
    paths = pcq.save_all(
        history=[{"epoch": 0, "eval_acc": 0.9}],
        finalize=True,
        output_dir=explicit,
    )

    for key in (
        "config",
        "metrics",
        "run_summary",
        "manifest",
        "run_record",
    ):
        assert paths[key].parent == explicit.resolve()
    assert not (tmp_path / "output" / "config.json").exists()
