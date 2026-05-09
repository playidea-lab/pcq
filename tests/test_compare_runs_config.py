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


# v3.0.2 — GT-2 / G9-2 dogfood-driven regression tests.
#
# sequential 비교의 일반 케이스: gen 0 학습 후 cq.yaml 이 gen 1 상태로 덮어
# 쓰여지면, 디스크의 cq.yaml 은 latest 상태이므로 옛 sha 의 configs 를 복원
# 못 한다. 두 path 모두 같은 file 을 read → 같은 dict → diff = empty.
#
# fix: output_dir/config.json 은 매 run 마다 pcq.save_config_snapshot 으로
# 저장된 cfg snapshot 을 가지고 있으므로, cq.yaml read 결과가 empty 또는
# 동일 dict 일 때 config.json 으로 fallback 한다.


def _write_config_json(out_dir: Path, cfg: dict) -> None:
    """save_config_snapshot 이 작성하는 형식을 흉내. provenance _ prefix 포함."""
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dict(cfg)
    # provenance metadata — fallback 시 _ prefix key 는 무시되어야 한다.
    snapshot.setdefault("_git_sha", "deadbeef")
    snapshot.setdefault("_pcq_version", "test")
    (out_dir / "config.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )


def test_compare_runs_uses_config_json_when_cq_yaml_unreadable(tmp_path: Path):
    """v3.0.2: cq.yaml 디스크 latest 일 때 config.json fallback 으로 diff 복원.

    GT-2 / G9-2 핵심 케이스 — cq.yaml.sha256 mismatch 가 인지되지만 두 RunRecord
    가 같은 디스크 file 을 가리키므로 yaml read 결과가 동일. config.json 이
    있으면 그것을 fallback 으로 사용해 effective configs diff 를 복원.
    """
    project = tmp_path / "proj"
    project.mkdir()
    # 디스크의 cq.yaml 은 gen 1 (latest) 상태.
    cq_latest = {
        "name": "demo",
        "cmd": "python train.py",
        "configs": {"a": 1, "b": 99},
    }
    write_yaml(cq_latest, project / "cq.yaml")

    a_out = project / "out_a"
    b_out = project / "out_b"
    # 두 RunRecord 의 cq_yaml_sha256 은 다르지만 (옛 / 새), path 는 같은 cq.yaml.
    _build_minimal_run_record(
        a_out, "cq.yaml", "a" * 64, run_id="run-a",
    )
    _build_minimal_run_record(
        b_out, "cq.yaml", "b" * 64, run_id="run-b",
    )
    # 각 run output 에 config.json snapshot — gen 0 / gen 1 의 effective cfg.
    _write_config_json(a_out, {"a": 1, "b": 2, "output_dir": str(a_out)})
    _write_config_json(b_out, {"a": 1, "b": 99, "output_dir": str(b_out)})

    diff = compare_runs(a_out, b_out)
    keys = {c["key"] for c in diff.config_changes}
    # b 변경 (2 → 99) 은 detect 되어야 한다.
    assert "b" in keys, f"'b' missing: changes={diff.config_changes}"
    # output_dir 도 다르므로 detect.
    assert "output_dir" in keys, f"'output_dir' missing: {diff.config_changes}"
    # 동일 키 (a=1) 는 보이지 않아야.
    assert "a" not in keys
    # provenance metadata (_ prefix) 는 절대 보이지 않아야.
    for k in keys:
        assert not k.startswith("_"), f"provenance key leaked: {k}"
    # decision_facts.config_changed 가 자동으로 True 가 되어야.
    assert diff.decision_facts.get("config_changed") is True


def test_compare_runs_config_changes_for_sequential_dogfood_pattern(
    tmp_path: Path,
):
    """v3.0.2: GT-2 / G9-2 회귀 — sequential gen 0→1 시나리오 5+ 변경 detect.

    실제 dogfood 패턴 그대로 — LogReg → HistGBM 전환에 5 axis 변경 (model,
    output_dir, hgb_max_iter, hgb_max_depth, hgb_lr).
    """
    project = tmp_path / "proj"
    project.mkdir()
    # 디스크의 cq.yaml 은 gen 1 (latest, HistGBM) 상태.
    cq_latest = {
        "name": "tabular",
        "cmd": "uv run python train.py",
        "configs": {
            "model": "hgb",
            "output_dir": "outputs/gen1",
            "hgb_max_iter": 200,
            "hgb_max_depth": 6,
            "hgb_lr": 0.1,
        },
    }
    write_yaml(cq_latest, project / "cq.yaml")

    a_out = project / "outputs" / "gen0"
    b_out = project / "outputs" / "gen1"
    _build_minimal_run_record(
        a_out, "cq.yaml", _sha256_str("gen0"), run_id="gen-0",
    )
    _build_minimal_run_record(
        b_out, "cq.yaml", _sha256_str("gen1"), run_id="gen-1",
    )
    # gen 0 (LogReg) snapshot.
    _write_config_json(a_out, {
        "model": "logreg",
        "output_dir": "outputs/gen0",
        "logreg_C": 1.0,
    })
    # gen 1 (HistGBM) snapshot.
    _write_config_json(b_out, {
        "model": "hgb",
        "output_dir": "outputs/gen1",
        "hgb_max_iter": 200,
        "hgb_max_depth": 6,
        "hgb_lr": 0.1,
    })

    diff = compare_runs(a_out, b_out)
    keys = {c["key"] for c in diff.config_changes}
    # 5 axis 모두 detect — model/output_dir 변경 + hgb_* 추가 + logreg_C 제거.
    expected = {
        "model", "output_dir",
        "hgb_max_iter", "hgb_max_depth", "hgb_lr",
        "logreg_C",
    }
    missing = expected - keys
    assert not missing, f"missing keys {missing}, got {keys}"
    # decision_facts.config_changed True.
    assert diff.decision_facts.get("config_changed") is True


def test_compare_runs_config_json_fallback_skips_provenance_keys(
    tmp_path: Path,
):
    """v3.0.2: config.json fallback 은 _ prefix provenance metadata 를 무시.

    save_config_snapshot 은 _git_sha, _pcq_version, _recipe, _overrides 등을
    추가한다. 이들은 git/version 변화로 매 run 다를 수 있는데, config diff 의
    'effective configs' 측면에서는 노이즈이므로 제외한다.
    """
    project = tmp_path / "proj"
    project.mkdir()
    # cq.yaml 은 디스크에 있지만 두 run 의 sha 는 다름 (다른 file 처럼 행세).
    write_yaml(
        {"name": "x", "cmd": "x", "configs": {}},
        project / "cq.yaml",
    )

    a_out = project / "a"
    b_out = project / "b"
    _build_minimal_run_record(a_out, "cq.yaml", "a" * 64, run_id="run-a")
    _build_minimal_run_record(b_out, "cq.yaml", "b" * 64, run_id="run-b")

    # 같은 effective cfg 지만 provenance 만 다른 두 snapshot.
    _write_config_json(a_out, {"lr": 0.01})
    (a_out / "config.json").write_text(
        json.dumps({
            "lr": 0.01,
            "_git_sha": "AAA",
            "_pcq_version": "3.0.1",
            "_recipe": "baseline",
        }), encoding="utf-8",
    )
    (b_out / "config.json").write_text(
        json.dumps({
            "lr": 0.01,
            "_git_sha": "BBB",
            "_pcq_version": "3.0.2",
            "_recipe": "baseline",
        }), encoding="utf-8",
    )

    diff = compare_runs(a_out, b_out)
    keys = {c["key"] for c in diff.config_changes}
    for k in keys:
        assert not k.startswith("_"), (
            f"provenance key leaked into config_changes: {k} "
            f"(full changes: {diff.config_changes})"
        )
