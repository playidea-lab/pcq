"""compare_runs — RunRecord diff (v1.17)."""
import json
from pathlib import Path

import pcq
from pcq.agent.compare import RunDiff, compare_runs


def _setup_cfg(tmp_path: Path, **extra) -> Path:
    cfg = {"output_dir": str(tmp_path), "seed": 42}
    cfg.update(extra)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def _build_run(
    out_dir: Path,
    monkeypatch,
    history: list[dict],
    monitor: str = "eval_iou",
    mode: str = "max",
    overrides: list | None = None,
    recipe: str | None = None,
) -> Path:
    """헬퍼 — output_dir 에 표준 artifact 들을 작성하고 run_record.json 경로 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_extra: dict = {"monitor": monitor, "mode": mode}
    if recipe:
        cfg_extra["_recipe"] = recipe
    if overrides is not None:
        cfg_extra["_overrides"] = overrides
    cfg_path = _setup_cfg(out_dir, **cfg_extra)
    monkeypatch.setenv("CQ_CONFIG_JSON", str(cfg_path))
    pcq.save_metrics(history)
    pcq.save_run_summary(history=history, status="completed", recipe=recipe)
    pcq.save_manifest()
    return pcq.finalize_run(history=history)


def test_compare_returns_empty_diff_when_records_missing(tmp_path):
    diff = compare_runs(tmp_path / "a", tmp_path / "b")
    assert isinstance(diff, RunDiff)
    assert diff.metric_delta is None
    assert diff.a_run_id == ""


def test_compare_metric_improved_max_mode(tmp_path, monkeypatch):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    _build_run(
        a_dir, monkeypatch,
        [{"epoch": 0, "eval_iou": 0.5}, {"epoch": 1, "eval_iou": 0.6}],
        monitor="eval_iou", mode="max",
    )
    _build_run(
        b_dir, monkeypatch,
        [{"epoch": 0, "eval_iou": 0.7}, {"epoch": 1, "eval_iou": 0.8}],
        monitor="eval_iou", mode="max",
    )
    diff = compare_runs(a_dir, b_dir)
    assert diff.target_metric == "eval_iou"
    # b_best (0.8) - a_best (0.6) = 0.2
    assert diff.metric_delta == 0.2
    assert diff.metric_direction == "improved"
    assert diff.mode == "max"
    assert diff.best == {
        "a": 0.6,
        "b": 0.8,
        "delta": 0.2,
        "direction": "improved",
        "epoch_a": 1,
        "epoch_b": 1,
    }
    assert diff.last == {
        "a": 0.6,
        "b": 0.8,
        "delta": 0.2,
        "direction": "improved",
        "epoch_a": 1,
        "epoch_b": 1,
    }
    assert diff.validation["b"] == "pass"
    assert diff.decision_facts["comparable"] is True
    assert diff.decision_facts["best_improved"] is True
    assert diff.decision_facts["candidate_validated"] is True


def test_compare_metric_regressed_min_mode(tmp_path, monkeypatch):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    _build_run(
        a_dir, monkeypatch,
        [{"epoch": 0, "eval_loss": 0.3}],
        monitor="eval_loss", mode="min",
    )
    _build_run(
        b_dir, monkeypatch,
        [{"epoch": 0, "eval_loss": 0.5}],
        monitor="eval_loss", mode="min",
    )
    diff = compare_runs(a_dir, b_dir)
    # b - a = 0.2 (positive in min mode = regression)
    assert diff.metric_delta == 0.2
    assert diff.metric_direction == "regressed"


def test_compare_metric_tied(tmp_path, monkeypatch):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    _build_run(a_dir, monkeypatch, [{"epoch": 0, "eval_iou": 0.5}], mode="max")
    _build_run(b_dir, monkeypatch, [{"epoch": 0, "eval_iou": 0.5}], mode="max")
    diff = compare_runs(a_dir, b_dir)
    assert diff.metric_delta == 0.0
    assert diff.metric_direction == "tied"


def test_compare_accepts_run_record_json_path_directly(tmp_path, monkeypatch):
    """A/B 인자는 run_record.json 직접 또는 디렉토리 둘 다 가능."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_rr = _build_run(a_dir, monkeypatch, [{"epoch": 0, "eval_iou": 0.5}])
    b_rr = _build_run(b_dir, monkeypatch, [{"epoch": 0, "eval_iou": 0.7}])
    diff = compare_runs(a_rr, b_rr)
    assert diff.metric_delta == round(0.7 - 0.5, 6)


def test_compare_recipe_change_recorded(tmp_path, monkeypatch):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    _build_run(
        a_dir, monkeypatch, [{"epoch": 0, "eval_iou": 0.5}], recipe="recipe-a"
    )
    _build_run(
        b_dir, monkeypatch, [{"epoch": 0, "eval_iou": 0.5}], recipe="recipe-b"
    )
    diff = compare_runs(a_dir, b_dir)
    keys = {c["key"] for c in diff.config_changes}
    assert "recipe" in keys


def test_compare_overrides_change_recorded(tmp_path, monkeypatch):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    _build_run(a_dir, monkeypatch, [{"epoch": 0, "eval_iou": 0.5}], overrides=["lr"])
    _build_run(b_dir, monkeypatch, [{"epoch": 0, "eval_iou": 0.5}], overrides=["lr", "batch_size"])
    diff = compare_runs(a_dir, b_dir)
    keys = {c["key"] for c in diff.config_changes}
    assert "_overrides_keys" in keys


def test_compare_to_dict_skips_empty_lists(tmp_path, monkeypatch):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    _build_run(a_dir, monkeypatch, [{"epoch": 0, "eval_iou": 0.5}])
    _build_run(b_dir, monkeypatch, [{"epoch": 0, "eval_iou": 0.5}])
    diff = compare_runs(a_dir, b_dir)
    out = diff.to_dict()
    assert "schema_version" in out
    # config_changes / atom_changes / input_changes 가 모두 빈 list 면 제거.
    assert "atom_changes" not in out
    assert "input_changes" not in out


# ── v1.18 lineage 인지 ────────────────────────────────────────────────


def _write_minimal_record(
    out_dir: Path,
    run_id: str,
    parent_id: str | None = None,
    parent_path: str | None = None,
) -> None:
    """compare_runs 테스트용 minimal run_record.json (parent 정보만 핵심)."""
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


def test_compare_runs_detects_descendant(tmp_path):
    """b 가 a 의 후손이면 a_is_ancestor_of_b=True."""
    a_dir = tmp_path / "run_a"
    b_dir = tmp_path / "run_b"
    _write_minimal_record(a_dir, "run_a")
    _write_minimal_record(
        b_dir,
        "run_b",
        parent_id="run_a",
        parent_path=str(a_dir),
    )
    diff = compare_runs(a_dir, b_dir)
    assert diff.a_is_ancestor_of_b is True
    assert diff.b_is_ancestor_of_a is False


def test_compare_runs_no_lineage_relationship(tmp_path):
    """관계 없는 두 run 은 양쪽 모두 False."""
    a_dir = tmp_path / "run_a"
    b_dir = tmp_path / "run_b"
    _write_minimal_record(a_dir, "run_a")
    _write_minimal_record(b_dir, "run_b")
    diff = compare_runs(a_dir, b_dir)
    assert diff.a_is_ancestor_of_b is False
    assert diff.b_is_ancestor_of_a is False


def test_compare_runs_to_dict_includes_lineage_flags(tmp_path):
    """lineage flag 는 False 값도 항상 직렬화에 포함 (의미 있는 값)."""
    a_dir = tmp_path / "run_a"
    b_dir = tmp_path / "run_b"
    _write_minimal_record(a_dir, "run_a")
    _write_minimal_record(b_dir, "run_b")
    diff = compare_runs(a_dir, b_dir)
    out = diff.to_dict()
    assert "a_is_ancestor_of_b" in out
    assert "b_is_ancestor_of_a" in out
    assert out["a_is_ancestor_of_b"] is False
    assert out["b_is_ancestor_of_a"] is False


# ── v2.3: trajectory 시그널 ────────────────────────────────────────────


def _make_full_run(
    out_dir: Path,
    run_id: str,
    target: str,
    best_metric: float,
    last_metric: float,
    best_epoch: int,
    history_len: int,
    mode: str = "min",
    parent: str | None = None,
    parent_path: str | None = None,
) -> None:
    """test helper: run_record.json + metrics.json 한꺼번에 작성.

    summary.last 와 metrics.json history 길이를 명시적으로 제어 — best vs last
    trajectory 차이를 시뮬레이션.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rr: dict = {
        "schema_version": 1,
        "run": {"id": run_id, "status": "completed"},
        "execution": {},
        "source": {},
        "environment": {},
        "metrics": {
            "declared": [{"name": target, "mode": mode}],
            "history_path": "metrics.json",
        },
        "artifacts": [],
        "summary": {
            "target_metric": target,
            "best": {"epoch": best_epoch, "metrics": {target: best_metric}},
            "last": {
                "epoch": history_len - 1,
                "metrics": {target: last_metric},
            },
        },
    }
    if parent:
        rr["run"]["parent_run_id"] = parent
    if parent_path:
        rr["run"]["parent_run_path"] = parent_path
    (out_dir / "run_record.json").write_text(json.dumps(rr), encoding="utf-8")
    history = [
        {"epoch": i, target: best_metric if i == best_epoch else last_metric}
        for i in range(history_len)
    ]
    (out_dir / "metrics.json").write_text(
        json.dumps({"history": history}), encoding="utf-8"
    )


def test_compare_runs_last_metric_delta_when_best_tied(tmp_path):
    """v2.3: best 가 tied 여도 last 차이로 trajectory 변화 노출."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    # 둘 다 best=2.5 (epoch 0), 하지만 last 가 다름 (b 가 더 큰 loss = 더 나빠짐)
    _make_full_run(
        a, "run_a", target="eval_loss", best_metric=2.5, last_metric=2.6,
        best_epoch=0, history_len=3, mode="min",
    )
    _make_full_run(
        b, "run_b", target="eval_loss", best_metric=2.5, last_metric=3.5,
        best_epoch=0, history_len=3, mode="min",
    )
    diff = compare_runs(a, b)
    assert diff.metric_direction == "tied"
    assert diff.last_metric_delta is not None
    # b last (3.5) - a last (2.6) = 0.9
    assert diff.last_metric_delta == 0.9
    # min mode + delta 양수 → regressed
    assert diff.last_metric_direction == "regressed"
    assert diff.best_epoch_a == 0
    assert diff.best_epoch_b == 0
    # trajectory note 가 있어야 함
    assert any(
        "trajectory" in n.lower() or "epoch 0" in n.lower() for n in diff.notes
    )


def test_compare_runs_epoch_count(tmp_path):
    """v2.3: epochs_a / epochs_b 추출 (metrics.json history 길이)."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_full_run(
        a, "run_a", target="acc", best_metric=0.7, last_metric=0.65,
        best_epoch=0, history_len=5, mode="max",
    )
    _make_full_run(
        b, "run_b", target="acc", best_metric=0.8, last_metric=0.75,
        best_epoch=2, history_len=10, mode="max",
    )
    diff = compare_runs(a, b)
    assert diff.epochs_a == 5
    assert diff.epochs_b == 10
    assert diff.best_epoch_a == 0
    assert diff.best_epoch_b == 2


def test_compare_runs_both_picked_epoch_zero_note(tmp_path):
    """v2.3: 둘 다 best epoch=0 + tied → 'no learning' note 명시."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_full_run(
        a, "run_a", target="eval_loss", best_metric=1.0, last_metric=1.0,
        best_epoch=0, history_len=3, mode="min",
    )
    _make_full_run(
        b, "run_b", target="eval_loss", best_metric=1.0, last_metric=1.0,
        best_epoch=0, history_len=3, mode="min",
    )
    diff = compare_runs(a, b)
    assert diff.metric_direction == "tied"
    # 'epoch 0' 또는 'no learning' 키워드를 포함하는 note 가 있어야 함
    assert any("epoch 0" in n.lower() for n in diff.notes)


def test_compare_runs_decision_facts_include_status_failure_artifacts_source(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_full_run(
        a, "run_a", target="acc", best_metric=0.7, last_metric=0.65,
        best_epoch=1, history_len=3, mode="max",
    )
    _make_full_run(
        b, "run_b", target="acc", best_metric=0.6, last_metric=0.55,
        best_epoch=1, history_len=3, mode="max",
    )

    b_rr_path = b / "run_record.json"
    b_rr = json.loads(b_rr_path.read_text(encoding="utf-8"))
    b_rr["run"]["status"] = "failed"
    b_rr["validation"] = {"status": "fail", "report_path": "validation_report.json"}
    b_rr["summary"]["failure"] = {
        "error_code": "ERR_RUNTIME",
        "category": "nan_loss",
        "message": "loss became NaN",
    }
    b_rr["artifacts"] = [{"path": "metrics.json", "kind": "metrics"}]
    b_rr["source"] = {"git_sha": "candidate", "dirty": True}
    b_rr_path.write_text(json.dumps(b_rr), encoding="utf-8")

    a_rr_path = a / "run_record.json"
    a_rr = json.loads(a_rr_path.read_text(encoding="utf-8"))
    a_rr["validation"] = {"status": "pass", "report_path": "validation_report.json"}
    a_rr["artifacts"] = [
        {"path": "metrics.json", "kind": "metrics"},
        {"path": "model.pt", "kind": "model"},
    ]
    a_rr["source"] = {"git_sha": "baseline", "dirty": False}
    a_rr_path.write_text(json.dumps(a_rr), encoding="utf-8")

    diff = compare_runs(a, b)
    out = diff.to_dict()

    assert out["metric_direction"] == "regressed"
    assert out["validation"] == {"a": "pass", "b": "fail", "same": False}
    assert out["failure"]["b"]["error_code"] == "ERR_RUNTIME"
    assert out["artifacts"]["a_count"] == 2
    assert out["artifacts"]["b_count"] == 1
    assert out["source"]["same_git_sha"] is False
    assert out["decision_facts"]["candidate_failed"] is True
    assert out["decision_facts"]["candidate_validation_failed"] is True
    assert out["decision_facts"]["artifact_count_changed"] is True
    assert out["decision_facts"]["source_changed"] is True


def test_compare_runs_marks_different_target_metrics_incomparable(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_full_run(
        a, "run_a", target="acc", best_metric=0.7, last_metric=0.65,
        best_epoch=1, history_len=3, mode="max",
    )
    _make_full_run(
        b, "run_b", target="loss", best_metric=0.3, last_metric=0.35,
        best_epoch=1, history_len=3, mode="min",
    )

    diff = compare_runs(a, b)
    out = diff.to_dict()

    assert out["a_target_metric"] == "acc"
    assert out["b_target_metric"] == "loss"
    assert out["metric_direction"] == "incomparable"
    assert out["decision_facts"]["same_target_metric"] is False
    assert out["decision_facts"]["comparable"] is False


def test_compare_runs_loads_failure_from_run_summary_sidecar(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_full_run(
        a, "run_a", target="acc", best_metric=0.7, last_metric=0.65,
        best_epoch=1, history_len=3, mode="max",
    )
    _make_full_run(
        b, "run_b", target="acc", best_metric=0.6, last_metric=0.55,
        best_epoch=1, history_len=3, mode="max",
    )
    (b / "run_summary.json").write_text(
        json.dumps({
            "status": "failed",
            "failure": {
                "error_code": "ERR_RUNTIME",
                "category": "nan_loss",
                "message": "loss became NaN",
            },
        }),
        encoding="utf-8",
    )

    diff = compare_runs(a, b)
    out = diff.to_dict()

    assert out["failure"]["b"]["error_code"] == "ERR_RUNTIME"
    assert out["failure"]["same"] is False
