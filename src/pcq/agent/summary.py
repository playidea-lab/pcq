"""pcq.agent.summary — completed-run summary reader.

summarize_run(output_dir): output 디렉토리 읽어 RunSummary 합성 (training code import X).

v4.0: build_run_summary(Experiment) 제거 — pcq.Experiment 자체가 사라짐.
contract script 는 cq.save_run_summary() 가 직접 호출.
"""
from __future__ import annotations

import json
from pathlib import Path

from pcq.agent.schema import EpochSummary, RunSummary


def _find_best(
    history: list[dict],
    monitor_key: str | None,
    mode: str = "min",
) -> EpochSummary | None:
    if not history or monitor_key is None:
        return None
    eligible = [(i, e) for i, e in enumerate(history) if monitor_key in e]
    if not eligible:
        return None
    if mode == "min":
        idx, best_entry = min(eligible, key=lambda x: x[1][monitor_key])
    else:
        idx, best_entry = max(eligible, key=lambda x: x[1][monitor_key])
    return EpochSummary(
        epoch=int(best_entry.get("epoch", idx)),
        metrics={
            k: float(v)
            for k, v in best_entry.items()
            if k != "epoch" and isinstance(v, (int, float))
        },
        checkpoint="best.ckpt",
    )


def _last_epoch(history: list[dict]) -> EpochSummary | None:
    if not history:
        return None
    e = history[-1]
    return EpochSummary(
        epoch=int(e.get("epoch", len(history) - 1)),
        metrics={
            k: float(v)
            for k, v in e.items()
            if k != "epoch" and isinstance(v, (int, float))
        },
        checkpoint="last.ckpt",
    )


def build_run_summary(
    *,
    history: list[dict],
    monitor_key: str | None,
    monitor_mode: str = "min",
    recipe: str | None = None,
    output_dir: str | Path | None = None,
    provenance: dict | None = None,
    early_stopped_at: int | None = None,
    status: str = "completed",
) -> RunSummary:
    """RunSummary 객체 합성 — contract script 가 직접 호출 가능한 형태.

    v4.0: Experiment 인스턴스 의존을 제거하고 explicit kwargs 만 받는다.
    history, monitor 정보는 contract script 가 자체 관리.
    """
    monitor = (
        {"name": monitor_key, "mode": monitor_mode}
        if monitor_key
        else None
    )

    summary = RunSummary(
        status=status,
        recipe=recipe,
        monitor=monitor,
        target=monitor,   # target == monitor (v1.7 이후 동일)
        best=_find_best(history, monitor_key, monitor_mode),
        last=_last_epoch(history),
        artifacts={
            "config": "config.json",
            "metrics": "metrics.json",
            "manifest": "manifest.json",
        },
        provenance=dict(provenance or {}),
        early_stopped_at=early_stopped_at,
    )
    out_path = Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        if (out_path / "best.ckpt").exists():
            summary.artifacts["best_checkpoint"] = "best.ckpt"
        if (out_path / "last.ckpt").exists():
            summary.artifacts["last_checkpoint"] = "last.ckpt"
        if (out_path / "model.pt").exists():
            summary.artifacts["model"] = "model.pt"
    return summary


def summarize_run(output_dir: str | Path) -> RunSummary:
    """완료된 output 디렉토리 → RunSummary. training code import X.

    Reads:
      - run_summary.json (있으면 그대로 로드)
      - 없으면 metrics.json + config.json + manifest.json 에서 합성
    """
    out = Path(output_dir)
    if not out.exists():
        return RunSummary(
            status="unknown",
            warnings=[f"output_dir not found: {out}"],
        )

    # 1) run_summary.json 이 이미 있으면 그대로 로드
    summary_path = out / "run_summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            return _summary_from_dict(data)
        except (json.JSONDecodeError, OSError):
            # 깨진 파일이면 합성 경로로 fallback
            pass

    # 2) 합성 모드
    summary = RunSummary(
        status="partial",
        warnings=["run_summary.json not present, synthesized"],
    )
    metrics_path = out / "metrics.json"
    config_path = out / "config.json"

    if not metrics_path.exists():
        summary.status = "failed"
        summary.failure = {
            "category": "config_error",
            "message": "metrics.json missing",
            "suggested_fix": "fit() did not complete; check stderr",
        }
        return summary

    try:
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = metrics_data.get("history", [])
    except (json.JSONDecodeError, OSError) as e:
        summary.status = "failed"
        summary.failure = {
            "category": "config_error",
            "message": f"metrics.json unreadable: {e}",
            "suggested_fix": "rerun with verbose",
        }
        return summary

    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            summary.recipe = cfg.get("_recipe")
            summary.provenance = {
                "git_sha": cfg.get("_git_sha"),
                "pcq_version": cfg.get("_pcq_version"),
                "overrides": cfg.get("_overrides", []),
            }
            monitor_name = cfg.get("monitor", "eval_loss")
            mode = cfg.get("mode", "min")
            summary.monitor = {"name": monitor_name, "mode": mode}
            summary.target = summary.monitor
            summary.best = _find_best(history, monitor_name, mode)
        except (json.JSONDecodeError, OSError):
            pass

    summary.last = _last_epoch(history)
    artifacts = {
        "model": "model.pt" if (out / "model.pt").exists() else None,
        "config": "config.json" if config_path.exists() else None,
        "metrics": "metrics.json",
        "manifest": "manifest.json" if (out / "manifest.json").exists() else None,
        "last_checkpoint": "last.ckpt" if (out / "last.ckpt").exists() else None,
        "best_checkpoint": "best.ckpt" if (out / "best.ckpt").exists() else None,
    }
    summary.artifacts = {k: v for k, v in artifacts.items() if v is not None}
    summary.status = "completed" if history else "partial"
    return summary


def _summary_from_dict(data: dict) -> RunSummary:
    """dict (parsed JSON) → RunSummary dataclass. 누락 필드는 None/default."""
    s = RunSummary(
        status=data.get("status", "unknown"),
        recipe=data.get("recipe"),
        monitor=data.get("monitor"),
        target=data.get("target"),
        artifacts=data.get("artifacts", {}),
        provenance=data.get("provenance", {}),
        early_stopped_at=data.get("early_stopped_at"),
        warnings=data.get("warnings", []),
        failure=data.get("failure"),
    )
    if data.get("best"):
        s.best = EpochSummary(**data["best"])
    if data.get("last"):
        s.last = EpochSummary(**data["last"])
    return s
