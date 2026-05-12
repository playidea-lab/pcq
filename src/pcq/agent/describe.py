"""pcq.agent.describe — human/agent friendly RunRecord summary.

agent 가 RunRecord 만으로 다음 실험을 판단할 수 있도록, run_record.json /
metrics.json / run_summary.json 을 합쳐 압축 요약을 만든다.

read-side 도구이며 side-effect 없음.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# RunDescription dict 직렬화 시 빈 값이어도 항상 노출하는 필드.
# (status / epochs_completed / git_sha 등은 0/false/"" 여도 의미가 있음.)
_ALWAYS_KEEP_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "name",
        "status",
        "epochs_completed",
        "git_sha",
        "dirty",
        "python",
        "platform",
        "validation_status",
        "partial",
    }
)


@dataclass
class RunDescription:
    """compact RunRecord summary.

    JSON 직렬화 시 빈 값(None / "" / [] / {}) 은 _ALWAYS_KEEP_KEYS 에 속하지
    않으면 제거하여 agent 컨텍스트를 깨끗하게 유지한다.
    """

    schema_version: int = 1
    run_id: str = ""
    name: str = ""
    status: str = "unknown"
    output_dir: str = ""
    cmd: str = ""
    target_metric: str | None = None
    mode: str | None = None
    best: dict | None = None
    best_value: float | None = None
    best_epoch: int | None = None
    last: dict | None = None
    last_value: float | None = None
    last_epoch: int | None = None
    epochs_completed: int = 0
    duration_seconds: float | None = None
    partial: bool = False
    last_updated_at: str | None = None
    parent_run_id: str | None = None
    parent_run_path: str | None = None
    git_sha: str = ""
    dirty: bool = False
    python: str = ""
    platform: str = ""
    inputs_summary: list[str] = field(default_factory=list)
    metrics_declared: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    artifacts_summary: dict[str, int] = field(default_factory=dict)
    plan_id: str | None = None
    recipe: str | None = None
    validation_status: str = "unknown"
    validation_report_path: str | None = None
    reproducibility_evidence: dict = field(default_factory=dict)
    decision_facts: dict = field(default_factory=dict)
    failure: dict | None = None
    # attribution — v3.0: 중첩 객체 + 플랫 표면 (에이전트 쿼리 편의)
    attribution: dict | None = None
    attribution_author_kind: str | None = None
    attribution_committer_kind: str | None = None
    attribution_operator: str | None = None
    attribution_session_id: str | None = None
    # worker_spec — T-WSPEC-5: 중첩 객체 + 플랫 표면 (에이전트 쿼리 편의)
    worker_spec: dict | None = None
    worker_spec_cpu_model: str | None = None
    worker_spec_memory_gb: float | None = None
    worker_spec_accelerator_kind: str | None = None
    worker_spec_gpu_model_0: str | None = None
    # fingerprint — T-WFP-5: 중첩 객체 + 플랫 표면 (에이전트 쿼리 편의)
    fingerprint: dict | None = None
    fingerprint_modality: str | None = None
    fingerprint_task_kind: str | None = None
    fingerprint_n_samples: int | None = None
    fingerprint_size_class: str | None = None

    def to_dict(self) -> dict:
        out: dict = {}
        for k, v in self.__dict__.items():
            empty = v in (None, "", [], {})
            if not empty or k in _ALWAYS_KEEP_KEYS:
                out[k] = v
        return out


def _parse_iso(ts: str | None) -> datetime | None:
    """ISO-8601 (Z suffix 포함) → datetime. 실패 시 None."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _drop_empty(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            compact = _drop_empty(v)
            if compact not in (None, "", [], {}):
                out[k] = compact
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            compact = _drop_empty(item)
            if compact not in (None, "", [], {}):
                out.append(compact)
        return out
    return value


def _metric_value(summary: dict | None, target: str | None) -> float | None:
    if not summary or not target:
        return None
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(target)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _compact_epoch_summary(summary: dict | None, target: str | None) -> dict | None:
    if not isinstance(summary, dict) or not summary:
        return None
    out: dict = {}
    if summary.get("epoch") is not None:
        out["epoch"] = summary.get("epoch")
    value = _metric_value(summary, target)
    if value is not None:
        out["value"] = value
    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and metrics:
        out["metrics"] = dict(metrics)
    if summary.get("checkpoint"):
        out["checkpoint"] = summary.get("checkpoint")
    return out or None


def _mode_from_metrics(metrics: dict, target: str | None) -> str | None:
    if not target:
        return None
    declared = metrics.get("declared") or []
    if not isinstance(declared, list):
        return None
    for item in declared:
        if not isinstance(item, dict):
            continue
        if item.get("name") == target and item.get("mode") in ("min", "max"):
            return str(item["mode"])
    return None


def _reproducibility_evidence(rr: dict) -> dict:
    source = rr.get("source") or {}
    environment = rr.get("environment") or {}
    config = rr.get("config") or {}
    input_summary = rr.get("input_summary") or {}
    metrics = rr.get("metrics") or {}

    evidence = {
        "source": {
            "git_sha": source.get("git_sha"),
            "dirty": bool(source.get("dirty", False)),
            "changed_files_count": len(source.get("changed_files") or []),
            "patch_sha256": source.get("patch_sha256"),
            "cq_yaml_path": source.get("cq_yaml_path"),
            "cq_yaml_sha256": source.get("cq_yaml_sha256"),
        },
        "environment": {
            "python": environment.get("python"),
            "platform": environment.get("platform"),
            "pcq_version": environment.get("pcq_version"),
            "torch_version": environment.get("torch_version"),
            "device": environment.get("device"),
            "cuda_available": environment.get("cuda_available"),
            "gpu_count": environment.get("gpu_count"),
            "lockfile": environment.get("lockfile"),
            "lockfile_sha256": environment.get("lockfile_sha256"),
        },
        "config": {
            "cq_yaml_path": config.get("cq_yaml_path"),
            "cq_yaml_sha256": config.get("cq_yaml_sha256"),
            "config_json_path": config.get("config_json_path"),
            "config_json_sha256": config.get("config_json_sha256"),
            "seed": config.get("seed"),
            "strictness": config.get("strictness"),
            "output_dir": config.get("output_dir"),
        },
        "inputs": input_summary,
        "metrics": {
            "declared": metrics.get("declared") or [],
            "history_path": metrics.get("history_path"),
        },
    }
    return _drop_empty(evidence)


def _decision_facts(desc: RunDescription) -> dict:
    input_count = 0
    inputs = desc.reproducibility_evidence.get("inputs")
    if isinstance(inputs, dict):
        input_count = int(inputs.get("count") or 0)
    env = desc.reproducibility_evidence.get("environment") or {}
    cfg = desc.reproducibility_evidence.get("config") or {}
    return {
        "run_completed": desc.status == "completed",
        "run_failed": desc.status == "failed",
        "run_partial": bool(desc.partial),
        "validation_passed": desc.validation_status == "pass",
        "validation_failed": desc.validation_status == "fail",
        "has_failure": bool(desc.failure),
        "has_target_metric": bool(desc.target_metric),
        "has_best": desc.best_value is not None,
        "has_last": desc.last_value is not None,
        "has_parent": bool(desc.parent_run_id or desc.parent_run_path),
        "artifact_count": len(desc.artifacts),
        "metric_count": len(desc.metrics_declared),
        "input_count": input_count,
        "dirty_source": bool(desc.dirty),
        "has_lockfile": bool(env.get("lockfile_sha256")),
        "has_cq_yaml_hash": bool(cfg.get("cq_yaml_sha256")),
    }


def describe_run(output_dir: str | Path) -> RunDescription:
    """RunRecord + metrics + summary 를 합쳐 compact summary 반환.

    output_dir 은 run_record.json 이 있는 디렉토리.
    파일이 없으면 status="no_record", 깨졌으면 status="corrupted".
    """
    out = Path(output_dir).resolve()
    desc = RunDescription(output_dir=str(out))

    rr_path = out / "run_record.json"
    if not rr_path.exists():
        desc.status = "no_record"
        return desc

    try:
        rr = json.loads(rr_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        desc.status = "corrupted"
        return desc

    desc.schema_version = rr.get("schema_version", 1)

    # run section
    run = rr.get("run") or {}
    desc.run_id = str(run.get("id", ""))
    desc.name = str(run.get("name", ""))
    desc.status = str(run.get("status", "unknown"))
    desc.partial = bool(run.get("partial", False))
    desc.last_updated_at = run.get("last_updated_at")
    desc.parent_run_id = run.get("parent_run_id")
    desc.parent_run_path = run.get("parent_run_path")

    # execution
    execution = rr.get("execution") or {}
    desc.cmd = str(execution.get("cmd", ""))

    # source / environment
    src = rr.get("source") or {}
    desc.git_sha = str(src.get("git_sha", ""))
    desc.dirty = bool(src.get("dirty", False))
    env = rr.get("environment") or {}
    desc.python = str(env.get("python", ""))
    desc.platform = str(env.get("platform", ""))

    # duration
    t0 = _parse_iso(run.get("started_at"))
    t1 = _parse_iso(run.get("finished_at"))
    if t0 and t1:
        desc.duration_seconds = (t1 - t0).total_seconds()

    # summary section — target_metric + best/last
    summary = rr.get("summary") or {}
    desc.target_metric = summary.get("target_metric")
    metrics = rr.get("metrics") or {}
    desc.metrics_declared = list(metrics.get("declared") or [])
    desc.mode = _mode_from_metrics(metrics, desc.target_metric)
    best = summary.get("best") or {}
    last = summary.get("last") or {}
    if best:
        desc.best = _compact_epoch_summary(best, desc.target_metric)
        desc.best_epoch = best.get("epoch")
        desc.best_value = _metric_value(best, desc.target_metric)
    if last:
        desc.last = _compact_epoch_summary(last, desc.target_metric)
        desc.last_epoch = last.get("epoch")
        desc.last_value = _metric_value(last, desc.target_metric)

    # epochs from metrics.json (history len)
    metrics_path = out / "metrics.json"
    if metrics_path.exists():
        try:
            mdata = json.loads(metrics_path.read_text(encoding="utf-8"))
            history = mdata.get("history", []) or []
            desc.epochs_completed = len(history)
        except json.JSONDecodeError:
            pass

    # inputs summary
    inputs = rr.get("inputs") or {}
    if isinstance(inputs, dict):
        for input_key, meta in inputs.items():
            if isinstance(meta, dict):
                name = meta.get("name", "")
                version = meta.get("version", "")
                tag = f"{input_key}:{name}" + (f"@{version}" if version else "")
            else:
                tag = f"{input_key}:{meta}"
            desc.inputs_summary.append(tag)

    # artifacts summary — kind 별 count
    artifacts = rr.get("artifacts") or []
    for entry in artifacts:
        if isinstance(entry, dict):
            kind = entry.get("kind") or "other"
            desc.artifacts.append(dict(entry))
        else:
            kind = "other"
        desc.artifacts_summary[kind] = desc.artifacts_summary.get(kind, 0) + 1

    # agent
    agent = rr.get("agent") or {}
    desc.plan_id = agent.get("plan_id")
    desc.recipe = agent.get("recipe")

    # validation
    validation = rr.get("validation") or {}
    desc.validation_status = str(validation.get("status", "unknown"))
    desc.validation_report_path = validation.get("report_path")
    desc.reproducibility_evidence = _reproducibility_evidence(rr)

    # failure (run_summary.json 에서)
    rs_path = out / "run_summary.json"
    rs = _read_json(rs_path)
    if rs:
        if not desc.mode:
            monitor = rs.get("monitor") or {}
            if monitor.get("mode") in ("min", "max"):
                desc.mode = monitor.get("mode")
        failure = rs.get("failure")
        if failure:
            desc.failure = failure

    # attribution — run_record 의 중첩 객체를 그대로 보존하고 플랫 표면도 노출.
    raw_attribution = rr.get("attribution")
    if isinstance(raw_attribution, dict):
        desc.attribution = raw_attribution
        author = raw_attribution.get("author") or {}
        committer = raw_attribution.get("committer") or {}
        desc.attribution_author_kind = author.get("kind") or None
        desc.attribution_committer_kind = committer.get("kind") or None
        desc.attribution_operator = raw_attribution.get("operator") or None
        desc.attribution_session_id = raw_attribution.get("session_id") or None

    # worker_spec — T-WSPEC-5: run_record 의 중첩 객체를 그대로 보존하고 플랫 표면도 노출.
    raw_worker_spec = rr.get("worker_spec")
    if isinstance(raw_worker_spec, dict):
        desc.worker_spec = raw_worker_spec
        cpu = raw_worker_spec.get("cpu") or {}
        memory = raw_worker_spec.get("memory") or {}
        accelerator = raw_worker_spec.get("accelerator") or {}
        gpus = accelerator.get("gpus") or []
        desc.worker_spec_cpu_model = cpu.get("model") or None
        total_gb = memory.get("total_gb")
        desc.worker_spec_memory_gb = float(total_gb) if isinstance(total_gb, (int, float)) else None
        desc.worker_spec_accelerator_kind = accelerator.get("kind") or None
        gpu_0 = gpus[0] if gpus else {}
        desc.worker_spec_gpu_model_0 = gpu_0.get("model") or None if isinstance(gpu_0, dict) else None

    # fingerprint — T-WFP-5: run_record 의 중첩 객체를 그대로 보존하고 플랫 표면도 노출.
    raw_fingerprint = rr.get("fingerprint")
    if isinstance(raw_fingerprint, dict):
        desc.fingerprint = raw_fingerprint
        desc.fingerprint_modality = raw_fingerprint.get("modality") or None
        desc.fingerprint_task_kind = raw_fingerprint.get("task_kind") or None
        n_samples = raw_fingerprint.get("n_samples")
        desc.fingerprint_n_samples = int(n_samples) if isinstance(n_samples, (int, float)) else None
        desc.fingerprint_size_class = raw_fingerprint.get("size_class") or None

    desc.decision_facts = _decision_facts(desc)

    return desc
