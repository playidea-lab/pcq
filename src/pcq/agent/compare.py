"""pcq.agent.compare — RunRecord diff for agent decisions.

두 run 의 RunRecord 를 비교하여 metric_delta, config_changes, input_changes
등 agent 가 즉시 활용 가능한 형태로 반환한다. read-side 도구.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RunDiff:
    """두 RunRecord 의 agent-decision-ready diff."""

    schema_version: int = 1
    a_run_id: str = ""
    b_run_id: str = ""
    target_metric: str | None = None
    a_target_metric: str | None = None
    b_target_metric: str | None = None
    mode: str | None = None
    best: dict = field(default_factory=dict)
    metric_delta: float | None = None         # best b - best a
    metric_direction: str | None = None       # improved / regressed / tied / incomparable
    last: dict = field(default_factory=dict)
    # v2.3: trajectory 시그널 — best 만으로는 hyperparameter 효과를 못 보는 경우
    # (둘 다 epoch 0 이 best 인 경우) 가 있어 last epoch 비교를 추가.
    last_metric_delta: float | None = None    # last_b - last_a
    last_metric_direction: str | None = None  # improved/regressed/tied/incomparable
    epochs_a: int | None = None               # metrics.json history 길이
    epochs_b: int | None = None
    best_epoch_a: int | None = None           # summary.best.epoch
    best_epoch_b: int | None = None
    config_changes: list[dict] = field(default_factory=list)
    atom_changes: list[dict] = field(default_factory=list)
    input_changes: list[dict] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    failure: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    duration_a: float | None = None
    duration_b: float | None = None
    a_status: str = ""
    b_status: str = ""
    # v1.18: lineage 인지 — A/B 가 ancestor/descendant 관계인지.
    a_is_ancestor_of_b: bool = False
    b_is_ancestor_of_a: bool = False
    # v2.3: agent 가 즉시 활용 가능한 explanatory notes
    # (예: "best is tied, but last epoch differs — trajectory shifted").
    notes: list[str] = field(default_factory=list)
    decision_facts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        # 빈 값 제거. 단, schema_version 은 항상 유지. lineage bool 필드는
        # False 가 의미 있으므로 항상 포함.
        out: dict = {"schema_version": self.schema_version}
        for k, v in self.__dict__.items():
            if k == "schema_version":
                continue
            if k in ("a_is_ancestor_of_b", "b_is_ancestor_of_a"):
                out[k] = v
                continue
            if v in (None, "", [], {}):
                continue
            out[k] = v
        return out


def _load_record(path: str | Path) -> dict | None:
    """run_record.json 직접 파일 또는 output 디렉토리 둘 다 지원."""
    p = Path(path).resolve()
    target: Path | None = None
    if p.is_file():
        target = p
    elif p.is_dir():
        candidate = p / "run_record.json"
        if candidate.exists():
            target = candidate
    if target is None:
        return None
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None

    # finalize_run 이전/이후 경로 차이로 failure summary 가 run_summary.json 에만
    # 남는 경우가 있다. compare 는 read-side 이므로 sidecar 를 읽어 보강만 한다.
    run_summary_path = target.parent / "run_summary.json"
    if run_summary_path.exists():
        try:
            run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            run_summary = None
        if isinstance(run_summary, dict):
            failure = run_summary.get("failure")
            if isinstance(failure, dict) and failure:
                summary = record.setdefault("summary", {})
                if isinstance(summary, dict) and "failure" not in summary:
                    summary["failure"] = failure
    return record


def _parse_iso(ts: str | None) -> datetime | None:
    """ISO-8601 → datetime. 실패 시 None."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _duration(run_section: dict) -> float | None:
    """run.started_at / run.finished_at → seconds."""
    t0 = _parse_iso(run_section.get("started_at"))
    t1 = _parse_iso(run_section.get("finished_at"))
    if t0 and t1:
        return (t1 - t0).total_seconds()
    return None


def _metric_mode(record: dict, target: str) -> str:
    """metrics.declared 에서 target 의 mode 추출. 없으면 'min' default."""
    declared = (record.get("metrics") or {}).get("declared") or []
    for entry in declared:
        if isinstance(entry, dict) and entry.get("name") == target:
            mode = entry.get("mode")
            if mode in ("min", "max"):
                return mode
    return "min"


def _classify_delta(delta: float, mode: str) -> str:
    """delta + mode → improved/regressed/tied direction string.

    v2.3: best 와 last 두 곳에서 동일 로직 사용 — DRY.
    """
    if delta == 0:
        return "tied"
    if (mode == "max" and delta > 0) or (mode == "min" and delta < 0):
        return "improved"
    return "regressed"


def _metric_value(record: dict, which: str, target: str | None) -> float | None:
    if not target:
        return None
    summary = record.get("summary") or {}
    metrics = ((summary.get(which) or {}).get("metrics") or {})
    value = metrics.get(target)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _summary_epoch(record: dict, which: str) -> int | None:
    value = ((record.get("summary") or {}).get(which) or {}).get("epoch")
    return value if isinstance(value, int) else None


def _direction_for_values(
    a_value: float | None, b_value: float | None, mode: str
) -> tuple[float | None, str]:
    if a_value is None or b_value is None:
        return None, "incomparable"
    delta = round(float(b_value) - float(a_value), 6)
    return delta, _classify_delta(delta, mode)


def _validation_status(record: dict) -> str:
    return str((record.get("validation") or {}).get("status", "unknown"))


def _failure(record: dict) -> dict | None:
    failure = (record.get("summary") or {}).get("failure")
    return failure if isinstance(failure, dict) and failure else None


def _artifact_summary(record: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    artifacts = record.get("artifacts") or []
    if not isinstance(artifacts, list):
        return out
    for entry in artifacts:
        kind = "other"
        if isinstance(entry, dict):
            kind = str(entry.get("kind") or "other")
        out[kind] = out.get(kind, 0) + 1
    return out


def _source_summary(record: dict) -> dict:
    source = record.get("source") or {}
    if not isinstance(source, dict):
        return {}
    out: dict = {}
    if source.get("git_sha"):
        out["git_sha"] = source.get("git_sha")
    if "dirty" in source:
        out["dirty"] = bool(source.get("dirty", False))
    changed_files = source.get("changed_files")
    if changed_files:
        out["changed_files_count"] = len(changed_files)
    if source.get("cq_yaml_sha256"):
        out["cq_yaml_sha256"] = source.get("cq_yaml_sha256")
    return out


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


def _populate_pair_facts(
    diff: RunDiff,
    *,
    a: dict,
    b: dict,
    mode: str,
) -> None:
    a_best = _metric_value(a, "best", diff.target_metric)
    b_best = _metric_value(b, "best", diff.target_metric)
    best_delta, best_direction = _direction_for_values(a_best, b_best, mode)
    diff.best = {
        "a": a_best,
        "b": b_best,
        "delta": best_delta,
        "direction": best_direction,
        "epoch_a": _summary_epoch(a, "best"),
        "epoch_b": _summary_epoch(b, "best"),
    }
    diff.best = _drop_empty(diff.best)

    a_last = _metric_value(a, "last", diff.target_metric)
    b_last = _metric_value(b, "last", diff.target_metric)
    last_delta, last_direction = _direction_for_values(a_last, b_last, mode)
    diff.last = {
        "a": a_last,
        "b": b_last,
        "delta": last_delta,
        "direction": last_direction,
        "epoch_a": _summary_epoch(a, "last"),
        "epoch_b": _summary_epoch(b, "last"),
    }
    diff.last = _drop_empty(diff.last)

    diff.validation = {
        "a": _validation_status(a),
        "b": _validation_status(b),
        "same": _validation_status(a) == _validation_status(b),
    }
    failure_a = _failure(a)
    failure_b = _failure(b)
    diff.failure = _drop_empty({
        "a": failure_a,
        "b": failure_b,
        "same": failure_a == failure_b,
    })
    diff.artifacts = {
        "a_count": len(a.get("artifacts") or []),
        "b_count": len(b.get("artifacts") or []),
        "a_summary": _artifact_summary(a),
        "b_summary": _artifact_summary(b),
    }
    source_a = _source_summary(a)
    source_b = _source_summary(b)
    a_git_sha = source_a.get("git_sha")
    b_git_sha = source_b.get("git_sha")
    a_cq_yaml_sha = source_a.get("cq_yaml_sha256")
    b_cq_yaml_sha = source_b.get("cq_yaml_sha256")
    same_git_sha = a_git_sha == b_git_sha if (a_git_sha or b_git_sha) else None
    same_cq_yaml_sha256 = (
        a_cq_yaml_sha == b_cq_yaml_sha if (a_cq_yaml_sha or b_cq_yaml_sha) else None
    )
    dirty_changed = (
        source_a.get("dirty") != source_b.get("dirty")
        if ("dirty" in source_a or "dirty" in source_b)
        else None
    )
    diff.source = _drop_empty({
        "a": source_a,
        "b": source_b,
        "same_git_sha": same_git_sha,
        "same_cq_yaml_sha256": same_cq_yaml_sha256,
        "dirty_changed": dirty_changed,
    })


def _decision_facts(diff: RunDiff) -> dict:
    same_target_metric = bool(
        diff.a_target_metric
        and diff.b_target_metric
        and diff.a_target_metric == diff.b_target_metric
    )
    return {
        "comparable": same_target_metric
        and diff.metric_direction != "incomparable",
        "same_target_metric": same_target_metric,
        "best_improved": diff.metric_direction == "improved",
        "best_regressed": diff.metric_direction == "regressed",
        "best_tied": diff.metric_direction == "tied",
        "last_improved": diff.last_metric_direction == "improved",
        "last_regressed": diff.last_metric_direction == "regressed",
        "last_tied": diff.last_metric_direction == "tied",
        "candidate_completed": diff.b_status == "completed",
        "candidate_failed": diff.b_status == "failed",
        "candidate_validated": diff.validation.get("b") == "pass",
        "candidate_validation_failed": diff.validation.get("b") == "fail",
        "config_changed": bool(diff.config_changes),
        "input_changed": bool(diff.input_changes),
        "artifact_count_changed": (
            diff.artifacts.get("a_count") != diff.artifacts.get("b_count")
        ),
        "source_changed": diff.source.get("same_git_sha") is False,
        "same_cq_yaml": bool(diff.source.get("same_cq_yaml_sha256", False)),
        "has_lineage_relation": bool(
            diff.a_is_ancestor_of_b or diff.b_is_ancestor_of_a
        ),
    }


def _count_history_epochs(path: str | Path) -> int | None:
    """metrics.json 의 history 길이. 없거나 파싱 실패 시 None.

    path 가 run_record.json 파일이면 같은 디렉토리의 metrics.json 을,
    디렉토리이면 그 안의 metrics.json 을 읽는다.
    """
    p = Path(path).resolve()
    if p.is_file():
        candidate = p.parent / "metrics.json"
    elif p.is_dir():
        candidate = p / "metrics.json"
    else:
        return None
    if not candidate.exists():
        return None
    try:
        m = json.loads(candidate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    history = m.get("history") if isinstance(m, dict) else None
    return len(history) if isinstance(history, list) else None


def compare_runs(a_path: str | Path, b_path: str | Path) -> RunDiff:
    """두 RunRecord 를 비교해 agent-decision-ready diff 반환."""
    diff = RunDiff()
    a = _load_record(a_path)
    b = _load_record(b_path)
    if a is None or b is None:
        return diff

    # IDs / status
    a_run = a.get("run") or {}
    b_run = b.get("run") or {}
    diff.a_run_id = str(a_run.get("id", ""))
    diff.b_run_id = str(b_run.get("id", ""))
    diff.a_status = str(a_run.get("status", ""))
    diff.b_status = str(b_run.get("status", ""))

    # target metric — A 우선, 없으면 B
    a_summary = a.get("summary") or {}
    b_summary = b.get("summary") or {}
    a_target = a_summary.get("target_metric")
    b_target = b_summary.get("target_metric")
    diff.a_target_metric = str(a_target) if a_target else None
    diff.b_target_metric = str(b_target) if b_target else None
    target = a_target or b_target
    diff.target_metric = str(target) if target else None

    # metric_delta + direction
    mode = _metric_mode(a, target) if target else "min"
    diff.mode = mode if target else None
    _populate_pair_facts(diff, a=a, b=b, mode=mode)
    if target:
        a_best = ((a_summary.get("best") or {}).get("metrics") or {}).get(target)
        b_best = ((b_summary.get("best") or {}).get("metrics") or {}).get(target)
        if isinstance(a_best, (int, float)) and isinstance(b_best, (int, float)):
            delta = round(float(b_best) - float(a_best), 6)
            diff.metric_delta = delta
            diff.metric_direction = _classify_delta(delta, mode)
        else:
            diff.metric_direction = "incomparable"

        # v2.3: last metric trajectory — best 가 tied 여도 last 에서 차이가
        # 보이면 hyperparameter 효과가 trajectory 에 영향을 줬다는 신호.
        a_last = ((a_summary.get("last") or {}).get("metrics") or {}).get(target)
        b_last = ((b_summary.get("last") or {}).get("metrics") or {}).get(target)
        if isinstance(a_last, (int, float)) and isinstance(b_last, (int, float)):
            last_delta = round(float(b_last) - float(a_last), 6)
            diff.last_metric_delta = last_delta
            diff.last_metric_direction = _classify_delta(last_delta, mode)

        # v2.3: best epoch 추출 (정수 캐스팅 — 직렬화 호환).
        a_best_epoch = (a_summary.get("best") or {}).get("epoch")
        b_best_epoch = (b_summary.get("best") or {}).get("epoch")
        if isinstance(a_best_epoch, int):
            diff.best_epoch_a = a_best_epoch
        if isinstance(b_best_epoch, int):
            diff.best_epoch_b = b_best_epoch

    # v2.3: epoch count from metrics.json (target 없어도 항상 추출 시도).
    diff.epochs_a = _count_history_epochs(a_path)
    diff.epochs_b = _count_history_epochs(b_path)

    # config_changes — agent.recipe / agent.overrides 변경
    a_agent = a.get("agent") or {}
    b_agent = b.get("agent") or {}
    a_overrides_list = a_agent.get("overrides") or []
    b_overrides_list = b_agent.get("overrides") or []
    if sorted(a_overrides_list) != sorted(b_overrides_list):
        diff.config_changes.append(
            {
                "key": "_overrides_keys",
                "a": sorted(a_overrides_list),
                "b": sorted(b_overrides_list),
            }
        )
    if a_agent.get("recipe") != b_agent.get("recipe"):
        diff.config_changes.append(
            {
                "key": "recipe",
                "a": a_agent.get("recipe"),
                "b": b_agent.get("recipe"),
            }
        )

    # v2.12: cq.yaml.configs 실제 dict diff (G1-4 fix).
    # RunRecord.config.cq_yaml_path 또는 source.cq_yaml_path 를 통해 cq.yaml 을
    # read 하고 두 configs 를 비교. agent.overrides 만으로는 set_config 로 변경된
    # axis 가 안 잡혀서 dogfood gen 0→1 에서 5 axis 변경에도 config_changes=[] 였음.
    yaml_changes = _diff_cq_yaml_configs(a, a_path, b, b_path)
    diff.config_changes.extend(yaml_changes)

    # input_changes
    a_inputs = a.get("inputs") or {}
    b_inputs = b.get("inputs") or {}
    keys = sorted(set(a_inputs) | set(b_inputs))
    for k in keys:
        ia, ib = a_inputs.get(k), b_inputs.get(k)
        if ia != ib:
            diff.input_changes.append({"input": k, "a": ia, "b": ib})

    # duration
    diff.duration_a = _duration(a_run)
    diff.duration_b = _duration(b_run)

    # v2.3: explanatory notes — agent 가 즉시 활용할 수 있는 trajectory 시그널.
    _populate_notes(diff)

    # v1.18 lineage: ancestor 관계. compare_runs 가 lineage chain 을 따라가
    # b_run_id 가 a 의 chain 에 등장하면 b 가 a 의 ancestor (즉 a 는 b 의 후손).
    # path 가 file/dir 두 형태 모두 허용되므로 그대로 전달.
    try:
        from pcq.agent.lineage import is_descendant_of

        if diff.b_run_id:
            diff.a_is_ancestor_of_b = is_descendant_of(b_path, diff.a_run_id)
        if diff.a_run_id:
            diff.b_is_ancestor_of_a = is_descendant_of(a_path, diff.b_run_id)
    except Exception:
        # lineage 가 임의 path 에 대해 실패해도 compare_runs 는 완성되어야 함.
        pass

    diff.decision_facts = _decision_facts(diff)

    return diff


def _resolve_cq_yaml_for_record(
    record: dict, record_path: str | Path
) -> Path | None:
    """RunRecord 에서 cq.yaml 위치를 best-effort 로 복구.

    우선순위:
      1. record.config.cq_yaml_path
      2. record.source.cq_yaml_path
    값은 RunRecord 작성 당시의 project_root 기준 relative. 복구는 다음 후보 dir
    들을 차례로 시도:
      a. record_path 가 dir 이면 그 자체 / parent / parent.parent
      b. record_path 가 file 이면 그 parent / parent.parent

    매칭되는 first existing path 반환. 없으면 None (graceful fallback).
    """
    cfg_section = record.get("config") if isinstance(record.get("config"), dict) else {}
    src_section = record.get("source") if isinstance(record.get("source"), dict) else {}
    rel = cfg_section.get("cq_yaml_path") or src_section.get("cq_yaml_path")
    if not rel:
        return None
    rel_str = str(rel)

    p = Path(record_path).resolve()
    candidate_roots: list[Path] = []
    if p.is_dir():
        candidate_roots.append(p)
        candidate_roots.append(p.parent)
        candidate_roots.append(p.parent.parent)
    elif p.is_file():
        candidate_roots.append(p.parent)
        candidate_roots.append(p.parent.parent)
    else:
        # 경로 자체가 없어도 parent / parent.parent 시도 (test fixture 대응).
        candidate_roots.append(p.parent)
        candidate_roots.append(p.parent.parent)

    rel_path = Path(rel_str)
    if rel_path.is_absolute() and rel_path.exists():
        return rel_path

    for root in candidate_roots:
        cand = (root / rel_path).resolve()
        if cand.exists():
            return cand
    return None


def _read_cq_yaml_configs(path: Path) -> dict | None:
    """cq.yaml.configs 만 read. 실패 시 None."""
    try:
        from pcq.agent.yaml_io import read_yaml

        data = read_yaml(path)
    except Exception:  # noqa: BLE001 — yaml read 실패는 graceful skip.
        return None
    if not isinstance(data, dict):
        return None
    cfg = data.get("configs")
    if not isinstance(cfg, dict):
        return {}
    return dict(cfg)


def _read_run_config_json(run_path: str | Path) -> dict | None:
    """v3.0.2: output_dir/config.json 에서 effective configs read.

    save_config_snapshot 가 매 run 마다 작성하는 cfg snapshot. _ prefix 의
    provenance metadata (_git_sha, _pcq_version, _recipe, _overrides) 는
    제외하여 cq.yaml.configs 와 같은 abstraction layer 의 dict 로 정규화.

    sequential gen 비교에서 디스크의 cq.yaml 이 latest 로 덮어 써져 옛 run 의
    configs 를 복원하지 못할 때 사용한다 (G9-2 / GT-2 dogfood).

    path 가 file 이면 같은 dir, dir 이면 그 자체에서 config.json 을 찾는다.
    실패 시 None — 호출 측은 graceful skip.
    """
    p = Path(run_path).resolve()
    if p.is_file():
        cfg_json = p.parent / "config.json"
    elif p.is_dir():
        cfg_json = p / "config.json"
    else:
        # 경로 자체가 없어도 parent 기반 best-effort.
        cfg_json = p.parent / "config.json"
    if not cfg_json.exists():
        return None
    try:
        data = json.loads(cfg_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    # provenance metadata (_ prefix) 제외 — effective configs 만.
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


# v3.0.2: cq.yaml.configs 비교 시 internal/provenance 키 제외 — config.json
# fallback 과 cq.yaml 두 경로 모두에서 일관 적용.
_CONFIG_DIFF_SKIP_KEYS = frozenset({
    "_overrides_data",
    "_overrides",
    "_parent_run_id",
    "_parent_run_path",
})


def _diff_configs_dicts(a_cfg: dict, b_cfg: dict) -> list[dict]:
    """두 cfg dict 의 key-level diff. cq.yaml.configs / config.json 공용."""
    keys = sorted(set(a_cfg) | set(b_cfg))
    out: list[dict] = []
    for k in keys:
        if k in _CONFIG_DIFF_SKIP_KEYS or str(k).startswith("_"):
            continue
        in_a = k in a_cfg
        in_b = k in b_cfg
        a_v = a_cfg.get(k)
        b_v = b_cfg.get(k)
        if in_a and in_b and a_v == b_v:
            continue
        if not in_a or not in_b:
            entry: dict = {"key": k}
            if in_a:
                entry["a"] = a_v
            if in_b:
                entry["b"] = b_v
            out.append(entry)
            continue
        out.append({"key": k, "a": a_v, "b": b_v})
    return out


def _diff_cq_yaml_configs(
    a_record: dict, a_path: str | Path,
    b_record: dict, b_path: str | Path,
) -> list[dict]:
    """두 RunRecord 의 effective configs dict diff.

    우선순위:
      1. 두 cq_yaml_sha256 동일 → 변화 없음 확정 → 빈 list.
      2. 두 cq.yaml read 가능 + diff 결과 있음 → 그 diff 사용.
      3. 그 외 (yaml 미접근 / 디스크 latest 로 덮어 써져 두 dict 동일 등) →
         output_dir/config.json fallback. 두 snapshot diff 사용 (v3.0.2).

    GT-2 / G9-2 회귀: sequential gen 비교에서 cq.yaml 이 gen 1 상태로 덮어
    써지면 두 path 가 같은 file 을 가리켜 yaml diff = empty 가 된다. 이때
    각 run output_dir/config.json 에 저장된 effective cfg snapshot 으로
    fallback 한다.
    """
    a_cfg_sec = a_record.get("config") if isinstance(a_record.get("config"), dict) else {}
    b_cfg_sec = b_record.get("config") if isinstance(b_record.get("config"), dict) else {}
    a_sha = a_cfg_sec.get("cq_yaml_sha256")
    b_sha = b_cfg_sec.get("cq_yaml_sha256")
    # 1. sha256 동일 → 변화 없음 확정.
    if a_sha and b_sha and a_sha == b_sha:
        return []

    # 2. cq.yaml read 시도.
    yaml_diff: list[dict] = []
    a_yaml_path = _resolve_cq_yaml_for_record(a_record, a_path)
    b_yaml_path = _resolve_cq_yaml_for_record(b_record, b_path)
    if a_yaml_path is not None and b_yaml_path is not None:
        a_cfg = _read_cq_yaml_configs(a_yaml_path)
        b_cfg = _read_cq_yaml_configs(b_yaml_path)
        if a_cfg is not None and b_cfg is not None:
            yaml_diff = _diff_configs_dicts(a_cfg, b_cfg)
    if yaml_diff:
        return yaml_diff

    # 3. config.json fallback (v3.0.2 — GT-2 / G9-2).
    # cq.yaml 결과가 비어 있을 때만 진입. sha 가 다르다고 알려져 있으면
    # 실제 변경이 있을 가능성이 높으므로 config.json 으로 재시도.
    a_json = _read_run_config_json(a_path)
    b_json = _read_run_config_json(b_path)
    if a_json is None or b_json is None:
        return yaml_diff  # 빈 list (graceful) 또는 yaml 결과.
    return _diff_configs_dicts(a_json, b_json)


def _populate_notes(diff: RunDiff) -> None:
    """v2.3: trajectory 신호를 사람이 읽을 수 있는 note 로 변환.

    핵심 케이스:
      1. best 가 tied 인데 last 가 다름 → trajectory 변화 (hyperparameter 효과).
      2. 둘 다 best epoch=0 + tied → 학습 안 됨 ('no learning' 신호).

    note 는 정보 손실 없이 raw 필드를 보완하는 보조 시그널 — agent 는 raw 필드를
    여전히 1차 source 로 사용해야 한다.
    """
    if (
        diff.metric_direction == "tied"
        and diff.last_metric_direction in ("improved", "regressed")
        and diff.last_metric_delta is not None
    ):
        diff.notes.append(
            f"best is tied (both runs picked best epoch with same metric value), "
            f"but last epoch differs: {diff.last_metric_direction} "
            f"({diff.last_metric_delta:+.4f}). hyperparameter change affected "
            f"trajectory."
        )
    if (
        diff.best_epoch_a == 0
        and diff.best_epoch_b == 0
        and diff.metric_direction == "tied"
    ):
        diff.notes.append(
            "both runs picked epoch 0 as best — likely same initial weights "
            "(seed) and no improvement during training. agent should consider "
            "this a 'no learning' signal."
        )
