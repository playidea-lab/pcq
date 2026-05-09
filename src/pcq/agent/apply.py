"""pcq.agent.apply — bounded application of ExperimentPlan to a project.

v4.0 allowed mutations:
  - cq.yaml: configs.<key> = <value>            ← set_config
  - cq.yaml: configs._parent_run_id  = ...      ← plan.parent_run_id
  - cq.yaml: configs._parent_run_path = ...     ← plan.parent_run_path

Provenance: .pcq/plans/<plan_id>.json (자동 저장).
train.py 는 절대 수정하지 않는다 — 사용자 영역.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pcq.agent.plan import ChangeOp, ExperimentPlan, ExperimentPlanSet
from pcq.agent.yaml_io import read_yaml, write_yaml


@dataclass
class ApplyResult:
    """apply_plan 결과 — JSON-safe."""

    schema_version: int = 1
    status: str = "applied"   # "applied" | "no_changes" | "rejected"
    plan_id: str = ""
    files_changed: list[str] = field(default_factory=list)
    operations: list[dict] = field(default_factory=list)
    rejected_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "plan_id": self.plan_id,
            "files_changed": self.files_changed,
            "operations": self.operations,
            "rejected_reasons": self.rejected_reasons,
            "warnings": self.warnings,
        }


def apply_plan(
    project_root: str | Path,
    plan: ExperimentPlan | dict,
) -> ApplyResult:
    """Plan 을 project 에 적용. cq.yaml configs 만 수정.

    Returns ApplyResult — status:
      - "applied": cq.yaml 변경됨, .pcq/plans/<id>.json 저장
      - "no_changes": 모든 op 가 idempotent (현재 값과 동일)
      - "rejected": plan 검증 실패 — 어떠한 파일도 쓰지 않음
    """
    if isinstance(plan, dict):
        plan = ExperimentPlan.from_dict(plan)

    result = ApplyResult(plan_id=plan.id)

    # ── 1. plan 자체 검증 ─────────────────────────────────────────────
    plan_errors = plan.validate()
    if plan_errors:
        result.status = "rejected"
        result.rejected_reasons.extend(plan_errors)
        return result

    root = Path(project_root).resolve()
    cq_path = root / "cq.yaml"
    if not cq_path.exists():
        result.status = "rejected"
        result.rejected_reasons.append(f"cq.yaml not found at {cq_path}")
        return result

    # ── 2. cq.yaml 읽기 ──────────────────────────────────────────────
    try:
        cq_data = read_yaml(cq_path)
    except Exception as e:  # noqa: BLE001
        result.status = "rejected"
        result.rejected_reasons.append(f"cq.yaml unreadable: {e}")
        return result

    if not isinstance(cq_data, dict):
        result.status = "rejected"
        result.rejected_reasons.append(
            f"cq.yaml top-level must be a mapping, got {type(cq_data).__name__}"
        )
        return result

    if "configs" not in cq_data or not isinstance(cq_data.get("configs"), dict):
        cq_data["configs"] = {}
    configs: dict = cq_data["configs"]

    # ── 3. 각 change 적용 ────────────────────────────────────────────
    cq_changed = False
    pending_ops: list[dict] = []

    for c in plan.changes:
        if c.op == "set_config":
            op_record: dict = c.to_dict()
            old_val = configs.get(c.key)
            if old_val == c.value:
                op_record["status"] = "noop"
            else:
                op_record["previous_value"] = old_val
                op_record["status"] = "applied"
                configs[c.key] = c.value
                cq_changed = True
            pending_ops.append(op_record)
            continue

        # 알 수 없는 op — 안전하게 reject (validate 가 잡았어야 함)
        result.rejected_reasons.append(f"unsupported op: {c.op!r}")

    if result.rejected_reasons:
        result.status = "rejected"
        # rejected 면 cq.yaml 절대 안 씀, provenance 도 안 남김.
        return result

    # ── 3b. v1.18 lineage: plan 의 parent_run_id / parent_run_path 를
    # cq.yaml.configs._parent_run_id / _parent_run_path 로 주입.
    # finalize_run 이 RunRecord.run.parent_run_* 에 자동 기록한다.
    if plan.parent_run_id:
        old = configs.get("_parent_run_id")
        if old != plan.parent_run_id:
            configs["_parent_run_id"] = plan.parent_run_id
            cq_changed = True
            pending_ops.append(
                {
                    "op": "set_parent_run_id",
                    "value": plan.parent_run_id,
                    "previous_value": old,
                    "status": "applied",
                }
            )
        else:
            pending_ops.append(
                {
                    "op": "set_parent_run_id",
                    "value": plan.parent_run_id,
                    "status": "noop",
                }
            )
    if plan.parent_run_path:
        old = configs.get("_parent_run_path")
        if old != plan.parent_run_path:
            configs["_parent_run_path"] = plan.parent_run_path
            cq_changed = True
            pending_ops.append(
                {
                    "op": "set_parent_run_path",
                    "value": plan.parent_run_path,
                    "previous_value": old,
                    "status": "applied",
                }
            )
        else:
            pending_ops.append(
                {
                    "op": "set_parent_run_path",
                    "value": plan.parent_run_path,
                    "status": "noop",
                }
            )

    # 모든 op 가 통과 — 기록
    result.operations = pending_ops

    # ── 4. cq.yaml 저장 ──────────────────────────────────────────────
    if cq_changed:
        write_yaml(cq_data, cq_path)
        result.files_changed.append("cq.yaml")
        result.status = "applied"
    else:
        result.status = "no_changes"

    # ── 5. provenance: .pcq/plans/<plan_id>.json ────────────────────
    plans_dir = root / ".pcq" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": 1,
        "plan": plan.to_dict(),
        "applied_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files_changed": list(result.files_changed),
        "operations": list(result.operations),
        "status": result.status,
    }
    plan_path = plans_dir / f"{plan.id}.json"
    plan_path.write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )

    return result


@dataclass
class PlanSetApplyResult:
    """apply_planset 의 set-level 결과 — JSON-safe.

    expanded[i].apply 는 멤버 plan 적용 결과 (ApplyResult.to_dict()).
    set 자체가 reject 되면 status="rejected" + rejected_reasons 채움 — 어떤
    output_dir 도 만들지 않는다 (atomic).
    """

    schema_version: int = 1
    status: str = "applied"          # "applied" | "rejected"
    set_id: str = ""
    expanded: list[dict] = field(default_factory=list)
    rejected_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "set_id": self.set_id,
            "expanded": list(self.expanded),
            "rejected_reasons": list(self.rejected_reasons),
        }


# v4.2 (GM-5): planset 멤버 dir 가 자족적으로 실행 가능하도록 root 의 workspace
# 파일을 symlink 한다. dogfood research/mcp-dogfood: 멤버 cq.yaml 의 cmd
# (`python train.py`) 가 가리키는 train.py 는 root 에만 있고 멤버 dir 에는 없어
# `pcq run --path member/dir` 가 실패했음.
_WORKSPACE_LINK_FILES = (
    "train.py",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
)


def _link_workspace_files(root: Path, member_dir: Path) -> list[str]:
    """root 의 workspace 파일을 member_dir 에 symlink (없으면 copy fallback).

    v4.2 (GM-5): apply_planset 멤버가 root 와 동일하게 train.py / pyproject.toml
    을 볼 수 있어야 `pcq run --path member/dir` 가 정상 동작한다. cq.yaml 은
    이미 멤버 별 patch 본이 있으므로 link 대상이 아님.

    Returns: 실제로 link/copy 된 파일 이름 list (이미 존재하거나 root 에 없는
    파일은 skip).
    """
    import shutil

    linked: list[str] = []
    for fname in _WORKSPACE_LINK_FILES:
        src = root / fname
        if not src.exists():
            continue
        dst = member_dir / fname
        if dst.exists() or dst.is_symlink():
            continue
        try:
            # 절대 경로 symlink — member_dir 의 위치와 무관.
            dst.symlink_to(src.resolve())
            linked.append(fname)
        except OSError:
            # symlink 권한/지원 없으면 copy 로 fallback (Windows non-admin 등).
            try:
                shutil.copy2(src, dst)
                linked.append(fname)
            except OSError:
                # copy 도 실패하면 graceful skip — 멤버 실행 시 명확한
                # 'train.py not found' 에러로 노출되도록 둠.
                pass
    return linked


def _normalize_member_output_dir(
    plan: ExperimentPlan, expanded_dir: Path
) -> ExperimentPlan:
    """v2.12 (G7-1): expand 시 멤버 plan 의 output_dir set_config 정규화.

    relative path → expanded_dir 안의 'output' 으로 강제 normalize.
    train.py 가 expanded_dir 에서 실행될 때 pcq.output_dir() 이
    `<expanded>/<member_value>` 로 이중 nesting 되는 것을 방지.

    absolute path → 그대로 보존 (사용자 의도 존중).

    멤버 plan 의 output_dir set_config op 가 없으면 plan 그대로 반환.
    """
    new_changes: list[ChangeOp] = []
    changed = False
    for c in plan.changes:
        if c.op == "set_config" and c.key == "output_dir":
            raw = c.value
            if isinstance(raw, str):
                p = Path(raw).expanduser()
                if not p.is_absolute():
                    # relative → expanded_dir 안의 'output' 으로 강제.
                    # 이중 nesting (`<expanded>/<raw>`) 방지.
                    new_changes.append(ChangeOp(
                        op="set_config",
                        key="output_dir",
                        value="output",
                    ))
                    changed = True
                    continue
            # absolute 또는 비-string 값 — 그대로.
        new_changes.append(c)

    if not changed:
        return plan
    # ExperimentPlan 은 dataclass — 새 인스턴스로 바꿔준다.
    return ExperimentPlan(
        schema_version=plan.schema_version,
        id=plan.id,
        intent=plan.intent,
        base=dict(plan.base),
        target=dict(plan.target),
        changes=new_changes,
        validation_policy=plan.validation_policy,
        parent_run_id=plan.parent_run_id,
        parent_run_path=plan.parent_run_path,
    )


def _ensure_member_output_dir(plan: ExperimentPlan) -> ExperimentPlan:
    """v4.2 (GM-6): plan 에 set_config(output_dir=...) 가 없으면 'output' 자동 주입.

    이전 동작: 멤버가 output_dir 를 안 적으면 root cq.yaml 의 output_dir 를
    그대로 상속 → N 멤버가 모두 같은 output_dir 를 공유 → artifact 충돌.
    v4.2: 멤버가 output_dir 를 명시 안 하면 'output' (멤버 dir 기준) 자동 주입.
    명시한 멤버는 그대로 — 사용자 의도 우선.
    """
    has_output_dir = any(
        c.op == "set_config" and c.key == "output_dir"
        for c in plan.changes
    )
    if has_output_dir:
        return plan
    new_changes = list(plan.changes) + [
        ChangeOp(op="set_config", key="output_dir", value="output"),
    ]
    return ExperimentPlan(
        schema_version=plan.schema_version,
        id=plan.id,
        intent=plan.intent,
        base=dict(plan.base),
        target=dict(plan.target),
        changes=new_changes,
        validation_policy=plan.validation_policy,
        parent_run_id=plan.parent_run_id,
        parent_run_path=plan.parent_run_path,
    )


def apply_planset(
    project_root: str | Path,
    planset: ExperimentPlanSet | dict,
    *,
    output_pattern: str = "runs/exp{i}",
    force: bool = False,
) -> PlanSetApplyResult:
    """ExperimentPlanSet 을 N 개 output_dir 로 expand 적용.

    각 멤버 plan 은:
      1. project_root/output_pattern.format(i=i, plan_id=plan.id) 에 cq.yaml 복사
      2. set.parent_run_id/parent_run_path 를 멤버 plan 에 propagate (멤버가 안 적었을 때)
      3. apply_plan() 으로 cq.yaml 수정 — output_dir 도 patch.

    output_pattern 은 `{i}` (zero-based index) 또는 `{plan_id}` 를 포함할 수
    있다. 미지원 placeholder 는 silently 그대로 둔다 (str.format 표준).

    v2.12 (G7-1): 멤버 plan 의 set_config(output_dir=...) 에서 relative path 는
    expanded_dir 안의 'output' 으로 normalize. 이중 nesting 방지. absolute path
    는 그대로 보존.

    Args:
        project_root: cq.yaml 가진 base project. 멤버 directory 도 같은 project
                      이어야 함 (cq.yaml 을 복사해서 시작).
        planset: ExperimentPlanSet 또는 dict.
        output_pattern: expand 패턴 (기본 "runs/exp{i}"). project_root 기준
                        relative.
        force: True 면 기존 output_dir 를 덮어씀. False (default) 면 기존
                존재 시 reject (rejected_reasons 에 사유).

    Returns:
        PlanSetApplyResult.
    """
    if isinstance(planset, dict):
        planset = ExperimentPlanSet.from_dict(planset)

    result = PlanSetApplyResult(set_id=planset.id)

    # ── 1. set-level 검증 ────────────────────────────────────────────
    set_errors = planset.validate()
    if set_errors:
        result.status = "rejected"
        result.rejected_reasons.extend(set_errors)
        return result

    root = Path(project_root).resolve()
    base_cq = root / "cq.yaml"
    if not base_cq.exists():
        result.status = "rejected"
        result.rejected_reasons.append(f"cq.yaml not found at {base_cq}")
        return result

    # ── 2. 각 멤버 plan expand ───────────────────────────────────────
    base_text = base_cq.read_text(encoding="utf-8")
    expanded: list[dict] = []
    for i, plan in enumerate(planset.plans):
        # parent_run_id / parent_run_path 는 set → plan 으로 propagate (plan 미명시 시).
        if planset.parent_run_id and not plan.parent_run_id:
            plan.parent_run_id = planset.parent_run_id
        if planset.parent_run_path and not plan.parent_run_path:
            plan.parent_run_path = planset.parent_run_path

        # output_dir 패턴 적용
        try:
            rel_dir = output_pattern.format(i=i, plan_id=plan.id)
        except (KeyError, IndexError) as e:
            result.status = "rejected"
            result.rejected_reasons.append(
                f"output_pattern format failed for plan {plan.id!r}: {e}"
            )
            return result
        target_dir = (root / rel_dir).resolve()

        # 기존 dir 처리
        if target_dir.exists() and not force:
            expanded.append({
                "plan_id": plan.id,
                "output_path": str(target_dir),
                "status": "skipped",
                "reason": "directory exists (use --force to overwrite)",
            })
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        # cq.yaml 복사 (base 그대로 — apply_plan 이 그 위에 patch).
        (target_dir / "cq.yaml").write_text(base_text, encoding="utf-8")

        # v4.2 (GM-5): 멤버 dir 가 자족적으로 실행 가능하도록 train.py 등 link.
        # link 결과는 expanded entry 에 기록 (debug 용).
        linked_files = _link_workspace_files(root, target_dir)

        # v2.12 (G7-1): 멤버 plan 의 relative output_dir 은 'output' 으로 normalize.
        # absolute path 는 보존. 이중 nesting 방지.
        normalized_plan = _normalize_member_output_dir(plan, target_dir)
        # v4.2 (GM-6): 멤버가 output_dir 를 명시 안 했으면 'output' 자동 주입.
        # N 멤버가 root output_dir 를 공유해 충돌하던 dogfood 회귀 fix.
        normalized_plan = _ensure_member_output_dir(normalized_plan)
        member_result = apply_plan(target_dir, normalized_plan)
        member_dict = member_result.to_dict()
        expanded_entry = {
            "plan_id": plan.id,
            "output_path": str(target_dir),
            "status": member_result.status,
            "apply": member_dict,
        }
        if linked_files:
            expanded_entry["linked_files"] = linked_files
        expanded.append(expanded_entry)
        # 멤버 reject 는 set 전체 reject 사유에 포함 — 단 이미 만든 dir 는 그대로.
        if member_result.status == "rejected":
            result.rejected_reasons.extend(
                f"plan {plan.id!r}: {r}"
                for r in member_result.rejected_reasons
            )

    result.expanded = expanded
    if result.rejected_reasons:
        result.status = "rejected"
    else:
        result.status = "applied"
    return result
