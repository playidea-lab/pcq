"""pcq.agent.apply — bounded application of ExperimentPlan to a project.

Allowed mutations (v1.11):
  - cq.yaml: configs.<key> = <value>     ← set_config
  - cq.yaml: configs._overrides_data.<atom_key> = AtomRef.to_dict()  ← set_atom
  - set_dataset_transform → set_atom merge=true on dataset_<split>

v1.13: script-style 프로젝트는 set_atom / set_dataset_transform 거부.

Provenance: .pcq/plans/<plan_id>.json (자동 저장).
train.py / recipes/ 는 절대 수정하지 않는다 — 사용자 영역.
"""
from __future__ import annotations

import ast
import importlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pcq.agent.plan import ChangeOp, ExperimentPlan, ExperimentPlanSet
from pcq.agent.yaml_io import read_yaml, write_yaml
from pcq.registry.spec import AtomRef


def _is_script_entrypoint(project_root: Path) -> bool:
    """train.py 가 contract script (pcq.config 호출, Trainer/Experiment 없음) 인지.

    가벼운 AST 스캔만 — load_project_atoms 부작용 회피. inspect_project 와는
    분리되어, apply-plan 의 path 결정에만 사용.
    """
    train_py = project_root / "train.py"
    if not train_py.exists():
        return False
    try:
        tree = ast.parse(train_py.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return False

    has_cq_config = False
    has_trainer_or_experiment = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # pcq.config()
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "pcq"
                and func.attr == "config"
            ):
                has_cq_config = True
            # pcq.Trainer(...) | Trainer(...) | pcq.Trainer.from_cfg(...)
            if isinstance(func, ast.Attribute) and func.attr == "Trainer":
                has_trainer_or_experiment = True
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "Trainer"
            ):
                has_trainer_or_experiment = True
            if isinstance(func, ast.Name) and func.id == "Trainer":
                has_trainer_or_experiment = True
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                if (
                    (isinstance(base, ast.Attribute) and base.attr == "Experiment")
                    or (isinstance(base, ast.Name) and base.id == "Experiment")
                ):
                    has_trainer_or_experiment = True
    return has_cq_config and not has_trainer_or_experiment


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


# atom kind → registry 매핑 (지연 import 로 작성)
def _registry_map() -> dict:
    from pcq import registry as registry_pkg

    return {
        "model": registry_pkg.models,
        "dataset": registry_pkg.datasets,
        "loss": registry_pkg.losses,
        "optim": registry_pkg.optims,
        "sched": registry_pkg.scheds,
        "metric": registry_pkg.metrics,
    }


def _resolve_base_atom_ref(preset: str | None, atom_key: str) -> AtomRef | None:
    """base recipe SPEC 에서 atom_key 의 AtomRef 추출. 실패 시 None.

    v1.11 set_atom merge=True 시 기존 override 가 없을 때 base 의 atom 정보를
    상속하기 위해 사용. SPEC.atoms 가 RecipeSpec metadata-first 형식인 경우만
    동작 — recipe 가 callable 직접 등록 (구식) 인 경우 None.
    """
    if not preset:
        return None
    try:
        from pcq.trainer import _import_recipe

        fn = _import_recipe(preset)
        mod = importlib.import_module(fn.__module__)
        spec = getattr(mod, "SPEC", None)
        if spec is None:
            return None
        atom = spec.atoms.get(atom_key)
        if isinstance(atom, AtomRef):
            return AtomRef(
                kind=atom.kind, name=atom.name, params=dict(atom.params)
            )
    except Exception:
        return None
    return None


def _apply_set_atom(
    c: ChangeOp,
    overrides_data: dict,
    preset: str | None,
    reg_map: dict,
    result: ApplyResult,
) -> tuple[bool, dict | None]:
    """set_atom 적용 (merge / replace 분기).

    Returns:
        (applied_or_noop, op_record). op_record None 이면 reject 되었음
        (result.rejected_reasons 에 사유 추가됨).
    """
    op_record: dict = c.to_dict()
    kind = c._infer_kind()
    reg = reg_map.get(kind)
    if reg is None:
        result.rejected_reasons.append(
            f"set_atom: unknown kind {kind!r} for atom {c.atom!r}"
        )
        return False, None

    # merge 모드: 기존 override 또는 base recipe AtomRef 로부터 상속
    if c.merge:
        existing = overrides_data.get(c.atom)
        base_name: str | None = None
        base_params: dict = {}
        if isinstance(existing, dict) and existing.get("kind") == kind:
            base_name = existing.get("name")
            base_params = dict(existing.get("params") or {})
        else:
            # 기존 override 없음 — base recipe 의 SPEC.atoms 에서 상속 시도
            base_ref = _resolve_base_atom_ref(preset, c.atom)
            if base_ref is not None and base_ref.kind == kind:
                base_name = base_ref.name
                base_params = dict(base_ref.params)

        merged_params = {**base_params, **(c.params or {})}
        new_name = c.name if c.name else base_name
        if not new_name:
            result.rejected_reasons.append(
                f"set_atom merge=true: cannot resolve base atom name for "
                f"{c.atom!r} (no existing override and base recipe has no "
                f"AtomRef for this slot)"
            )
            return False, None
        ref = AtomRef(kind=kind, name=new_name, params=merged_params)
    else:
        # 완전 교체 (v1.10 동작)
        ref = AtomRef(kind=kind, name=c.name or "", params=dict(c.params or {}))

    ref_errors = reg.validate_ref(ref)
    if ref_errors:
        result.rejected_reasons.append(
            f"set_atom {c.atom}={ref.name}: {'; '.join(ref_errors)}"
        )
        return False, None

    new_ref = ref.to_dict()
    old_ref = overrides_data.get(c.atom)
    if old_ref == new_ref:
        op_record["status"] = "noop"
        return False, op_record

    op_record["previous_ref"] = old_ref
    op_record["status"] = "applied"
    # set_dataset_transform 경유면 사용된 효과적 atom 키와 name 도 기록 (디버깅 보조)
    op_record["resolved_name"] = ref.name
    overrides_data[c.atom] = new_ref
    return True, op_record


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

    # ── 1b. v1.13: script-style 프로젝트는 atom-level op 거부 ──────
    # set_config 만 허용 (atom system 미사용).
    # Keep this as a narrow AST scan; apply-plan only needs entrypoint kind and
    # should not depend on the broader inspect schema.
    is_script = _is_script_entrypoint(Path(project_root).resolve())

    if is_script:
        for c in plan.changes:
            if c.op == "set_atom":
                result.rejected_reasons.append(
                    "set_atom requires recipe-style project; this project "
                    "uses contract script (entrypoint kind=script). Use "
                    "set_config or edit script directly."
                )
            elif c.op == "set_dataset_transform":
                result.rejected_reasons.append(
                    "set_dataset_transform requires recipe-style project "
                    "(script project detected)"
                )
        if result.rejected_reasons:
            result.status = "rejected"
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
    overrides_data: dict = configs.setdefault("_overrides_data", {})
    preset = plan.base.get("preset")

    # ── 3. 각 change 적용 (registry-aware) ───────────────────────────
    reg_map = _registry_map()
    cq_changed = False
    pending_ops: list[dict] = []

    for c in plan.changes:
        # set_dataset_transform 은 set_atom merge=True 로 desugar
        effective = c
        if c.op == "set_dataset_transform":
            effective = c.to_set_atom()

        if effective.op == "set_config":
            op_record: dict = effective.to_dict()
            old_val = configs.get(effective.key)
            if old_val == effective.value:
                op_record["status"] = "noop"
            else:
                op_record["previous_value"] = old_val
                op_record["status"] = "applied"
                configs[effective.key] = effective.value
                cq_changed = True
            pending_ops.append(op_record)
            continue

        if effective.op == "set_atom":
            applied, op_record = _apply_set_atom(
                effective, overrides_data, preset, reg_map, result
            )
            if op_record is None:
                continue
            # set_dataset_transform 원본 op 정보를 보조 기록 (provenance)
            if c.op == "set_dataset_transform":
                op_record["sugar_op"] = "set_dataset_transform"
                op_record["sugar_split"] = c.split
            if applied:
                cq_changed = True
            pending_ops.append(op_record)
            continue

        # 알 수 없는 op — 안전하게 reject (validate 가 잡았어야 함)
        result.rejected_reasons.append(f"unsupported op: {effective.op!r}")

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

    # _overrides_data 가 비어 있으면 키 자체 제거 (clean YAML)
    if not overrides_data:
        configs.pop("_overrides_data", None)

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

        # set output_dir 도 cq.yaml.configs.output_dir 에 주입 (멤버 plan 이
        # 별도 set_config 안 한 경우, 표준 위치 "output" 으로).
        # apply_plan 이 cq.yaml 을 수정하므로 그 안에서 처리.

        # v2.12 (G7-1): 멤버 plan 의 relative output_dir 은 'output' 으로 normalize.
        # absolute path 는 보존. 이중 nesting 방지.
        normalized_plan = _normalize_member_output_dir(plan, target_dir)
        member_result = apply_plan(target_dir, normalized_plan)
        member_dict = member_result.to_dict()
        expanded.append({
            "plan_id": plan.id,
            "output_path": str(target_dir),
            "status": member_result.status,
            "apply": member_dict,
        })
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
