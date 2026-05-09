"""pcq.agent.inspect — project structure inspection without training.

규칙 (AGENT_OPERABILITY §`pcq inspect`):
- heavy optional deps import 금지
- 큰 모델 인스턴스화 금지
- 데이터셋 다운로드 금지
- 출력 artifact 생성 금지
- machine-readable warnings (텍스트 출력 X)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from pcq.agent.schema import (
    CqYamlSummary,
    EntrypointInfo,
    OutputsInfo,
    ProjectInspection,
    RecipeInfo,
)
from pcq.agent.yaml_io import read_yaml as _read_yaml


def _normalize_meta(value: object) -> dict:
    """nested dict 정규화 — dict 가 아니면 {value: ...} 로 wrap.

    ruamel.yaml CommentedMap → 일반 dict 로 평탄화 (assertEqual 호환성 + JSON 직렬화).
    """
    if isinstance(value, dict):
        return {str(k): _flatten(v) for k, v in value.items()}
    return {"value": _flatten(value)}


def _flatten(value: object) -> object:
    """JSON-safe 변환 — dict/list 안의 ruamel CommentedMap/CommentedSeq 도 평탄화."""
    if isinstance(value, dict):
        return {str(k): _flatten(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_flatten(v) for v in value]
    return value


def _parse_cq_yaml(path: Path) -> CqYamlSummary:
    """cq.yaml 전체 구조를 yaml_io.read_yaml 로 파싱.

    v1.15: list-style metrics 영구 호환 + dict-style metrics_schema + inputs 추출.
    """
    summary = CqYamlSummary(path=str(path))
    try:
        data = _read_yaml(path)
    except Exception as e:  # noqa: BLE001 — yaml read 는 다양한 예외
        summary.parse_error = f"yaml read failed: {type(e).__name__}: {e}"
        return summary

    if not isinstance(data, dict):
        summary.parse_error = "cq.yaml top-level must be a mapping"
        return summary

    # name, cmd
    name = data.get("name")
    if isinstance(name, str):
        summary.name = name
    cmd = data.get("cmd")
    if isinstance(cmd, str):
        summary.cmd = cmd

    # metrics — list 또는 dict
    metrics_raw = data.get("metrics")
    if isinstance(metrics_raw, list):
        summary.declared_metrics = [str(m) for m in metrics_raw]
        summary.metrics_schema = {}
    elif isinstance(metrics_raw, dict):
        # dict-style: keys 가 metric 이름, values 가 schema dict.
        summary.declared_metrics = sorted(str(k) for k in metrics_raw.keys())
        summary.metrics_schema = {
            str(k): _normalize_meta(v) for k, v in metrics_raw.items()
        }
    else:
        summary.declared_metrics = []
        summary.metrics_schema = {}

    # artifacts — list 또는 dict (v1.15+)
    artifacts_raw = data.get("artifacts")
    if isinstance(artifacts_raw, list):
        summary.artifacts = [str(a) for a in artifacts_raw]
    elif isinstance(artifacts_raw, dict):
        summary.artifacts = sorted(str(k) for k in artifacts_raw.keys())
    else:
        summary.artifacts = []

    # inputs — dict (v1.15 신규). cq URI 는 opaque string 으로 보존.
    inputs_raw = data.get("inputs")
    if isinstance(inputs_raw, dict):
        summary.inputs = {
            str(k): _normalize_meta(v) for k, v in inputs_raw.items()
        }
    else:
        summary.inputs = {}

    return summary


# 흔히 쓰는 ML framework 의 root 패키지 — entrypoint 의 import 에서 검출.
_KNOWN_ML_FRAMEWORKS = {
    "torch", "torchvision", "transformers", "sklearn",
    "xgboost", "lightgbm", "lightning", "pytorch_lightning",
    "tensorflow", "keras", "jax", "flax", "tabpfn", "pycaret",
    "fastai", "huggingface_hub", "peft", "accelerate", "timm",
    "joblib",
}


def _detect_entrypoint(project_root: Path, cmd: str | None) -> EntrypointInfo:
    """cmd 가 가리키는 train.py 에서 entrypoint kind 와 metadata 를 AST 로 탐색.

    v1.13 확장: detected_imports (ML framework) + cq_calls (pcq.X() 호출).
    """
    entry_path: Path | None = None
    if cmd:
        # "uv run python examples/train.py" → "examples/train.py" 추출
        for tok in cmd.split():
            if tok.endswith(".py"):
                entry_path = project_root / tok
                break
    if entry_path is None or not entry_path.exists():
        # fallback: 표준 위치 탐색
        for cand in ("train.py", "examples/train.py"):
            p = project_root / cand
            if p.exists():
                entry_path = p
                break

    if entry_path is None:
        return EntrypointInfo(path=None)

    rel = entry_path.relative_to(project_root) if entry_path.is_relative_to(project_root) else entry_path
    info = EntrypointInfo(path=str(rel))
    try:
        tree = ast.parse(entry_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        info.kind = "script"
        return info

    detected_frameworks: set[str] = set()
    cq_call_set: set[str] = set()
    found_trainer = False
    found_experiment = False
    found_cq_config = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in _KNOWN_ML_FRAMEWORKS:
                    detected_frameworks.add(root_pkg)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg in _KNOWN_ML_FRAMEWORKS:
                    detected_frameworks.add(root_pkg)
        elif isinstance(node, ast.Call):
            func = node.func
            # pcq.X(...) 호출 — Attribute 위치에서 pcq.attr 추출
            # cq_calls 키 이름은 호환을 위해 그대로 (의미: "library top-level call")
            if isinstance(func, ast.Attribute):
                # pcq.foo(...)
                if isinstance(func.value, ast.Name) and func.value.id == "pcq":
                    cq_call_set.add(f"pcq.{func.attr}")
                    if func.attr == "config":
                        found_cq_config = True
                # pcq.Trainer 직접 호출 또는 pcq.Trainer.from_cfg(...)
                if func.attr == "Trainer":
                    if (isinstance(func.value, ast.Name) and func.value.id == "pcq"):
                        found_trainer = True
                # pcq.Trainer.from_cfg(...) 형태
                if (
                    isinstance(func.value, ast.Attribute)
                    and func.value.attr == "Trainer"
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "pcq"
                ):
                    found_trainer = True
            elif isinstance(func, ast.Name) and func.id == "Trainer":
                # `from pcq import Trainer; Trainer(...)`
                found_trainer = True
            # preset 추출 — Trainer 호출에서만
            is_trainer_call = (
                (isinstance(func, ast.Attribute) and func.attr == "Trainer"
                 and isinstance(func.value, ast.Name) and func.value.id == "pcq")
                or (isinstance(func, ast.Name) and func.id == "Trainer")
            )
            if is_trainer_call:
                for kw in node.keywords:
                    if kw.arg == "preset" and isinstance(kw.value, ast.Constant):
                        info.preset = kw.value.value
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                if (
                    (isinstance(base, ast.Attribute) and base.attr == "Experiment")
                    or (isinstance(base, ast.Name) and base.id == "Experiment")
                ):
                    found_experiment = True

    info.detected_imports = sorted(detected_frameworks)
    info.cq_calls = sorted(cq_call_set)

    # kind 우선순위: trainer > experiment > script (pcq.config 만으로도 script)
    if found_trainer:
        info.kind = "trainer"
    elif found_experiment:
        info.kind = "experiment"
    elif found_cq_config:
        info.kind = "script"
    else:
        info.kind = "script"
    return info


def _list_recipes_for_inspection() -> list[RecipeInfo]:
    """v4.0: pcq 내장 recipe 카탈로그가 사라졌으므로 항상 빈 리스트.

    하위 호환을 위해 schema 의 recipes 필드는 유지하되 콘텐츠는 항상 [].
    """
    return []


def _detect_outputs(
    project_root: Path,
    _cq_yaml: CqYamlSummary | None,
    resolved_output_dir: Path | None = None,
) -> OutputsInfo:
    """output_dir 위치 추정 + artifact 존재 여부 점검.

    v2.5: ResolvedConfig.output_dir 우선 — cq.yaml.configs.output_dir 또는
    CQ_CONFIG_JSON.output_dir 기반. 없으면 legacy 'output' 디폴트.

    READ-ONLY: output_dir 이 없거나 비어 있어도 새로 만들지 않음.

    v1.14: manifest.json 발견 시 schema_version + files count 함께 노출.
    v2.5 (P2 #5): output_dir 비어 있으면 status="empty" 명시.
    """
    info = OutputsInfo()

    # 후보 결정: ResolvedConfig 의 output_dir 우선, 없으면 legacy 'output'.
    candidates: list[Path] = []
    if resolved_output_dir is not None:
        candidates.append(resolved_output_dir)
    legacy = project_root / "output"
    if legacy not in candidates:
        candidates.append(legacy)

    for d in candidates:
        if not d.exists() or not d.is_dir():
            continue
        try:
            rel = d.relative_to(project_root)
            info.output_dir = str(rel)
        except ValueError:
            # output_dir이 project_root 밖이면 absolute 사용.
            info.output_dir = str(d)
        manifest_path = d / "manifest.json"
        info.has_manifest = manifest_path.exists()
        info.has_metrics = (d / "metrics.json").exists()
        info.has_summary = (d / "run_summary.json").exists()
        # v1.16: RunRecord + validation report.
        info.has_run_record = (d / "run_record.json").exists()
        info.has_validation_report = (
            d / "validation_report.json"
        ).exists()
        # manifest 내부 스키마 정보 — agent 가 v1/v2 분기 가능하게.
        if info.has_manifest:
            try:
                m = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                info.manifest_schema_version = m.get("schema_version")
                info.manifest_files_count = len(m.get("files", []))
            except (json.JSONDecodeError, OSError):
                # 손상된 manifest 는 has_manifest=True 만 유지, 스키마/카운트 unknown.
                pass

        # v2.5: status — empty / partial / complete.
        any_artifact = (
            info.has_manifest
            or info.has_metrics
            or info.has_summary
            or info.has_run_record
        )
        all_core = info.has_manifest and info.has_metrics and info.has_run_record
        if not any_artifact:
            info.status = "empty"
        elif all_core:
            info.status = "complete"
        else:
            info.status = "partial"
        break
    return info


def _build_cq_yaml_summary_from_resolver(rc) -> CqYamlSummary:
    """ResolvedConfig 를 CqYamlSummary 로 변환 (inspect 가 반환하는 형태)."""
    summary = CqYamlSummary(path=str(rc.cq_yaml_path) if rc.cq_yaml_path else "")
    if rc.name:
        summary.name = rc.name
    if rc.cmd:
        summary.cmd = rc.cmd
    summary.declared_metrics = list(rc.declared_metrics)
    summary.metrics_schema = dict(rc.metrics_schema)
    summary.artifacts = list(rc.artifacts)
    summary.inputs = dict(rc.inputs)
    if rc.parse_errors:
        summary.parse_error = "; ".join(rc.parse_errors)
    return summary


def inspect_project(
    path: str | Path = ".",
    *,
    load_project_atoms: bool = False,
) -> ProjectInspection:
    """Project structure 검사. 학습/heavy import 없이.

    v2.5: read-only — output_dir 가 없거나 비어 있어도 mkdir 하지 않음.
    cq.yaml parse_errors 가 있으면 insp.errors 에 명시.

    Returns:
        ProjectInspection — JSON-serializable 구조체.
    """
    from pcq.agent.resolver import resolve_project

    project_root = Path(path).resolve()
    insp = ProjectInspection(project_root=str(project_root))

    if not project_root.exists():
        insp.errors.append(f"path does not exist: {project_root}")
        return insp

    # cq.yaml 검출 — root 또는 examples/ 안.
    # examples/cq.yaml 도 지원하기 위해 직접 glob (resolver는 root만 봄).
    cq_yaml_paths = list(project_root.glob("cq.yaml")) + list(
        project_root.glob("examples/cq.yaml")
    )
    rc = None
    if cq_yaml_paths:
        cq_yaml_path = cq_yaml_paths[0]
        insp.has_cq_yaml = True
        # v2.2: ResolvedConfig 단일 view에서 CqYamlSummary 파생.
        rc = resolve_project(cq_yaml_path=cq_yaml_path)
        insp.cq_yaml = _build_cq_yaml_summary_from_resolver(rc)
        insp.project_type = "pcq"
        insp.entrypoint = _detect_entrypoint(project_root, insp.cq_yaml.cmd)
        # v2.0.2: Trainer.from_cfg(cfg) 패턴은 preset이 cq.yaml.configs.preset
        # 에 있음. AST 추출이 None이면 cq.yaml에서 fallback 추출.
        if not insp.cq_yaml.declared_metrics:
            insp.warnings.append("cq.yaml has no declared metrics")
        # v2.5 (P2 #4): malformed cq.yaml — parse_errors → insp.errors.
        for pe in rc.parse_errors:
            insp.errors.append(f"cq.yaml: {pe}")
    else:
        insp.warnings.append("no cq.yaml found")
        insp.entrypoint = EntrypointInfo(path=None)

    # v4.0: project-local atom system 제거. load_project_atoms 인자는 받지만 noop.
    insp.project_atoms_loaded = {
        "loaded": False,
        "reason": "atom registry removed in v4.0",
    }

    # 등록된 recipe 카탈로그
    insp.recipes = _list_recipes_for_inspection()

    # output 디렉토리 — ResolvedConfig.output_dir 우선 (v2.5).
    resolved_out = rc.output_dir if rc is not None else None
    insp.outputs = _detect_outputs(project_root, insp.cq_yaml, resolved_out)

    return insp
