"""pcq.contract — CQ contract artifact helpers.

Framework-agnostic. Use these from any project-local script (HF Trainer,
sklearn, XGBoost, custom code) to produce standard pcq artifacts:
config.json, metrics.json, manifest.json, run_summary.json,
run_record.json (v1.16+), validation_report.json (v1.16+).

These helpers do NOT depend on pcq.Experiment or pcq.Trainer. They write the
same files that Experiment.fit() writes, so contract scripts and Experiment
runs produce indistinguishable output to CQ.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any



# manifest schema v2 enrich 시 sha256 계산 chunk 크기 (1 MiB).
_SHA256_CHUNK_SIZE = 1 << 20


# manifest 자동 검출 시 확장자 → kind 매핑.
_KIND_BY_SUFFIX: dict[str, str] = {
    ".pt": "model",
    ".pth": "model",
    ".pkl": "model",
    ".joblib": "model",
    ".safetensors": "model",
    ".bin": "model",
    ".onnx": "model",
    ".ckpt": "checkpoint",
}

# 특수 파일명 → kind. None 이면 manifest 에서 제외 (자기참조 회피).
_SPECIAL_FILES: dict[str, str | None] = {
    "metrics.json": "metrics",
    "config.json": "config",
    "run_summary.json": "summary",
    "manifest.json": None,
}


def _run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run git with a bounded timeout. Callers handle return codes."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        timeout=5,
    )


def _git_sha(cwd: Path | None = None) -> str:
    """현재 git HEAD sha. git 없거나 timeout 이면 빈 문자열."""
    try:
        out = _run_git(["rev-parse", "HEAD"], cwd=cwd)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _pcq_version() -> str:
    """cq 패키지 __version__. 실패 시 'unknown'."""
    try:
        from pcq import __version__

        return __version__
    except ImportError:
        return "unknown"


def _cq_config_available() -> bool:
    """CQ_CONFIG_JSON env var 가 설정되어 있는지."""

    return bool(os.environ.get("CQ_CONFIG_JSON"))


def _classify_exception(exc: BaseException) -> tuple[str, str, dict]:
    """(error_code, category, evidence) 자동 분류 (v2.11).

    명시 pcq.save_all(failure={...}) 가 우선. 자동 분류는 unhandled exception
    폴백용 — caller 가 except 블록에서 호출.

    pcq 은 enum + structured field 만 발급. suggested_fix 자연어 생성은
    하지 않는다 (agent 정책 영역).
    """
    if isinstance(exc, ImportError):
        # exc.name (3.6+) 우선, fallback 으로 exception 메시지.
        module = getattr(exc, "name", None) or str(exc)
        return (
            "ERR_MISSING_DEPENDENCY",
            "missing_dependency",
            {"module": str(module), "exception_type": type(exc).__name__},
        )
    if isinstance(exc, MemoryError):
        return (
            "ERR_OUT_OF_MEMORY",
            "oom",
            {"exception_type": type(exc).__name__},
        )
    if isinstance(exc, TimeoutError):
        return (
            "ERR_TIMEOUT",
            "timeout",
            {"exception_type": type(exc).__name__},
        )
    if isinstance(exc, FileNotFoundError):
        return (
            "ERR_DATASET_UNAVAILABLE",
            "dataset_missing",
            {
                "exception_type": type(exc).__name__,
                "path": str(getattr(exc, "filename", "") or ""),
            },
        )
    return (
        "ERR_RUNTIME",
        "unknown_exception",
        {"exception_type": type(exc).__name__},
    )


def _normalize_failure(failure: dict | None) -> dict | None:
    """failure dict 를 v2.11 FailureInfo schema 로 정규화.

    - error_code 미지정 + category 있으면 derive (category_to_error_code).
    - evidence 미지정이면 빈 dict 보장.
    - suggested_fix 는 agent 영역 — 손대지 않는다 (자연어 보존).
    """
    if not failure:
        return failure
    from pcq.agent.run_record import FailureInfo

    info = FailureInfo.from_dict(failure)
    # FailureInfo.to_dict 는 빈 값 제거 — 그러나 evidence 는 의미 있으면 명시 보존.
    out = info.to_dict()
    # caller 가 명시한 추가 키 (예: agent 가 추가한 future-proof 필드) 보존.
    for k, v in failure.items():
        if k in ("error_code", "category", "message", "evidence", "suggested_fix"):
            continue
        out[k] = v
    # evidence 가 빈 dict 라도 enum 명시면 노출 (agent 가 키 존재 여부로 분기).
    if "error_code" in out and "evidence" not in out:
        out["evidence"] = {}
    return out


def _resolve_write_context(
    cfg: dict | None = None,
    output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
):
    """contract helper용 RunContext — write-side mkdir 단일 경로.

    Priority for output_dir: explicit output_dir → explicit cfg["output_dir"]
    → CQ_CONFIG_JSON → cq.yaml.configs.output_dir → project_root/output.
    cfg 가 명시되면 그것의 output_dir 이 RunContext 의 output_dir 로 override
    된다.
    """
    from pcq.agent.resolver import resolve_run_context

    explicit_out = output_dir
    if explicit_out is None and cfg is not None and "output_dir" in cfg:
        explicit_out = cfg["output_dir"]
    return resolve_run_context(
        path=project_root,
        output_dir=explicit_out,
        ensure_output_dir=True,
    )


def save_config_snapshot(
    extra: dict | None = None,
    cfg: dict | None = None,
    output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """cfg + provenance 를 output_dir/config.json 에 저장.

    추가 metadata: _git_sha, _pcq_version. extra 가 주어지면 merge.

    Args:
        extra: 추가 metadata (예: {"_recipe": "...", "_overrides": [...]}).
        cfg: 명시 cfg dict. None 이면 RunContext 의 cfg (cq.yaml.configs +
             CQ_CONFIG_JSON merge) 사용. Experiment 처럼 in-memory cfg 가
             권위 있는 경우 명시 전달.

    Returns:
        config.json Path.
    """
    ctx = _resolve_write_context(
        cfg, output_dir=output_dir, project_root=project_root
    )
    if cfg is None:
        cfg = dict(ctx.cfg)
    snapshot = dict(cfg)
    snapshot.setdefault("output_dir", str(ctx.output_dir))
    snapshot["_git_sha"] = _git_sha()
    snapshot["_pcq_version"] = _pcq_version()
    if extra:
        snapshot.update(extra)
    path = ctx.output_dir / "config.json"
    path.write_text(
        json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
    )
    return path


def save_metrics(
    history: list[dict],
    output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """epoch 별 metrics history 를 output_dir/metrics.json 에 저장.

    Args:
        history: list of dicts. 각 entry 는 {"epoch": int, ...metric_key: value}.

    Returns:
        metrics.json Path.
    """
    ctx = _resolve_write_context(
        output_dir=output_dir, project_root=project_root
    )
    path = ctx.output_dir / "metrics.json"
    path.write_text(
        json.dumps({"history": history}, indent=2), encoding="utf-8"
    )
    return path


def _autodetect_files(out_dir: Path) -> list[dict]:
    """output_dir 스캔 + extension 기반 kind heuristic."""
    files: list[dict] = []
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        if name in _SPECIAL_FILES:
            kind = _SPECIAL_FILES[name]
            if kind is None:
                # manifest.json 자기 자신 제외
                continue
        else:
            kind = _KIND_BY_SUFFIX.get(p.suffix, "other")
        files.append({"path": name, "kind": kind})
    return files


def _sha256_file(path: Path, chunk_size: int = _SHA256_CHUNK_SIZE) -> str:
    """파일의 sha256 hex digest. 1MB chunked read 로 메모리 효율 유지."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_metadata(out_dir: Path, rel_path: str) -> dict:
    """schema v2 enrich 용 sha256 + size_bytes + created_at 계산.

    파일이 존재하지 않으면 빈 dict 반환 (caller 가 안전하게 spread 가능).
    """
    full_path = out_dir / rel_path
    if not full_path.exists():
        return {}
    stat = full_path.stat()
    # mtime 기반 ISO-8601 UTC. 'Z' suffix 로 RFC3339 형태 정규화.
    created_at = (
        datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return {
        "sha256": _sha256_file(full_path),
        "size_bytes": stat.st_size,
        "created_at": created_at,
    }


def save_manifest(
    files: list | None = None,
    enrich: bool | None = None,
    output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """artifact manifest 를 output_dir/manifest.json 에 저장.

    Args:
        files: explicit list — [(path_str, kind_str), ...] 또는
               [{"path": ..., "kind": ...}, ...]. None 이면 output_dir
               자동 스캔 + heuristic kind 추정.
        enrich: True 면 schema_version=2 — 각 entry 에 sha256/size_bytes/
                created_at 추가. False 면 schema_version=1 (path + kind 만).
                None 이면 cfg["manifest_checksums"] (default True) 따름.

    Returns:
        manifest.json Path.
    """
    ctx = _resolve_write_context(
        output_dir=output_dir, project_root=project_root
    )
    out = ctx.output_dir

    # enrich 결정 — 명시값 우선, 없으면 cfg, 그것도 없으면 default True.
    if enrich is None:
        enrich = bool(ctx.cfg.get("manifest_checksums", True))

    if files is None:
        files = _autodetect_files(out)

    normalized: list[dict] = []
    for entry in files:
        if isinstance(entry, tuple):
            base: dict = {"path": str(entry[0]), "kind": str(entry[1])}
        elif isinstance(entry, dict):
            base = {
                "path": str(entry["path"]),
                "kind": str(entry.get("kind", "other")),
            }
        else:
            continue
        if enrich:
            meta = _file_metadata(out, base["path"])
            if meta:
                base.update(meta)
        normalized.append(base)

    schema_version = 2 if enrich else 1
    manifest = {"schema_version": schema_version, "files": normalized}
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _find_best(
    history: list[dict], monitor: str, mode: str
) -> dict | None:
    """history 에서 monitor key 가 있는 entry 들 중 best 선택."""
    eligible = [(i, e) for i, e in enumerate(history) if monitor in e]
    if not eligible:
        return None
    if mode == "min":
        idx, entry = min(eligible, key=lambda x: x[1][monitor])
    else:
        idx, entry = max(eligible, key=lambda x: x[1][monitor])
    return {
        "epoch": int(entry.get("epoch", idx)),
        "metrics": {
            k: float(v)
            for k, v in entry.items()
            if k != "epoch" and isinstance(v, (int, float))
        },
        "checkpoint": "best.ckpt",
    }


def _last_epoch(history: list[dict]) -> dict | None:
    """history 의 마지막 epoch entry 요약."""
    if not history:
        return None
    e = history[-1]
    return {
        "epoch": int(e.get("epoch", len(history) - 1)),
        "metrics": {
            k: float(v)
            for k, v in e.items()
            if k != "epoch" and isinstance(v, (int, float))
        },
        "checkpoint": "last.ckpt",
    }


def save_run_summary(
    history: list[dict],
    status: str = "completed",
    artifacts: dict[str, str] | None = None,
    failure: dict | None = None,
    recipe: str | None = None,
    early_stopped_at: int | None = None,
    overrides: list | None = None,
    output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """run_summary.json 저장 — best/last 자동 계산.

    monitor 와 mode 는 cfg["monitor"], cfg["mode"] 에서 인식
    (default eval_loss/min). best/last 는 history + monitor 에서 자동 도출.
    git_sha + pcq_version 은 자동 추가.

    Args:
        history: epoch 별 metric dict 리스트.
        status: "completed" | "failed" | "partial".
        artifacts: {"model": "model.pkl", ...} 형태의 artifact 경로 딕셔너리.
        failure: 실패 시 {"category": ..., "message": ..., "suggested_fix": ...}.
        recipe: recipe 이름 (cfg._recipe 가 우선).
        early_stopped_at: early stop epoch (Experiment.fit 호환).
        overrides: provenance.overrides 명시값. None 이면 cfg["_overrides"] 사용.

    Returns:
        run_summary.json Path.
    """
    ctx = _resolve_write_context(
        output_dir=output_dir, project_root=project_root
    )
    cfg = dict(ctx.cfg)
    out = ctx.output_dir
    monitor_name = cfg.get("monitor", "eval_loss")
    mode = cfg.get("mode", "min")

    best = _find_best(history, monitor_name, mode) if history else None
    last = _last_epoch(history) if history else None

    resolved_overrides = (
        list(overrides) if overrides is not None
        else list(cfg.get("_overrides", []))
    )

    # v1.17: failure category 자동 분류 (이미 명시 카테고리가 있으면 유지).
    # v2.11: failure dict 를 FailureInfo 로 정규화 — error_code/evidence 추가.
    if failure:
        from pcq.agent.failure_classifier import enrich_failure

        failure = enrich_failure(failure)
        failure = _normalize_failure(failure)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "recipe": recipe or cfg.get("_recipe"),
        "monitor": {"name": monitor_name, "mode": mode},
        "target": {"name": monitor_name, "mode": mode},
        "best": best,
        "last": last,
        "artifacts": artifacts or {},
        "provenance": {
            "git_sha": _git_sha(),
            "pcq_version": _pcq_version(),
            "overrides": resolved_overrides,
        },
        "early_stopped_at": early_stopped_at,
        "warnings": [],
        "failure": failure,
    }
    # None 제거 (단, monitor/target/artifacts/provenance/warnings 은 유지)
    summary = {k: v for k, v in summary.items() if v is not None}
    path = out / "run_summary.json"
    path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return path


def _git_dirty(cwd: Path | None = None) -> bool:
    """`git status --porcelain` 비어있지 않으면 True (dirty repo).

    git 없거나 timeout 이면 False (보수적 — clean 으로 본다).
    """
    try:
        out = _run_git(["status", "--porcelain"], cwd=cwd)
        return bool(out.stdout.strip()) if out.returncode == 0 else False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _git_changed_files(cwd: Path | None = None) -> list[str]:
    """`git status --porcelain` 기반 변경 파일 목록.

    `git diff --name-only`와 달리 untracked 파일도 포함한다. rename 은 새
    경로를 기록한다.
    """
    try:
        out = _run_git(["status", "--porcelain"], cwd=cwd)
        if out.returncode != 0:
            return []
        files: list[str] = []
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1].strip()
            if path:
                files.append(path)
        return sorted(set(files))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _git_patch_sha256(cwd: Path | None = None) -> str | None:
    """`git diff HEAD` 의 sha256 — patch identity. 변경 없으면 None."""
    try:
        out = _run_git(["diff", "HEAD"], cwd=cwd)
        if out.returncode != 0 or not out.stdout:
            return None
        return hashlib.sha256(out.stdout.encode("utf-8")).hexdigest()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


_LOCKFILE_NAMES: tuple[str, ...] = (
    "uv.lock", "poetry.lock", "pdm.lock", "conda-lock.yml",
    "Pipfile.lock", "requirements.lock",
)
# Walk up at most this many ancestor directories looking for a lockfile.
_LOCKFILE_ANCESTOR_LIMIT = 8


def _find_lockfile(start: Path | None = None) -> tuple[str, Path] | None:
    """start부터 ancestor 디렉토리를 walk-up하며 lockfile 탐색.

    project root marker(.git, pyproject.toml)에 도달하면 stop —
    nested project가 부모 project의 lockfile을 잘못 가져오지 않도록.

    Returns:
        (lockfile_name, absolute_path) 또는 None.
    """
    cwd = (start or Path.cwd()).resolve()
    ancestors = [cwd, *cwd.parents][:_LOCKFILE_ANCESTOR_LIMIT]
    for d in ancestors:
        for name in _LOCKFILE_NAMES:
            p = d / name
            if p.exists():
                return name, p
        # 같은 디렉토리에서 lockfile 없으면 project root marker 검사 후 stop.
        if (d / ".git").exists() or (d / "pyproject.toml").exists():
            return None
    return None


def _environment_snapshot(project_root: Path | None = None) -> dict:
    """python 버전 + platform + lockfile sha256 (자동 감지).

    lockfile은 cwd부터 ancestor를 walk-up하며 탐색 (uv.lock > poetry.lock >
    Pipfile.lock > requirements.lock 우선순위). project root (.git 또는
    pyproject.toml 있는 디렉토리)에 도달하면 ascent 중단 — nested project가
    부모 project의 lockfile을 잘못 가져오지 않도록.
    """
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()}-{platform.machine()}",
        "pcq_version": _pcq_version(),
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        out["torch_version"] = str(torch.__version__)
        out["cuda_available"] = cuda_available
        out["cuda_version"] = getattr(torch.version, "cuda", None)
        out["device"] = "cuda" if cuda_available else "cpu"
        if cuda_available:
            out["gpu_count"] = int(torch.cuda.device_count())
            if torch.cuda.device_count() > 0:
                out["gpu_model"] = str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 — torch import/runtime probing is best-effort
        out["device"] = "unknown"

    world_size = os.environ.get("WORLD_SIZE")
    if world_size:
        try:
            out["world_size"] = int(world_size)
        except ValueError:
            pass

    found = _find_lockfile(project_root)
    if found is not None:
        name, p = found
        out["lockfile"] = name
        out["lockfile_sha256"] = _sha256_file(p)
    return out


def _relative_path(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _source_snapshot(
    record_patch: bool = False,
    project_root: Path | None = None,
    cq_yaml_path: Path | None = None,
) -> dict:
    """git_sha + dirty (+ optional patch_sha256, changed_files)."""
    sha = _git_sha(project_root)
    dirty = _git_dirty(project_root)
    out: dict[str, Any] = {"git_sha": sha, "dirty": dirty}
    if dirty:
        out["changed_files"] = _git_changed_files(project_root)
        if record_patch:
            out["patch_sha256"] = _git_patch_sha256(project_root)
    if cq_yaml_path is not None and cq_yaml_path.exists():
        out["cq_yaml_path"] = _relative_path(cq_yaml_path, project_root)
        out["cq_yaml_sha256"] = _sha256_file(cq_yaml_path)
    return out


def _read_cq_yaml_inputs_and_metrics() -> tuple[dict, list[dict]]:
    """cq.yaml 의 inputs section + metrics schema declarations 추출.

    cq.yaml 이 없거나 파싱 실패하면 (빈 dict, []) 반환.
    Inputs 의 cq:// URI 등은 opaque 하게 그대로 유지.
    """
    cq_yaml_path = Path("cq.yaml")
    if not cq_yaml_path.exists():
        return {}, []
    try:
        from pcq.agent.yaml_io import read_yaml

        cqy = read_yaml(cq_yaml_path)
    except Exception:  # noqa: BLE001 — yaml read 다양한 예외, opaque
        return {}, []
    if not isinstance(cqy, dict):
        return {}, []

    raw_inputs = cqy.get("inputs", {})
    inputs = (
        {str(k): _flatten_yaml_value(v) for k, v in raw_inputs.items()}
        if isinstance(raw_inputs, dict)
        else {}
    )

    metrics_decl: list[dict] = []
    mraw = cqy.get("metrics", [])
    if isinstance(mraw, dict):
        for name, schema in mraw.items():
            entry: dict[str, Any] = {"name": str(name)}
            if isinstance(schema, dict):
                for k in ("mode", "split", "aggregation", "sample_count"):
                    if k in schema:
                        entry[k] = schema[k]
            metrics_decl.append(entry)
    elif isinstance(mraw, list):
        for name in mraw:
            metrics_decl.append({"name": str(name)})

    return inputs, metrics_decl


def _flatten_yaml_value(value: Any) -> Any:
    """ruamel CommentedMap/CommentedSeq → 일반 dict/list (JSON 직렬화)."""
    if isinstance(value, dict):
        return {str(k): _flatten_yaml_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_flatten_yaml_value(v) for v in value]
    return value


def _config_evidence_snapshot(rc: Any, cfg: dict, output_dir: Path) -> dict:
    """RunRecord.config evidence.

    Stores identity, not the full resolved config, so records stay compact and
    secrets accidentally injected through config are not copied again.
    """
    out: dict[str, Any] = {}
    project_root = rc.project_root
    cq_yaml_path = rc.cq_yaml_path
    if cq_yaml_path is not None and cq_yaml_path.exists():
        out["cq_yaml_path"] = _relative_path(cq_yaml_path, project_root)
        out["cq_yaml_sha256"] = _sha256_file(cq_yaml_path)
    config_json = output_dir / "config.json"
    if config_json.exists():
        out["config_json_path"] = "config.json"
        out["config_json_sha256"] = _sha256_file(config_json)
    seed = cfg.get("seed", cfg.get("random_seed"))
    if seed is not None:
        out["seed"] = seed
    if "strictness" in cfg:
        out["strictness"] = cfg.get("strictness")
    out["output_dir"] = _relative_path(output_dir, project_root)
    return out


def _input_evidence_summary(inputs: dict) -> dict:
    """Compact input identity coverage for validators and agents."""
    identity: dict[str, dict[str, bool]] = {}
    for name, meta in inputs.items():
        if not isinstance(meta, dict):
            identity[str(name)] = {
                "has_uri": False,
                "has_path": False,
                "has_sha256": False,
                "has_manifest": False,
                "opaque": False,
            }
            continue
        identity[str(name)] = {
            "has_uri": bool(meta.get("uri")),
            "has_path": bool(meta.get("path")),
            "has_sha256": bool(meta.get("sha256")),
            "has_manifest": bool(meta.get("manifest")),
            "opaque": bool(meta.get("opaque")),
        }
    return {
        "count": len(inputs),
        "names": sorted(str(k) for k in inputs),
        "identity": identity,
    }


def _now_utc_iso() -> str:
    """ISO-8601 UTC string with 'Z' suffix (RFC3339 normalized)."""
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _build_run_record(
    *,
    ctx: Any,
    status: str,
    started_at: str | None,
    finished_at: str | None,
    record_patch: bool,
    plan_id: str | None,
    intent: str | None,
    parent_run_id: str | None,
    parent_run_path: str | None,
    partial: bool,
    last_updated_at: str | None,
):
    """Build a RunRecord from the resolved context + already-written artifacts.

    Single source of truth shared by finalize_run() (status: completed/failed/
    partial) and save_partial_run_record() (status: running/checkpointed).

    Reads optional artifacts from ctx.output_dir if they exist:
      manifest.json → artifacts list
      run_summary.json → summary {target_metric, best, last} + monitor fallback

    Returns the assembled RunRecord (not yet written).
    """
    from pcq.agent.run_record import (
        AgentInfo,
        EnvironmentInfo,
        ExecutionInfo,
        MetricsInfo,
        RunInfo,
        RunRecord,
        SourceInfo,
        ValidationInfo,
    )

    rc = ctx.rc
    cfg = ctx.cfg
    out = ctx.output_dir

    # 1. run
    run_id = cfg.get("run_id") or (
        f"run_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:6]}"
    )
    # v2.3: cq.yaml top-level `name:` 도 RunInfo.name 의 fallback 으로 사용.
    # 우선순위: configs.name > resolver.name (cq.yaml top-level) > "".
    run_name = cfg.get("name") or rc.name or ""
    # v1.18 lineage: 명시 인자 > cfg._parent_run_id > cfg.parent_run_id.
    resolved_parent_id = (
        parent_run_id
        or cfg.get("_parent_run_id")
        or cfg.get("parent_run_id")
    )
    resolved_parent_path = (
        parent_run_path
        or cfg.get("_parent_run_path")
        or cfg.get("parent_run_path")
    )
    run = RunInfo(
        id=str(run_id),
        name=str(run_name) if run_name else "",
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        parent_run_id=str(resolved_parent_id) if resolved_parent_id else None,
        parent_run_path=str(resolved_parent_path) if resolved_parent_path else None,
        last_updated_at=last_updated_at,
        partial=partial,
    )

    # 2. execution
    execution = ExecutionInfo(
        cmd=str(cfg.get("_cmd") or rc.cmd or ""),
        cwd=".",
        config_path=(
            _relative_path(rc.cq_yaml_path, rc.project_root)
            if rc.cq_yaml_path is not None
            else "cq.yaml"
        ),
    )

    # 3. source
    source = SourceInfo(**_source_snapshot(
        record_patch=record_patch,
        project_root=rc.project_root,
        cq_yaml_path=rc.cq_yaml_path,
    ))

    # 4. environment
    environment = EnvironmentInfo(**_environment_snapshot(rc.project_root))

    # 5. inputs (from pcq.yaml.inputs — opaque) + metrics declared.
    inputs = dict(rc.inputs)
    input_summary = _input_evidence_summary(inputs)
    config_evidence = _config_evidence_snapshot(rc, cfg, out)
    metrics_schema_decl: list[dict] = []
    if rc.metrics_schema:
        for mname, mschema in rc.metrics_schema.items():
            entry: dict[str, Any] = {"name": str(mname)}
            for k in ("mode", "split", "aggregation", "sample_count"):
                if k in mschema:
                    entry[k] = mschema[k]
            metrics_schema_decl.append(entry)
    elif rc.declared_metrics:
        metrics_schema_decl = [{"name": n} for n in rc.declared_metrics]

    # 7. artifacts — manifest.json files entry (있으면)
    artifacts_list: list[dict] = []
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts_list = m.get("files", []) or []
        except json.JSONDecodeError:
            pass

    # 8. summary — run_summary.json 의 best/last/monitor (있으면)
    summary: dict[str, Any] = {}
    rs_path = out / "run_summary.json"
    rs_data: dict | None = None
    if rs_path.exists():
        try:
            rs = json.loads(rs_path.read_text(encoding="utf-8"))
            rs_data = rs
            target_metric = (rs.get("monitor") or {}).get("name")
            summary_raw = {
                "target_metric": target_metric,
                "best": rs.get("best"),
                "last": rs.get("last"),
            }
            summary = {k: v for k, v in summary_raw.items() if v}
        except json.JSONDecodeError:
            pass

    # 6. metrics — cq.yaml 에 declared 가 없으면 run_summary.monitor 로 합성.
    if not metrics_schema_decl and rs_data is not None:
        monitor = rs_data.get("monitor") or {}
        mname = monitor.get("name")
        mmode = monitor.get("mode")
        if mname:
            entry = {"name": str(mname)}
            if mmode in ("min", "max"):
                entry["mode"] = mmode
            metrics_schema_decl = [entry]
    metrics = MetricsInfo(
        declared=metrics_schema_decl, history_path="metrics.json"
    )

    # 9. agent
    agent = AgentInfo(
        plan_id=plan_id or cfg.get("_plan_id"),
        intent=intent or cfg.get("_plan_intent"),
        recipe=cfg.get("_recipe"),
        overrides=list(cfg.get("_overrides", []) or []),
    )

    # 10. validation — placeholder, post-validation 후 patch (finalize 시).
    validation = ValidationInfo(
        status="unknown", report_path="validation_report.json"
    )

    return RunRecord(
        run=run,
        execution=execution,
        source=source,
        environment=environment,
        config=config_evidence,
        inputs=inputs,
        input_summary=input_summary,
        metrics=metrics,
        artifacts=artifacts_list,
        summary=summary,
        agent=agent,
        validation=validation,
    )


def _atomic_write_json(target: Path, payload: dict) -> None:
    """Atomic JSON dump via tmp + os.replace.

    Readers are guaranteed to see either the previous valid JSON or the new one
    — never a half-written file. Uses unique tmp suffix to avoid concurrent
    finalize collisions.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp.{uuid.uuid4().hex[:8]}"
    tmp.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    os.replace(str(tmp), str(target))


def save_partial_run_record(
    history: list[dict] | None = None,
    *,
    output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    status: str = "running",
    intent: str | None = None,
    plan_id: str | None = None,
    parent_run_id: str | None = None,
    parent_run_path: str | None = None,
) -> Path:
    """Atomic partial dump of RunRecord while training is still in progress.

    Writes ``output_dir/run_record.json`` via tmp+rename. Sets:

      - ``run.status = status`` (default "running"; "checkpointed" also valid)
      - ``run.partial = True``
      - ``run.last_updated_at = <ISO-8601 UTC now>``

    On a subsequent ``finalize_run()`` the same file is rewritten with
    ``run.partial = False`` and ``run.status`` set to the final state.

    Concurrency: the tmp file + ``os.replace`` ensures readers always observe a
    fully-written JSON document — no partial reads. Multiple writers should
    serialize their calls; pcq does not enforce a lock.

    The ``history`` argument is accepted for API symmetry with
    ``save_run_summary()`` / ``finalize_run()`` but is not currently consumed —
    metrics are read from ``metrics.json`` if present. Pass it for forward
    compatibility.

    Args:
        history: epoch metrics so far (currently advisory).
        output_dir: explicit output_dir; resolved via RunContext otherwise.
        project_root: explicit project root; ancestor walk otherwise.
        status: "running" (default) | "checkpointed".
        intent, plan_id, parent_run_id, parent_run_path: same agent provenance
            inputs as ``finalize_run()``.

    Returns:
        Path to ``output_dir/run_record.json`` (post-write).
    """
    # 지연 import — 순환 회피.
    from pcq.agent.resolver import resolve_run_context

    if status not in ("running", "checkpointed"):
        # 다른 status 도 받지만 권고 (caller가 finalize_run을 사용해야 함).
        # 의미 흐림 방지를 위해 명시적으로 reject.
        raise ValueError(
            f"save_partial_run_record status must be 'running' or "
            f"'checkpointed' (got {status!r}); use finalize_run() for "
            f"completed/failed/partial."
        )

    ctx = resolve_run_context(
        path=project_root,
        output_dir=output_dir,
        ensure_output_dir=True,
    )
    cfg = ctx.cfg
    out = ctx.output_dir
    record_patch = bool(cfg.get("record_patch", False))
    now = _now_utc_iso()

    record = _build_run_record(
        ctx=ctx,
        status=status,
        started_at=cfg.get("_started_at"),
        finished_at=None,
        record_patch=record_patch,
        plan_id=plan_id,
        intent=intent,
        parent_run_id=parent_run_id,
        parent_run_path=parent_run_path,
        partial=True,
        last_updated_at=now,
    )

    rr_path = out / "run_record.json"
    _atomic_write_json(rr_path, record.to_dict())
    return rr_path


_VALID_ATTRIBUTION_KINDS: frozenset[str] = frozenset({"human", "agent"})


def build_attribution_object(
    operator: str | None = None,
    author_id: str | None = None,
    author_kind: str | None = None,
    committer_id: str | None = None,
    committer_kind: str | None = None,
    session_id: str | None = None,
    persona_id_author: str | None = None,
    persona_id_committer: str | None = None,
) -> dict | None:
    """attribution 객체를 생성하고 반환합니다.

    parent_run_id 패턴과 동일한 우선순위로 값을 결정합니다:
      1. 명시 인자 (explicit args)
      2. 환경변수 (CQ_ATTRIBUTION_*)
      3. auto-infer: operator 만 있으면 author=committer, kind="human"
      4. 아무것도 없으면 None 반환 (attribution 필드 생략)

    kind 검증: "human" | "agent" 이외는 ValueError.
    Phase 2 예약 필드 (signature 등)는 포함하지 않습니다.

    Args:
        operator: 작업을 수행한 사람/시스템의 식별자 (자유 문자열).
        author_id: 변경을 작성한 주체 id.
        author_kind: "human" 또는 "agent".
        committer_id: 변경을 커밋한 주체 id.
        committer_kind: "human" 또는 "agent".
        session_id: CQ 세션 id (opaque 보존).
        persona_id_author: author 의 페르소나 id.
        persona_id_committer: committer 의 페르소나 id.

    Returns:
        attribution dict (schema_version=1) 또는 None.

    Raises:
        ValueError: kind 가 "human" / "agent" 이외인 경우.
    """
    # 1단계: 명시 인자 우선, 없으면 환경변수 폴백.
    resolved_operator = operator or os.environ.get("CQ_ATTRIBUTION_OPERATOR")
    resolved_author_id = author_id or os.environ.get("CQ_ATTRIBUTION_AUTHOR_ID")
    resolved_author_kind = author_kind or os.environ.get("CQ_ATTRIBUTION_AUTHOR_KIND")
    resolved_committer_id = committer_id or os.environ.get("CQ_ATTRIBUTION_COMMITTER_ID")
    resolved_committer_kind = committer_kind or os.environ.get("CQ_ATTRIBUTION_COMMITTER_KIND")
    resolved_session_id = session_id or os.environ.get("CQ_ATTRIBUTION_SESSION_ID")
    resolved_persona_author = persona_id_author or os.environ.get("CQ_ATTRIBUTION_PERSONA_AUTHOR")
    resolved_persona_committer = persona_id_committer or os.environ.get("CQ_ATTRIBUTION_PERSONA_COMMITTER")

    # 2단계: auto-infer — operator 만 있고 author/committer 미지정이면 동일 주체로 설정.
    if resolved_operator and not resolved_author_id and not resolved_committer_id:
        resolved_author_id = resolved_operator
        resolved_committer_id = resolved_operator
        if not resolved_author_kind:
            resolved_author_kind = "human"
        if not resolved_committer_kind:
            resolved_committer_kind = "human"

    # 3단계: author/committer 가 모두 미지정이면 None 반환.
    if not resolved_author_id and not resolved_committer_id and not resolved_operator:
        return None

    # kind 기본값 — 명시 없으면 "human".
    final_author_kind: str = resolved_author_kind or "human"
    final_committer_kind: str = resolved_committer_kind or "human"

    # kind 검증.
    if final_author_kind not in _VALID_ATTRIBUTION_KINDS:
        raise ValueError(
            f"author_kind must be one of {sorted(_VALID_ATTRIBUTION_KINDS)!r}, "
            f"got {final_author_kind!r}"
        )
    if final_committer_kind not in _VALID_ATTRIBUTION_KINDS:
        raise ValueError(
            f"committer_kind must be one of {sorted(_VALID_ATTRIBUTION_KINDS)!r}, "
            f"got {final_committer_kind!r}"
        )

    return {
        "schema_version": 1,
        "author": {
            "kind": final_author_kind,
            "id": resolved_author_id or "",
            "persona_id": resolved_persona_author or None,
        },
        "committer": {
            "kind": final_committer_kind,
            "id": resolved_committer_id or "",
            "persona_id": resolved_persona_committer or None,
        },
        "operator": resolved_operator or None,
        "session_id": resolved_session_id or None,
    }


def finalize_run(
    history: list[dict] | None = None,
    status: str = "completed",
    artifacts: dict[str, str] | None = None,
    failure: dict | None = None,
    plan_id: str | None = None,
    intent: str | None = None,
    started_at: str | None = None,
    record_patch: bool | None = None,
    parent_run_id: str | None = None,
    parent_run_path: str | None = None,
    output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    attribution: dict | None = None,
) -> Path:
    """run_record.json + validation_report.json 작성.

    이미 작성된 contract artifacts (config.json, metrics.json, manifest.json,
    run_summary.json) 를 읽어 RunRecord 합성.

    Args:
        history: optional, metrics.json 없을 때 fallback (현재 직접 사용 X).
        status, artifacts, failure: run_summary 합성용 (호환성).
        plan_id, intent: agent provenance.
        started_at: started_at ISO timestamp (None 이면 비움).
        record_patch: dirty repo 시 patch_sha256 + changed_files 기록.
                       None 이면 cfg["record_patch"] 따름 (default False).
        parent_run_id: v1.18 lineage — parent run의 semantic id. 명시되면
                       cfg._parent_run_id / parent_run_id 보다 우선.
        parent_run_path: v1.18 lineage — parent output_dir 경로. 로컬 path
                         또는 cq:// URI (pcq은 opaque로만 보존).
        output_dir: v2.5 — explicit output_dir override (CLI finalize 에서
                    chdir/env tmp 트릭 제거). 없으면 RunContext 가 결정.
        project_root: v2.5 — explicit project_root override. cwd ancestor
                      walk-up 결과를 무시하고 이 디렉토리에서 cq.yaml 탐색.
        attribution: v3.0 — build_attribution_object() 반환값 또는 동일 형태
                     dict. run_record 의 attribution 필드로 그대로 전달.
                     None 이면 attribution 필드 생략.

    Returns:
        run_record.json Path.
    """
    # 지연 import — 순환 회피.
    from pcq.agent.resolver import resolve_run_context

    # v2.5: 단일 RunContext entry point — read+write semantics 통합.
    ctx = resolve_run_context(
        path=project_root,
        output_dir=output_dir,
        ensure_output_dir=True,
    )
    cfg = ctx.cfg
    out = ctx.output_dir

    # record_patch: 명시값 > cfg > default False
    if record_patch is None:
        record_patch = bool(cfg.get("record_patch", False))

    finished_at = _now_utc_iso()

    record = _build_run_record(
        ctx=ctx,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        record_patch=record_patch,
        plan_id=plan_id,
        intent=intent,
        parent_run_id=parent_run_id,
        parent_run_path=parent_run_path,
        partial=False,                   # finalize 는 항상 partial=False
        last_updated_at=finished_at,     # finalize 시점 = last_updated.
    )

    rr_path = out / "run_record.json"
    # attribution 필드를 run_record dict 에 추가 (있을 때만).
    record_dict = record.to_dict()
    if attribution is not None:
        record_dict["attribution"] = attribution
    # v2.11: atomic write — partial RunRecord 와 동일한 보장.
    _atomic_write_json(rr_path, record_dict)

    # post-run validation 실행 후 validation_report.json + run_record.validation patch
    _run_post_validation_and_patch(out, rr_path, strictness=cfg.get("strictness"))

    return rr_path


def _run_post_validation_and_patch(
    output_dir: Path, run_record_path: Path, strictness: int | None = None
) -> None:
    """post-run validation 실행 → validation_report.json + run_record.validation 갱신."""
    from pcq.agent.validate_run import validate_run

    report = validate_run(output_dir, strictness=strictness)
    vr_path = output_dir / "validation_report.json"
    vr_path.write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    # run_record.validation 갱신.
    try:
        rr = json.loads(run_record_path.read_text(encoding="utf-8"))
        rr["validation"] = {
            "status": report.status,
            "report_path": "validation_report.json",
        }
        run_record_path.write_text(
            json.dumps(rr, indent=2), encoding="utf-8"
        )
    except (json.JSONDecodeError, OSError):
        pass


def save_all(
    history: list[dict],
    status: str = "completed",
    artifacts: dict[str, str] | None = None,
    files: list | None = None,
    failure: dict | None = None,
    recipe: str | None = None,
    config_extra: dict | None = None,
    early_stopped_at: int | None = None,
    overrides: list | None = None,
    finalize: bool = True,
    plan_id: str | None = None,
    intent: str | None = None,
    parent_run_id: str | None = None,
    parent_run_path: str | None = None,
    output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    operator: str | None = None,
    author_id: str | None = None,
    author_kind: str | None = None,
    committer_id: str | None = None,
    committer_kind: str | None = None,
    session_id: str | None = None,
    persona_id_author: str | None = None,
    persona_id_committer: str | None = None,
) -> dict[str, Path]:
    """5+ 개 표준 artifact 묶음 작성. contract script 마지막 한 줄로 사용.

    순서: config → metrics → run_summary → manifest → (finalize) run_record
    + validation_report.

    v1.16: finalize=True (default) 면 run_record.json + validation_report.json
    까지 생성. False 로 하면 v1.15 동작 유지.
    v1.18: parent_run_id / parent_run_path 인자 — finalize_run 으로 그대로 전달.
    v2.5: output_dir / project_root 명시 인자 — finalize_run 으로 그대로 전달.
    v3.0: attribution 인자 — build_attribution_object() 로 결정 후 finalize_run 에 전달.
          명시 인자 없으면 환경변수(CQ_ATTRIBUTION_*) 자동 폴백. 둘 다 없으면 생략.

    Returns:
        {"config", "metrics", "manifest", "run_summary"} 항상.
        finalize=True 면 추가로 {"run_record", "validation_report"}.
    """
    paths: dict[str, Path] = {
        "config": save_config_snapshot(
            extra=config_extra,
            output_dir=output_dir,
            project_root=project_root,
        ),
        "metrics": save_metrics(
            history, output_dir=output_dir, project_root=project_root
        ),
        "run_summary": save_run_summary(
            history=history,
            status=status,
            artifacts=artifacts,
            failure=failure,
            recipe=recipe,
            early_stopped_at=early_stopped_at,
            overrides=overrides,
            output_dir=output_dir,
            project_root=project_root,
        ),
        "manifest": save_manifest(
            files=files, output_dir=output_dir, project_root=project_root
        ),
    }
    if finalize:
        # v3.0: attribution 객체를 한 번만 결정한 뒤 finalize_run 에 전달.
        attribution_obj = build_attribution_object(
            operator=operator,
            author_id=author_id,
            author_kind=author_kind,
            committer_id=committer_id,
            committer_kind=committer_kind,
            session_id=session_id,
            persona_id_author=persona_id_author,
            persona_id_committer=persona_id_committer,
        )
        rr_path = finalize_run(
            history=history,
            status=status,
            artifacts=artifacts,
            failure=failure,
            plan_id=plan_id,
            intent=intent,
            parent_run_id=parent_run_id,
            parent_run_path=parent_run_path,
            output_dir=output_dir,
            project_root=project_root,
            attribution=attribution_obj,
        )
        paths["run_record"] = rr_path
        vr_path = rr_path.parent / "validation_report.json"
        if vr_path.exists():
            paths["validation_report"] = vr_path
    return paths
