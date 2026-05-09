"""pcq.agent.resolver — ResolvedConfig + RunContext (cq.yaml interpretation SSOT).

v2.5: cq.yaml 해석을 라이브러리 전체의 공통 런타임 컨텍스트로 승격.

  cq.yaml + CQ_CONFIG_JSON + env
        ↓
  ResolvedConfig  (read-only, no mkdir, no chdir)
        ↓
  RunContext      (write-side; output_dir mkdir is ONLY here)
        ↓
  consumers (contract, core, Trainer, CLI inspect/validate/finalize)

Resolution priority (정확히 이대로):

| 대상 | 우선순위 |
|------|---------|
| project_root | explicit arg → `_cq_project_root` env → cq.yaml parent → cwd ancestor walk-up |
| cfg | cq.yaml.configs + CQ_CONFIG_JSON merge (env wins) |
| output_dir | explicit arg → CQ_CONFIG_JSON.output_dir → cq.yaml.configs.output_dir → project_root/output |
| relative output_dir | 항상 project_root 기준 (cwd 기준 금지) |
| run.name | cfg._run_name 또는 cfg.name → cq.yaml.name |
| execution.cmd | cfg._cmd → cq.yaml.cmd |
| declared metrics | CQ_DECLARED_METRICS → cq.yaml.metrics → cfg._metrics_declared |
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


# cq.yaml 파일명 후보. 우선순위는 list 순서.
_CQ_YAML_NAMES: tuple[str, ...] = ("cq.yaml", "pcq.yml")
# Walk up at most this many ancestor directories looking for cq.yaml.
_CQ_YAML_ANCESTOR_LIMIT = 8


@dataclass
class ResolvedConfig:
    """Normalized read-only view of an experiment project.

    **Read-side**: pure. No mkdir, no chdir, no env mutation.
    To create artifacts (mkdir output_dir), use ``RunContext`` instead.

    Single source of truth: agents/CLI commands consult this dataclass
    rather than re-parsing cq.yaml in each function. Constructed by
    resolve_project().
    """

    schema_version: int = 1
    project_root: Path | None = None  # absolute path to project root
    cq_yaml_path: Path | None = None  # absolute path to cq.yaml (or None if env-only)
    name: str = ""
    cmd: str = ""
    cfg: dict[str, Any] = field(default_factory=dict)  # cq.yaml.configs (normalized)
    declared_metrics: list[str] = field(default_factory=list)  # always list[str]
    metrics_schema: dict[str, dict] = field(default_factory=dict)  # dict-style metrics if present
    artifacts: list[str] = field(default_factory=list)  # cq.yaml.artifacts (list of glob strings)
    inputs: dict[str, dict] = field(default_factory=dict)  # cq.yaml.inputs (opaque)
    output_dir: Path | None = None  # absolute path; NOT created (read-only)
    parse_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-safe dict — None / 빈 컨테이너는 제거 (출력 가독성)."""
        out = {
            "schema_version": self.schema_version,
            "project_root": str(self.project_root) if self.project_root else None,
            "cq_yaml_path": str(self.cq_yaml_path) if self.cq_yaml_path else None,
            "name": self.name,
            "cmd": self.cmd,
            "cfg": self.cfg,
            "declared_metrics": self.declared_metrics,
            "metrics_schema": self.metrics_schema,
            "artifacts": self.artifacts,
            "inputs": self.inputs,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "parse_errors": self.parse_errors,
            "warnings": self.warnings,
        }
        # None / "" / [] / {} 값은 출력에서 제거 — schema_version만 항상 노출.
        return {
            k: v
            for k, v in out.items()
            if k == "schema_version" or v not in (None, "", [], {})
        }


@dataclass
class RunContext:
    """Write-time context — the *only* API path that creates output_dir.

    Built by ``resolve_run_context(ensure_output_dir=True)``. Wraps a
    ResolvedConfig with the write-side mkdir semantics. Every artifact
    helper (save_*, finalize_run, Experiment.fit) flows through this.

    Why split? Read-side (inspect, validate, resolve) must not mutate
    the filesystem — agents inspect projects without producing side
    effects. Write-side mkdir is concentrated in ONE location so the
    invariant "output_dir exists when artifacts get written" is local.
    """

    rc: ResolvedConfig

    @property
    def project_root(self) -> Path:
        return self.rc.project_root or Path.cwd().resolve()

    @property
    def output_dir(self) -> Path:
        # output_dir 은 resolve_run_context 가 ensure 했음. None 인 경우는
        # ensure_output_dir=False 로 호출됐을 때만 — 이 경우 fallback.
        return self.rc.output_dir or (self.project_root / "output")

    @property
    def cfg(self) -> dict:
        return self.rc.cfg

    @property
    def name(self) -> str:
        return self.rc.name

    @property
    def cmd(self) -> str:
        return self.rc.cmd

    @property
    def declared_metrics(self) -> list[str]:
        return self.rc.declared_metrics

    @property
    def metrics_schema(self) -> dict[str, dict]:
        return self.rc.metrics_schema

    @property
    def inputs(self) -> dict[str, dict]:
        return self.rc.inputs

    def artifact_path(self, name: str) -> Path:
        return self.output_dir / name


def _flatten_yaml_value(value: Any) -> Any:
    """ruamel CommentedMap/CommentedSeq → 일반 dict/list (JSON 직렬화).

    재귀적으로 nested 구조 평탄화.
    """
    if isinstance(value, dict):
        return {str(k): _flatten_yaml_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_flatten_yaml_value(v) for v in value]
    return value


def _walk_up_for_cq_yaml(start: Path) -> Path | None:
    """cwd부터 ancestor (max 8 levels) 탐색하며 cq.yaml 찾기.

    project root marker (.git, pyproject.toml) 만나면 stop —
    nested project가 부모 cq.yaml 잡지 않도록.

    같은 디렉토리에서 cq.yaml과 root marker가 둘 다 있으면 cq.yaml 채택.
    """
    cur = start.resolve()
    for d in [cur, *cur.parents][:_CQ_YAML_ANCESTOR_LIMIT]:
        for name in _CQ_YAML_NAMES:
            p = d / name
            if p.exists():
                return p.resolve()
        # cq.yaml 없는데 project root marker 있으면 ascent stop.
        if (d / ".git").exists() or (d / "pyproject.toml").exists():
            return None
    return None


def _normalize_metrics(raw: Any) -> tuple[list[str], dict[str, dict]]:
    """list-style 또는 dict-style metrics를 (declared_metrics, metrics_schema)로.

    list  → ([m1, m2, ...], {})
    dict  → (sorted([keys]), {k: {schema}, ...})
    other → ([], {})
    """
    if isinstance(raw, list):
        return [str(m) for m in raw], {}
    if isinstance(raw, dict):
        names = sorted(str(k) for k in raw.keys())
        schema: dict[str, dict] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                schema[str(k)] = _flatten_yaml_value(v)
            else:
                schema[str(k)] = {"value": _flatten_yaml_value(v)}
        return names, schema
    return [], {}


def _normalize_artifacts(raw: Any) -> list[str]:
    """artifacts: list[str] 또는 dict {path: meta} → list[str]."""
    if isinstance(raw, list):
        return [str(a) for a in raw]
    if isinstance(raw, dict):
        return sorted(str(k) for k in raw.keys())
    return []


def _normalize_inputs(raw: Any) -> dict[str, dict]:
    """inputs section을 {name: dict} 로 정규화. cq URI 등은 opaque."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[str(k)] = _flatten_yaml_value(v)
        else:
            out[str(k)] = {"value": _flatten_yaml_value(v)}
    return out


def _resolve_output_dir_value(
    raw: Any, project_root: Path
) -> Path:
    """raw output_dir → 절대경로. project_root 기준 (이미 절대면 그대로)."""
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def resolve_project(
    path: str | Path | None = None,
    cq_yaml_path: str | Path | None = None,
) -> ResolvedConfig:
    """Pure read of cq.yaml + CQ_CONFIG_JSON env. NO mkdir, NO chdir.

    To actually create artifacts (output_dir mkdir), use
    ``resolve_run_context(ensure_output_dir=True)`` instead.

    Resolution strategy:
      1. cq_yaml_path 명시 → 그대로 사용. project_root = its parent.
      2. path 명시 → project_root로 간주, 안에서 cq.yaml 검색.
      3. 둘 다 None → cwd, ancestor walk-up (max 8 levels).
         _cq_project_root env 도 인식 (테스트/명시 wiring 용).

    CQ_CONFIG_JSON env, when present, is merged INTO cfg (env wins).
    No cq.yaml found → ResolvedConfig with project_root=cwd, cfg from env-only.

    Args:
        path: project root 디렉토리. None이면 cwd ancestor walk-up.
        cq_yaml_path: 명시 cq.yaml 위치. project_root는 부모 디렉토리.

    Returns:
        ResolvedConfig — 정규화된 단일 view (no side effects).
    """
    rc = ResolvedConfig()

    # 1. project_root + cq.yaml 위치 결정
    env_root = os.environ.get("_cq_project_root")

    if cq_yaml_path is not None:
        cq_path = Path(cq_yaml_path).expanduser().resolve()
        if cq_path.exists():
            rc.cq_yaml_path = cq_path
            rc.project_root = cq_path.parent
        else:
            rc.parse_errors.append(f"cq_yaml_path not found: {cq_path}")
            rc.project_root = (
                Path(env_root).resolve()
                if env_root
                else Path.cwd().resolve()
            )
    elif path is not None:
        proj = Path(path).expanduser().resolve()
        rc.project_root = proj
        for name in _CQ_YAML_NAMES:
            p = proj / name
            if p.exists():
                rc.cq_yaml_path = p.resolve()
                break
    elif env_root:
        # _cq_project_root env: explicit project_root injection (서비스 워커 등).
        proj = Path(env_root).expanduser().resolve()
        rc.project_root = proj
        for name in _CQ_YAML_NAMES:
            p = proj / name
            if p.exists():
                rc.cq_yaml_path = p.resolve()
                break
    else:
        cwd = Path.cwd().resolve()
        cq_path = _walk_up_for_cq_yaml(cwd)
        if cq_path:
            rc.cq_yaml_path = cq_path
            rc.project_root = cq_path.parent
        else:
            rc.project_root = cwd

    # 2. cq.yaml 파싱 (P2 #4: malformed → parse_errors, NOT silent pass)
    cq_data: dict = {}
    if rc.cq_yaml_path is not None:
        try:
            from pcq.agent.yaml_io import read_yaml

            loaded = read_yaml(rc.cq_yaml_path)
            if isinstance(loaded, dict):
                cq_data = loaded
            else:
                rc.parse_errors.append(
                    f"cq.yaml top-level must be a mapping, got "
                    f"{type(loaded).__name__}"
                )
        except Exception as e:  # noqa: BLE001 — yaml read 다양한 예외, opaque
            rc.parse_errors.append(
                f"yaml read failed: {type(e).__name__}: {e}"
            )

    raw_name = cq_data.get("name")
    rc.name = raw_name if isinstance(raw_name, str) else ""
    raw_cmd = cq_data.get("cmd")
    rc.cmd = raw_cmd if isinstance(raw_cmd, str) else ""

    # 3. configs section
    raw_configs = cq_data.get("configs", {})
    rc.cfg = (
        _flatten_yaml_value(raw_configs)
        if isinstance(raw_configs, dict)
        else {}
    )

    # 4. CQ_CONFIG_JSON merge — env가 cq.yaml.configs를 override
    cfg_path_env = os.environ.get("CQ_CONFIG_JSON")
    if cfg_path_env:
        try:
            with open(cfg_path_env, encoding="utf-8") as f:
                env_cfg = json.load(f)
            if isinstance(env_cfg, dict):
                # env가 cq.yaml.configs를 override (env가 always-on 학습 시 주입되므로)
                rc.cfg.update(env_cfg)
        except (OSError, json.JSONDecodeError) as e:
            rc.warnings.append(
                f"CQ_CONFIG_JSON read failed: {type(e).__name__}: {e}"
            )

    # 5. metrics — list/dict 정규화
    rc.declared_metrics, rc.metrics_schema = _normalize_metrics(
        cq_data.get("metrics")
    )
    # cfg["_metrics_declared"]가 있으면 declared_metrics fallback (CQ_CONFIG_JSON 경유)
    if not rc.declared_metrics:
        cfg_decl = rc.cfg.get("_metrics_declared")
        if isinstance(cfg_decl, list):
            rc.declared_metrics = [str(m) for m in cfg_decl]

    # 6. artifacts / inputs
    rc.artifacts = _normalize_artifacts(cq_data.get("artifacts"))
    rc.inputs = _normalize_inputs(cq_data.get("inputs"))

    # 7. output_dir — cfg 기반 절대경로 (project_root 기준).
    # NO mkdir here — read-side는 부수효과 없음. RunContext가 mkdir 책임.
    raw_out = rc.cfg.get("output_dir", "output")
    rc.output_dir = _resolve_output_dir_value(
        raw_out, rc.project_root or Path.cwd().resolve()
    )

    return rc


def resolve_run_context(
    path: str | Path | None = None,
    cq_yaml_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    ensure_output_dir: bool = True,
) -> RunContext:
    """Write-side context — the ONLY mkdir owner for output_dir.

    Resolves a project (via resolve_project) then optionally creates the
    output_dir. Pass ``ensure_output_dir=False`` to skip mkdir (rare;
    inspect/validate use ``resolve_project`` directly instead).

    Args:
        path: project root.
        cq_yaml_path: explicit cq.yaml location.
        output_dir: explicit output_dir override (highest priority).
        ensure_output_dir: if True (default), mkdir parents=True.

    Returns:
        RunContext wrapping a ResolvedConfig.
    """
    rc = resolve_project(path=path, cq_yaml_path=cq_yaml_path)

    # explicit output_dir override
    if output_dir is not None:
        proj_root = rc.project_root or Path.cwd().resolve()
        new_out = _resolve_output_dir_value(output_dir, proj_root)
        rc = replace(rc, output_dir=new_out)

    if ensure_output_dir and rc.output_dir is not None:
        rc.output_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(rc=rc)
