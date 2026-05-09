"""pcq.core — cq.yaml 런타임 컨트랙트 어댑터.

5개 공개 함수:
- config()         : CQ_CONFIG_JSON 파싱
- log(**values)    : stdout @key=value 출력 + 미선언 메트릭 경고
- output_dir()     : 출력 디렉토리 생성/반환
- input_dir(name)  : worker가 fetch한 데이터 디렉토리 경로
- seed_everything(): random/numpy/torch 시드 고정
"""

from __future__ import annotations

import atexit
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# 미선언 메트릭 추적 (atexit 요약용)
_undeclared_warned: set[str] = set()
_undeclared_count: Counter[str] = Counter()
_atexit_registered = False

# Declared metrics 1회 캐시 (env/CQ_CONFIG_JSON 둘 다 시도)
_declared_cache: set[str] | None = None
_declared_cache_loaded: bool = False


def _ensure_atexit() -> None:
    # atexit 핸들러를 한 번만 등록한다.
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_print_undeclared_summary)
        _atexit_registered = True


def _print_undeclared_summary() -> None:
    # 프로세스 종료 시 누적된 미선언 메트릭 요약을 stderr로 출력한다.
    if _undeclared_count:
        total = sum(_undeclared_count.values())
        print(
            f"[cq] {len(_undeclared_count)} undeclared metric key(s) suppressed "
            f"({total} call(s)): {dict(_undeclared_count)}",
            file=sys.stderr,
        )


def _read_declared_metrics() -> set[str] | None:
    """cq.yaml.metrics 선언 목록을 결정한다.

    우선순위:
      1. CQ_DECLARED_METRICS 환경변수 (콤마 구분).
      2. CQ_CONFIG_JSON 이 가리키는 JSON 파일의 _metrics_declared 키.

    1회만 디스크에서 로드 (cached). 둘 다 없으면 None — 검증 skip.
    테스트는 _reset_declared_cache() 로 캐시 리셋.
    """
    global _declared_cache, _declared_cache_loaded
    if _declared_cache_loaded:
        return _declared_cache

    env_val = os.environ.get("CQ_DECLARED_METRICS")
    if env_val:
        _declared_cache = set(env_val.split(","))
        _declared_cache_loaded = True
        return _declared_cache

    # CQ_CONFIG_JSON 자동 로드 — _metrics_declared 키 검색
    cfg_path = os.environ.get("CQ_CONFIG_JSON")
    if cfg_path and os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            declared = cfg.get("_metrics_declared")
            if declared is not None:
                _declared_cache = set(declared)
                _declared_cache_loaded = True
                return _declared_cache
        except (json.JSONDecodeError, OSError):
            pass

    _declared_cache_loaded = True  # None 으로 cache → 다음 호출도 None
    return None


def _reset_declared_cache() -> None:
    """테스트용 declared metrics 캐시 리셋."""
    global _declared_cache, _declared_cache_loaded
    _declared_cache = None
    _declared_cache_loaded = False


def config() -> dict:
    """런타임 cfg dict 반환.

    우선순위 (v2.12: cq.yaml fallback 추가):
      1. CQ_CONFIG_JSON env 가 가리키는 JSON 파일 (worker invocation 표준).
      2. cq.yaml.configs (pcq.agent.resolver.resolve_project; cwd ancestor walk-up).
         env 와 cq.yaml 이 둘 다 있을 때는 resolver 가 둘을 merge — env wins.
      3. 둘 다 없으면 RuntimeError.

    fresh-user 가 `python train.py` 직접 실행할 때 (1) 없이 (2) 만으로 동작.
    PlanSet expand 시 N 개 dir 마다 manual env wiring 불필요.

    Raises:
        RuntimeError: env 미설정 + cq.yaml 도 못 찾음.
    """
    path = os.environ.get("CQ_CONFIG_JSON")
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # env 부재 → cq.yaml fallback (v2.5 ResolvedConfig 약속 이행).
    try:
        from pcq.agent.resolver import resolve_project
    except ImportError as e:  # 극단적 임포트 사이클 — 명시적으로 raise.
        raise RuntimeError(
            "CQ_CONFIG_JSON not set and resolver unavailable"
        ) from e

    rc = resolve_project()
    if rc.cq_yaml_path is None:
        raise RuntimeError(
            "no cq.yaml found and CQ_CONFIG_JSON not set "
            "(set CQ_CONFIG_JSON or run from a project with cq.yaml)"
        )
    # rc.cfg 는 cq.yaml.configs (없으면 빈 dict). env merge 는 resolver 가 처리하지만
    # 이 경로에서는 env 없음 — 그대로 반환.
    return dict(rc.cfg)


def _is_finite_numeric(v: Any) -> bool:
    """bool/string/NaN/inf 제외, 유한 numeric만 True.

    bool은 isinstance(bool, int)이 True이므로 명시적으로 먼저 거른다.
    """
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return math.isfinite(float(v))
    # numpy scalar (np.float32, np.int64 등)
    if isinstance(v, np.floating):
        return math.isfinite(float(v))
    if isinstance(v, np.integer):
        return True
    # 0-d numpy array 같은 케이스 — item()으로 시도
    if hasattr(v, "item") and hasattr(v, "shape") and getattr(v, "shape", None) == ():
        try:
            f = float(v.item())
            return math.isfinite(f)
        except (ValueError, TypeError):
            return False
    return False


def _format_value(value: Any) -> str:
    # numpy scalar는 Python 스칼라로 변환
    if hasattr(value, "item") and not isinstance(value, (int, float, bool)):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float):
        # %g는 trailing zero 제거 + 6자리 유효숫자. 0.42 -> "0.42", 1e-7 -> "1e-07".
        return f"{value:g}"
    return str(value)


def log(strict: bool = False, **values: Any) -> None:
    """finite numeric 값만 stdout `@key=value` 형식 출력.

    Args:
        strict: True면 미선언 key 즉시 RuntimeError. False면 첫 1회만 stderr 경고.
        **values: 메트릭 key=value (bool/str/NaN/inf는 silently 제외).
    """
    _ensure_atexit()
    declared = _read_declared_metrics()

    parts: list[str] = []
    for key, value in values.items():
        if not _is_finite_numeric(value):
            continue
        if declared is not None and key not in declared:
            if strict:
                raise RuntimeError(f"undeclared metric key (strict mode): {key!r}")
            _undeclared_count[key] += 1
            if key not in _undeclared_warned:
                _undeclared_warned.add(key)
                print(
                    f"[cq] warning: undeclared metric key {key!r} "
                    f"(declared: {sorted(declared)})",
                    file=sys.stderr,
                )
            continue
        parts.append(f"@{key}={_format_value(value)}")

    if parts:
        print(" ".join(parts))


def output_dir() -> Path:
    """cfg["output_dir"] 또는 "output" 디렉토리를 mkdir 후 Path 반환."""
    try:
        cfg = config()
        path = cfg.get("output_dir", "output")
    except RuntimeError:
        # CQ_CONFIG_JSON 없으면 기본값으로 fallback (로컬 개발 편의)
        path = "output"
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def input_dir(name: str) -> Path:
    """Worker가 fetch한 데이터 디렉토리 경로를 반환한다.

    우선순위:
      1. env CQ_INPUT_DIR_<NAME.upper()>
      2. cfg["inputs"][name]
    둘 다 없으면 FileNotFoundError.
    """
    env_key = f"CQ_INPUT_DIR_{name.upper()}"
    env_val = os.environ.get(env_key)
    if env_val:
        return Path(env_val)
    try:
        cfg = config()
        inputs = cfg.get("inputs", {})
        if name in inputs:
            return Path(inputs[name])
    except RuntimeError:
        # CQ_CONFIG_JSON 없는 환경에서는 env만 본다
        pass
    raise FileNotFoundError(
        f"input '{name}' not found (checked env {env_key} and cfg['inputs'][{name!r}])"
    )


def seed_everything(seed: int) -> None:
    """random/numpy/torch (cuda 포함) 시드 고정."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # if self.cfg.get("deterministic", True):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        # torch 미설치 환경 (core only)에서는 건너뛴다
        pass
