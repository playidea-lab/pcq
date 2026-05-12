"""pcq.core — cq.yaml 런타임 컨트랙트 어댑터.

6개 공개 함수:
- config()         : CQ_CONFIG_JSON 파싱
- log(**values)    : stdout @key=value 출력 + 미선언 메트릭 경고
- output_dir()     : 출력 디렉토리 생성/반환
- input_dir(name)  : worker가 fetch한 데이터 디렉토리 경로
- seed_everything(): random/numpy/torch 시드 고정
- worker_spec()    : 현재 실행 환경의 worker_spec dict 반환 (T-WSPEC-5)
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


def worker_spec() -> dict | None:
    """현재 실행 환경의 worker_spec dict 를 반환한다 (T-WSPEC-5).

    save_all() 이 내부적으로 자동 호출하므로 사용자는 직접 호출할 필요가 없다.
    build_worker_spec_object 를 env + cfg 기반으로 호출하고 결과 dict 만 반환.
    감지 실패 시 None 반환.
    """
    from pcq.contract import build_worker_spec_object

    try:
        cfg = config()
    except RuntimeError:
        # CQ_CONFIG_JSON 없는 환경에서는 cfg 없이 자동 감지만 수행
        cfg = {}
    spec, _ = build_worker_spec_object(cli_args=None, cfg=cfg)
    return spec


# ---------------------------------------------------------------------------
# T-WFP-5: pcq.fingerprint() — 데이터 핑거프린트 공개 API
# ---------------------------------------------------------------------------

# 모듈 레벨 캐시 (save_all 자동 픽업용). None 이면 호출된 적 없음.
_fingerprint_cache: dict | None = None
_fingerprint_warnings: list[dict] = []

# 규제 도메인 — R5 게이트 (fingerprint.py와 동일 정의)
_REGULATED_DOMAINS: frozenset[str] = frozenset({"medical", "financial", "regulated"})

# 허용 모달리티 (R12) — contract._VALID_FINGERPRINT_MODALITIES와 동일
_VALID_MODALITIES: frozenset[str] = frozenset({
    "tabular", "image", "text", "time_series", "audio", "graph", "other",
})


def fingerprint(
    X: Any,
    y: Any,
    modality: str,
    task_kind: str | None = None,
    domain: str = "general",
    sample_rows: int = 100_000,
) -> dict | None:
    """데이터 핑거프린트를 추출하고 모듈 캐시에 저장한다 (T-WFP-5).

    save_all() 이 캐시를 자동으로 픽업하여 run_record.json 의 fingerprint 섹션을 채운다.
    사용자는 save_all() 이전에 이 함수를 한 번 호출하면 된다.

    Args:
        X: 입력 데이터 (DataFrame, ndarray, list, dict 등 모달리티에 따라 다름).
        y: 타겟 레이블 또는 None.
        modality: 데이터 모달리티 — tabular|image|text|time_series|audio|graph|other (R12).
        task_kind: 태스크 종류 — classification|regression|... (선택).
        domain: 도메인 컨텍스트 — general|medical|financial|regulated|other.
                medical/financial/regulated 이면 R5 게이트 적용으로 추출 생략.
        sample_rows: 대용량 tabular 데이터 샘플링 한계 (기본 100_000).

    Returns:
        캐시된 detected_cache dict (save_all 픽업용). caller 가 직접 사용 가능.
        R5 게이트 적용 시 hints-only dict 반환.

    Raises:
        ValueError: modality 가 유효하지 않은 경우 (R12).
    """
    global _fingerprint_cache, _fingerprint_warnings

    # R12: modality enum 검증
    if modality not in _VALID_MODALITIES:
        raise ValueError(
            f"modality must be one of {sorted(_VALID_MODALITIES)!r}, got {modality!r}"
        )

    warnings: list[dict] = []

    # R5: 규제 도메인이면 추출 생략 → hints only + FINGERPRINT_DOMAIN_GATE_SKIP 경고
    if domain in _REGULATED_DOMAINS:
        hints_cache: dict = {
            "modality": modality,
            "task_kind": task_kind,
            "domain": domain,
            "sampled": False,
            "warnings": [{
                "code": "FINGERPRINT_DOMAIN_GATE_SKIP",
                "message": (
                    f"domain={domain!r}은 규제 도메인입니다. "
                    "자동 통계 추출을 생략하고 declared 경로를 사용합니다. "
                    "cq.yaml 의 fingerprint 섹션에 메타데이터를 명시하세요."
                ),
            }],
        }
        _fingerprint_cache = hints_cache
        _fingerprint_warnings = hints_cache["warnings"]
        print(
            "[cq] warning: FINGERPRINT_DOMAIN_GATE_SKIP — "
            f"domain={domain!r} 규제 도메인으로 자동 추출 비활성.",
            file=sys.stderr,
        )
        return hints_cache

    # 모달리티별 추출 함수 디스패치
    from pcq.fingerprint import (
        extract_audio,
        extract_graph,
        extract_image,
        extract_tabular,
        extract_text,
        extract_time_series,
    )

    _extractor_map = {
        "tabular": lambda: extract_tabular(X, y, sample_rows=sample_rows, domain=domain),
        "image": lambda: extract_image(X, y),
        "text": lambda: extract_text(X, y),
        "time_series": lambda: extract_time_series(X, y),
        "audio": lambda: extract_audio(X, y),
        "graph": lambda: extract_graph(X, y),
        "other": lambda: ({}, []),  # other 는 추출 없음 (hints only)
    }

    extractor = _extractor_map[modality]
    sub_dict, sub_warnings = extractor()
    warnings.extend(sub_warnings)

    # n_samples / size_class 결정
    n_samples: int | None = None
    sampled = False
    try:
        n_samples = int(len(X))  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        pass

    # size_class 계산 (fingerprint._size_class 재사용)
    size_class: str | None = None
    if n_samples is not None:
        from pcq.fingerprint import _size_class
        size_class = _size_class(n_samples)

    # sampled 여부 — sub_dict 에 sampled_rows 가 있으면 샘플링된 것
    if sub_dict.get("sampled_rows") is not None:
        sampled = True

    # detected_cache 조립 (build_fingerprint_object 가 읽는 형식)
    detected_cache: dict = {
        "modality": modality,
        "task_kind": task_kind,
        "domain": domain,
        "n_samples": n_samples,
        "size_class": size_class,
        "sampled": sampled,
        "warnings": warnings,
    }
    # 모달리티 서브객체 추가 (예: "tabular": {...})
    if sub_dict:
        detected_cache[modality] = sub_dict

    # 캐시 저장 (save_all 픽업용)
    _fingerprint_cache = detected_cache
    _fingerprint_warnings = warnings

    return detected_cache


def _reset_fingerprint_cache() -> None:
    """테스트용 fingerprint 캐시 리셋."""
    global _fingerprint_cache, _fingerprint_warnings
    _fingerprint_cache = None
    _fingerprint_warnings = []
