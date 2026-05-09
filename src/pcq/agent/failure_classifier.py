"""pcq.agent.failure_classifier — failure category 자동 분류.

agent 가 RunRecord/run_summary 만 보고 다음 행동을 결정할 수 있도록,
실패 메시지에서 카테고리를 휴리스틱하게 추출한다.

Categories (RUN_RECORD.md §"Result Semantics For Agents"):
  config_error / missing_dependency / dataset_missing / dataset_shape /
  label_contract / loss_contract / metric_contract / oom / nan_loss /
  timeout / distributed_write_race / unknown_exception
"""
from __future__ import annotations

import re

# (regex, category) — 위에서 아래로 평가, 첫 매칭 카테고리 반환.
# 더 구체적인 패턴이 위로 와야 한다.
_PATTERNS: list[tuple[str, str]] = [
    (r"out of memory|OOM|CUDA error.*memory|cuda.*out of memory", "oom"),
    (r"\bNaN\b|nan_loss|loss is NaN|loss.*nan|inf.*loss", "nan_loss"),
    (r"ModuleNotFoundError|ImportError.*No module", "missing_dependency"),
    (r"FileNotFoundError.*data|dataset.*not found|no such.*dataset", "dataset_missing"),
    (r"size mismatch|shape.*mismatch|expected.*got.*shape|dimension.*mismatch", "dataset_shape"),
    (r"ignore_index.*mismatch|target.*out of range|label.*out of range", "label_contract"),
    (r"loss.*contract|loss.*signature", "loss_contract"),
    (r"metric.*contract|metric.*signature|undeclared metric", "metric_contract"),
    (r"\btimeout\b|TimeoutError", "timeout"),
    (r"FileExistsError.*manifest|race condition|concurrent write", "distributed_write_race"),
    (r"CQ_CONFIG_JSON|cfg.*missing|config.*invalid", "config_error"),
]


def classify_failure(message: str) -> str:
    """error message → category (휴리스틱).

    매칭 안 되면 'unknown_exception'.
    """
    if not message:
        return "unknown_exception"
    for pattern, category in _PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            return category
    return "unknown_exception"


def enrich_failure(failure: dict | None) -> dict | None:
    """기존 failure dict 에 category 미설정/unknown 시 자동 분류 채움.

    이미 명시 카테고리가 있고 'unknown_exception' 이 아니면 유지.
    """
    if not failure:
        return failure
    existing = failure.get("category")
    if existing and existing != "unknown_exception":
        return failure
    enriched = dict(failure)
    enriched["category"] = classify_failure(failure.get("message", ""))
    return enriched
