"""pcq.metric — reference example metric atoms + opt-in stateful metrics.

Contract examples (``accuracy``, ``top_5_accuracy``, ``mse``, ``mae``,
``iou``, ``dice_score``, ``pixel_accuracy``) for the metric atom kind. NOT a
production metric catalog. Project-local metrics live in ``atoms/metrics.py``
via ``pcq.register_metric``.

기본은 functional 모듈의 함수형 metric. (logits, labels) → 스칼라 tensor.
Experiment._run_epoch 가 batch 단위 평균으로 집계 (mean | weighted_mean).

정확한 sample-weighted 집계가 필요하면 pcq.metric.stateful 사용.

Same atoms are also reachable via ``pcq.examples.metric.*``.
"""
from pcq.metric import stateful
from pcq.metric.functional import (
    accuracy,
    dice_score,
    iou,
    mean_absolute_error,
    mean_squared_error,
    pixel_accuracy,
    top_k_accuracy,
)

__all__ = [
    "accuracy",
    "dice_score",
    "iou",
    "mean_absolute_error",
    "mean_squared_error",
    "pixel_accuracy",
    "stateful",
    "top_k_accuracy",
]
