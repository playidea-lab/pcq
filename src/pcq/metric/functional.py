"""pcq.metric.functional — functional metric atoms.

각 함수는 (predictions, targets) → 스칼라 tensor 반환.
loss 와 동형으로 atom 으로 분리. 사용자는 training_step/eval_step 에서 직접 호출.

Aggregation 은 Experiment._run_epoch 가 batch-mean (또는 weighted_mean) 으로 처리.
정확한 sample-weighted 집계가 필요하면 pcq.metric.stateful 의 StatefulMetric 사용.
"""
from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Top-1 accuracy. logits: (B, C), labels: (B,) → 스칼라."""
    return (logits.argmax(dim=-1) == labels).float().mean()


def top_k_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, k: int = 5
) -> torch.Tensor:
    """Top-k accuracy. logits: (B, C), labels: (B,)."""
    pred_top_k = logits.topk(k, dim=-1).indices  # (B, k)
    return (pred_top_k == labels.unsqueeze(-1)).any(dim=-1).float().mean()


def mean_squared_error(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """MSE — predictions/targets 같은 shape."""
    return ((predictions - targets) ** 2).mean()


def mean_absolute_error(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """MAE — predictions/targets 같은 shape."""
    return (predictions - targets).abs().mean()


def iou(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1
) -> torch.Tensor:
    """Mean IoU per batch. logits: (B,C,H,W), labels: (B,H,W) → 스칼라.

    배치 내 등장한 클래스에 대한 평균 IoU. 정확한 sample-weighted IoU 가
    필요하면 pcq.metric.stateful.IoU 사용.
    """
    preds = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    valid = labels != ignore_index
    ious: list[torch.Tensor] = []
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        intersection = (pred_c & label_c & valid).sum().float()
        union = ((pred_c | label_c) & valid).sum().float()
        if union > 0:
            ious.append(intersection / union)
    if not ious:
        return torch.tensor(0.0)
    return torch.stack(ious).mean()


def dice_score(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1
) -> torch.Tensor:
    """Mean Dice score per batch. logits: (B,C,H,W), labels: (B,H,W)."""
    preds = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    valid = labels != ignore_index
    scores: list[torch.Tensor] = []
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        inter = (pred_c & label_c & valid).sum().float()
        denom = (pred_c & valid).sum().float() + (label_c & valid).sum().float()
        if denom > 0:
            scores.append(2 * inter / denom)
    if not scores:
        return torch.tensor(0.0)
    return torch.stack(scores).mean()


def pixel_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1
) -> torch.Tensor:
    """Pixel-wise accuracy. logits: (B,C,H,W), labels: (B,H,W)."""
    preds = logits.argmax(dim=1)
    valid = labels != ignore_index
    correct = ((preds == labels) & valid).sum().float()
    total = valid.sum().float()
    return correct / total.clamp(min=1)


# Registry 자동 등록 (string-name lookup 용)
from pcq._registry import metrics as _registry  # noqa: E402


# iou 의 metadata-aware factory: AtomRef.params 의 ignore_index 를 closure 로 캡처해
# (logits, labels) → scalar 형태의 callable 을 반환. 직접 호출 (pcq.metric.iou) 은 그대로.
def _iou_factory(ignore_index: int = -1):
    def _call(logits, labels):
        return iou(logits, labels, ignore_index=ignore_index)
    return _call


# v1.9: 모든 metric 이 metadata-aware (explicit). functional metric 은 zero-arg
# factory 가 callable 을 반환하는 패턴으로 통일 (build_ref 시점 closure 캡처).
_registry.register(
    "accuracy",
    factory=lambda: accuracy,
    meta={
        "tasks": ["classification"],
        "params": {},
        "input_contract": {"logits": ["B", "C"], "target": ["B"]},
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, "C-1"],
        },
        "metric_contract": {"mode": "max"},
        "smoke_safe": True,
        "description": (
            "Top-1 classification accuracy. [reference example]"
        ),
    },
)
_registry.register(
    "top_5_accuracy",
    factory=lambda: lambda logits, labels: top_k_accuracy(logits, labels, k=5),
    meta={
        "tasks": ["classification"],
        "params": {},
        "input_contract": {"logits": ["B", "C"], "target": ["B"]},
        "metric_contract": {"mode": "max"},
        "smoke_safe": True,
        "description": (
            "Top-5 classification accuracy. [reference example]"
        ),
    },
)
_registry.register(
    "mse",
    factory=lambda: mean_squared_error,
    meta={
        "tasks": ["regression"],
        "params": {},
        "input_contract": {
            "predictions": ["B", "..."],
            "target": ["B", "..."],
        },
        "metric_contract": {"mode": "min"},
        "smoke_safe": True,
        "description": "Mean squared error. [reference example]",
    },
)
_registry.register(
    "mae",
    factory=lambda: mean_absolute_error,
    meta={
        "tasks": ["regression"],
        "params": {},
        "input_contract": {
            "predictions": ["B", "..."],
            "target": ["B", "..."],
        },
        "metric_contract": {"mode": "min"},
        "smoke_safe": True,
        "description": "Mean absolute error. [reference example]",
    },
)
_registry.register(
    "iou",
    factory=_iou_factory,
    meta={
        "tasks": ["segmentation"],
        "params": {
            "ignore_index": {"type": "int", "default": -1},
        },
        "input_contract": {
            "logits": ["B", "C", "H", "W"],
            "target": ["B", "H", "W"],
        },
        "label_contract": {"ignore_index_param": "ignore_index"},
        "metric_contract": {"mode": "max"},
        "smoke_safe": True,
        "description": (
            "Mean IoU over classes present in batch. [reference example — "
            "use pcq.metric.stateful.IoU for sample-weighted aggregation]"
        ),
    },
)
_registry.register(
    "dice_score",
    factory=lambda ignore_index=-1: (
        lambda logits, labels: dice_score(
            logits, labels, ignore_index=ignore_index,
        )
    ),
    meta={
        "tasks": ["segmentation"],
        "params": {"ignore_index": {"type": "int", "default": -1}},
        "input_contract": {
            "logits": ["B", "C", "H", "W"],
            "target": ["B", "H", "W"],
        },
        "label_contract": {"ignore_index_param": "ignore_index"},
        "metric_contract": {"mode": "max"},
        "smoke_safe": True,
        "description": (
            "Mean Dice score per batch over present classes. "
            "[reference example]"
        ),
    },
)
_registry.register(
    "pixel_accuracy",
    factory=lambda ignore_index=-1: (
        lambda logits, labels: pixel_accuracy(
            logits, labels, ignore_index=ignore_index,
        )
    ),
    meta={
        "tasks": ["segmentation"],
        "params": {"ignore_index": {"type": "int", "default": -1}},
        "input_contract": {
            "logits": ["B", "C", "H", "W"],
            "target": ["B", "H", "W"],
        },
        "metric_contract": {"mode": "max"},
        "smoke_safe": True,
        "description": (
            "Pixel-wise classification accuracy (ignore_index aware). "
            "[reference example]"
        ),
    },
)
