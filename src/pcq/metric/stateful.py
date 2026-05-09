"""pcq.metric.stateful — opt-in stateful metric classes.

functional metric 은 batch-mean 으로 집계되는 반면, stateful 은 sum 누적 후 compute.
v1.3 은 사용자가 직접 update/compute. v1.4 에서 Experiment 자동 통합 예정.

사용 예:
    acc = pcq.metric.stateful.Accuracy()
    for batch in loader:
        x, y = batch
        acc.update(model(x), y)
    print(acc.compute())
"""
from __future__ import annotations

import torch


class StatefulMetric:
    """Base class. update() 호출하면 누적, compute() 로 최종값."""

    def update(self, *args, **kwargs) -> None:  # noqa: D401
        raise NotImplementedError

    def compute(self) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class Accuracy(StatefulMetric):
    """Sample-weighted top-1 accuracy. variable batch size 에서 정확."""

    def __init__(self) -> None:
        self.correct: int = 0
        self.total: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        # logits: (B, C), labels: (B,) — 정수 라벨
        self.correct += int((logits.argmax(-1) == labels).sum().item())
        self.total += int(labels.numel())

    def compute(self) -> float:
        return self.correct / max(1, self.total)

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class IoU(StatefulMetric):
    """Mean IoU across classes (per-pixel multi-class segmentation).

    누적된 intersection/union 으로 dataset-level mean IoU 계산.
    functional iou 가 batch-mean 인 반면 stateful 은 sample-weighted.
    """

    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.intersection = torch.zeros(num_classes)
        self.union = torch.zeros(num_classes)

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        # logits: (B, C, H, W), labels: (B, H, W)
        preds = logits.argmax(dim=1)
        mask = labels != self.ignore_index
        for c in range(self.num_classes):
            pred_c = preds == c
            label_c = labels == c
            self.intersection[c] += (pred_c & label_c & mask).sum().item()
            self.union[c] += ((pred_c | label_c) & mask).sum().item()

    def compute(self) -> float:
        valid = self.union > 0
        if not valid.any():
            return 0.0
        ious = self.intersection[valid] / self.union[valid]
        return float(ious.mean().item())

    def reset(self) -> None:
        self.intersection.zero_()
        self.union.zero_()
