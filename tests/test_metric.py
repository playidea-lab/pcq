"""pcq.metric atom tests."""
from __future__ import annotations

import torch

import pcq


def test_accuracy_basic():
    # 모든 예측 정답 → accuracy 1.0
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([1, 0, 1])
    assert pcq.metric.accuracy(logits, labels).item() == 1.0


def test_accuracy_partial():
    # 3개 중 2개 정답 → 2/3
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([1, 0, 0])
    assert abs(pcq.metric.accuracy(logits, labels).item() - 2 / 3) < 1e-6


def test_top_k_accuracy():
    # argmax 는 1, 정답은 2 → top-1 = 0, top-2 = 1
    logits = torch.tensor([[0.1, 0.5, 0.4]])
    labels = torch.tensor([2])
    assert pcq.metric.accuracy(logits, labels).item() == 0.0
    assert pcq.metric.top_k_accuracy(logits, labels, k=2).item() == 1.0


def test_mse_zero():
    p = torch.tensor([1.0, 2.0, 3.0])
    t = torch.tensor([1.0, 2.0, 3.0])
    assert pcq.metric.mean_squared_error(p, t).item() == 0.0


def test_mae_basic():
    # |1-0| + |2-4| = 1 + 2 = 3, /2 = 1.5
    p = torch.tensor([1.0, 2.0])
    t = torch.tensor([0.0, 4.0])
    assert pcq.metric.mean_absolute_error(p, t).item() == 1.5


# ── v1.3: Segmentation functional metrics ───────────────────────────────────
def test_iou_perfect():
    # 모든 pixel 을 class 0 으로 강하게 예측, label 도 모두 0 → IoU=1
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 0] = 10.0
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    assert pcq.metric.iou(logits, labels).item() > 0.99


def test_iou_ignore_index():
    # 일부 pixel 은 ignore_index=-1 — 평가 대상에서 제외되어 여전히 높은 IoU
    logits = torch.zeros(1, 3, 4, 4)
    logits[0, 0] = 10.0
    labels = torch.full((1, 4, 4), -1, dtype=torch.long)
    labels[0, 0:2, :] = 0  # 절반만 valid (= class 0)
    iou = pcq.metric.iou(logits, labels, ignore_index=-1).item()
    assert iou > 0.99


def test_dice_score_perfect():
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 0] = 10.0
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    assert pcq.metric.dice_score(logits, labels).item() > 0.99


def test_pixel_accuracy_perfect():
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 0] = 10.0
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    assert pcq.metric.pixel_accuracy(logits, labels).item() == 1.0


# ── v1.3: Stateful metrics ──────────────────────────────────────────────────
def test_stateful_accuracy_basic():
    metric = pcq.metric.stateful.Accuracy()
    metric.update(torch.tensor([[0.1, 0.9], [0.8, 0.2]]), torch.tensor([1, 0]))
    assert metric.compute() == 1.0
    # 누적 — 추가 batch 1개 (1개 틀림) → 2/3
    metric.update(torch.tensor([[0.1, 0.9]]), torch.tensor([0]))
    assert abs(metric.compute() - 2 / 3) < 1e-6


def test_stateful_accuracy_reset():
    metric = pcq.metric.stateful.Accuracy()
    metric.update(torch.tensor([[0.1, 0.9]]), torch.tensor([1]))
    metric.reset()
    assert metric.compute() == 0.0
    assert metric.total == 0


def test_stateful_iou_perfect():
    metric = pcq.metric.stateful.IoU(num_classes=2)
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 0] = 10.0
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    metric.update(logits, labels)
    assert metric.compute() > 0.99


def test_stateful_iou_reset():
    metric = pcq.metric.stateful.IoU(num_classes=3)
    logits = torch.randn(1, 3, 4, 4)
    labels = torch.randint(0, 3, (1, 4, 4))
    metric.update(logits, labels)
    metric.reset()
    assert metric.intersection.sum().item() == 0.0
    assert metric.union.sum().item() == 0.0
