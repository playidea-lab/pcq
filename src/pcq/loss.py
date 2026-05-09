"""pcq.loss — reference example loss atoms.

Contract examples (``cross_entropy``, ``dice``, ``focal``) for the loss atom
kind. NOT a production loss catalog. Project-local losses live in
``atoms/losses.py`` registered via ``pcq.register_loss`` with explicit
``label_contract`` metadata.

Same atoms are also reachable via ``pcq.examples.loss.*``.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def cross_entropy(
    ignore_index: int = -100, weight: torch.Tensor | None = None
) -> nn.Module:
    """Cross-entropy loss factory.

    Args:
        ignore_index: 무시할 target index (default -100, PyTorch 표준).
                      Segmentation 에선 ignore_index=-1 권장 (voc_seg 가 -1 사용).
        weight: 클래스별 가중치 (선택).
    """
    return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)


class _DiceLoss(nn.Module):
    """Soft Dice loss for segmentation.

    logits: (B, C, H, W) — class 차원에 softmax 적용
    targets: (B, H, W) — int64 class id, ignore_index 마스크 지원
    반환: 1 - mean(dice) per class.
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
        valid = targets != self.ignore_index  # (B, H, W)
        # ignore 픽셀은 0 으로 임시 치환 (one-hot 안전), 이후 mask 로 제외
        targets_clean = targets.clone()
        targets_clean[~valid] = 0
        target_onehot = F.one_hot(targets_clean, num_classes).permute(
            0, 3, 1, 2
        ).float()
        valid_mask = valid.unsqueeze(1).float()  # (B, 1, H, W)
        intersection = (probs * target_onehot * valid_mask).sum(dim=(2, 3))
        denom = (probs * valid_mask).sum(dim=(2, 3)) + (
            target_onehot * valid_mask
        ).sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class _FocalLoss(nn.Module):
    """Focal loss — class imbalance 에 강함. alpha=1, gamma=2 default.

    logits: (B, C, ...) — cross_entropy 와 동일한 shape
    targets: (B, ...) — int64 class id
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )
        # ce 가 0 인 위치 (ignore) 는 pt=1, focal weight=0 이라 자동 제외됨
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


def dice(smooth: float = 1.0, ignore_index: int = -1) -> nn.Module:
    """Dice loss factory."""
    return _DiceLoss(smooth, ignore_index)


def focal(
    alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = -100
) -> nn.Module:
    """Focal loss factory."""
    return _FocalLoss(alpha, gamma, ignore_index)


# Registry 자동 등록 (string-name lookup 용)
from pcq._registry import losses as _registry  # noqa: E402

# v1.8: cross_entropy 는 metadata-aware 등록 (params/contracts/tasks 명시).
# dice/focal 은 v1.9 에서 보강 — 현재는 inferred metadata.
_registry.register(
    "cross_entropy",
    factory=cross_entropy,
    meta={
        "tasks": ["classification", "segmentation"],
        "params": {
            "ignore_index": {
                "type": "int",
                "default": -100,
                "description": (
                    "target index to ignore. Segmentation 권장: -1 "
                    "(voc_seg 가 void/255 → -1 로 변환)."
                ),
            },
            "weight": {
                "type": "any",
                "default": None,
                "description": "per-class weighting tensor (optional)",
            },
        },
        "input_contract": {
            "logits": ["B", "C", "..."],
            "target": ["B", "..."],
        },
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, "C-1"],
            "ignore_index_param": "ignore_index",
        },
        "smoke_safe": True,
        "description": (
            "Cross-entropy loss with optional class weights and ignore_index. "
            "[reference example — common baseline; register project atoms for "
            "specialised losses]"
        ),
    },
)
_registry.register(
    "dice",
    factory=lambda smooth=1.0, ignore_index=-1: dice(
        smooth=smooth, ignore_index=ignore_index,
    ),
    meta={
        "tasks": ["segmentation"],
        "params": {
            "smooth": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "description": "Laplace smoothing constant",
            },
            "ignore_index": {"type": "int", "default": -1},
        },
        "input_contract": {
            "logits": ["B", "C", "H", "W"],
            "target": ["B", "H", "W"],
        },
        "label_contract": {
            "target_dtype": "int64",
            "ignore_index_param": "ignore_index",
        },
        "smoke_safe": True,
        "description": (
            "Soft Dice loss for segmentation. "
            "[reference example — register project atoms for boundary/class-"
            "weighted variants]"
        ),
    },
)
_registry.register(
    "focal",
    factory=lambda alpha=1.0, gamma=2.0, ignore_index=-100: focal(
        alpha=alpha, gamma=gamma, ignore_index=ignore_index,
    ),
    meta={
        "tasks": ["classification", "segmentation"],
        "params": {
            "alpha": {"type": "float", "default": 1.0, "min": 0.0},
            "gamma": {
                "type": "float",
                "default": 2.0,
                "min": 0.0,
                "description": (
                    "focusing parameter (gamma=0 → cross-entropy)"
                ),
            },
            "ignore_index": {"type": "int", "default": -100},
        },
        "input_contract": {
            "logits": ["B", "C", "..."],
            "target": ["B", "..."],
        },
        "label_contract": {
            "target_dtype": "int64",
            "ignore_index_param": "ignore_index",
        },
        "smoke_safe": True,
        "description": (
            "Focal loss — class imbalance robust. "
            "[reference example — register a project atom if you tune "
            "alpha/gamma per dataset]"
        ),
    },
)
