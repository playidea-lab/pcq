"""pcq.examples.models — reference example model atoms.

These models are provided as **contract examples**, NOT a production model
catalog. They exist so pcq's registry / Trainer / Experiment surface has a
working baseline for:
  - registry validation tests
  - onboarding new users (concrete reference for ``pcq.register_model`` usage)
  - CI smoke tests that catch framework breakage

For production research models, register your own atom in project-local
``atoms/`` — see ``pcq atoms scaffold model NAME``.

Available atoms (reference examples):
  - ``mlp(in_dim, hidden, out_dim)``  — flatten + Linear+ReLU stack
  - ``small_cnn(in_channels, num_classes)`` — 3-block Conv-BN-ReLU
  - ``resnet18`` — torchvision wrapper (replaces final fc)
  - ``unet`` — tiny 3-level segmentation reference (no timm dep)
  - ``text_classifier`` — Embedding + mean pool + Linear (NLP smoke)
  - ``deeplab_v3`` — torchvision wrapper for segmentation reference

``pcq.models`` re-exports these functions as a v2 compatibility facade.
"""
from __future__ import annotations

import torch
from torch import nn


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        # __repr__ 용 메타 (파라미터 변경시 동시 업데이트)
        self._meta = {"in_dim": in_dim, "hidden": hidden, "out_dim": out_dim}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def __repr__(self) -> str:
        m = self._meta
        return f"MLP({m['in_dim']} -> {m['hidden']} -> {m['out_dim']})"


class _SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            # Conv-BN-ReLU-MaxPool 한 블록
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, num_classes)
        self._meta = {"in_channels": in_channels, "num_classes": num_classes}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    def __repr__(self) -> str:
        m = self._meta
        return f"SmallCNN(in_channels={m['in_channels']}, num_classes={m['num_classes']})"


def mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Module:
    """MLP atom (Linear+ReLU stack with input flatten)."""
    return _MLP(in_dim, hidden, out_dim)


def small_cnn(in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    """3-block Conv-BN-ReLU-Pool + GAP + Linear."""
    return _SmallCNN(in_channels, num_classes)


def resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """torchvision ResNet-18 wrapper. 마지막 fc 를 num_classes 로 교체.

    pretrained=True 면 ImageNet weight 다운로드.
    Note: torchvision 필요. lazy import 로 cold start 단축.
    """
    # cold start 줄이기 위해 함수 내부에서 import
    from torchvision import models as tv_models

    if pretrained:
        m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    else:
        m = tv_models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class _TextClassifier(nn.Module):
    """Embedding + mean pool + Linear. Tokenizer 없이 정수 시퀀스 직접 받음."""

    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        # padding_idx=0 → token id 0 은 padding 으로 예약, embedding 학습 X
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.head = nn.Linear(embed_dim, num_classes)
        self._meta = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_classes": num_classes,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len) int64
        emb = self.embedding(x)  # (B, seq_len, embed_dim)
        pooled = emb.mean(dim=1)  # (B, embed_dim)
        return self.head(pooled)  # (B, num_classes)

    def __repr__(self) -> str:
        m = self._meta
        return (
            f"TextClassifier(vocab={m['vocab_size']}, "
            f"embed={m['embed_dim']}, cls={m['num_classes']})"
        )


def text_classifier(
    vocab_size: int, embed_dim: int = 64, num_classes: int = 2
) -> nn.Module:
    """간단한 텍스트 분류 모델 — Embedding + mean + Linear. NLP recipe demo 용."""
    return _TextClassifier(vocab_size, embed_dim, num_classes)


class _UNet(nn.Module):
    """Tiny UNet (3-level encoder/decoder). Segmentation baseline.

    timm/segmentation_models_pytorch 의존 없이 torch 만으로 구성. 입력
    (B, in_channels, H, W) → 출력 (B, num_classes, H, W). H/W 는 4의 배수 권장
    (downsample 두 번 후 upsample).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        base_ch: int = 32,
    ) -> None:
        super().__init__()

        def conv_block(c_in: int, c_out: int) -> nn.Sequential:
            # Conv-BN-ReLU x2 한 블록 (UNet 표준)
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)
        self.head = nn.Conv2d(base_ch, num_classes, 1)
        self._meta = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "base_ch": base_ch,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # Decoder + skip connection
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

    def __repr__(self) -> str:
        m = self._meta
        return (
            f"UNet(in={m['in_channels']}, "
            f"classes={m['num_classes']}, base={m['base_ch']})"
        )


def unet(
    in_channels: int = 3, num_classes: int = 21, base_ch: int = 32
) -> nn.Module:
    """Tiny 3-level UNet for segmentation. timm 의존 X."""
    return _UNet(in_channels, num_classes, base_ch)


def deeplab_v3(num_classes: int = 21, pretrained: bool = False) -> nn.Module:
    """torchvision DeepLabV3 (ResNet50 backbone).

    pcq[vision] extras (torchvision) 권장. pretrained=True 면 COCO weight
    다운로드 후 classifier head 만 num_classes 로 교체.
    """
    # cold start 줄이기 위해 함수 내부에서 import
    from torchvision import models as tv_models

    if pretrained:
        weights = (
            tv_models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        )
        m = tv_models.segmentation.deeplabv3_resnet50(weights=weights)
        # head 교체 — DeepLabHead 의 마지막 Conv2d 가 num_classes 출력
        m.classifier = tv_models.segmentation.deeplabv3.DeepLabHead(
            2048, num_classes
        )
    else:
        m = tv_models.segmentation.deeplabv3_resnet50(
            weights=None, num_classes=num_classes
        )
    return m


# Registry 자동 등록 (string-name lookup 용)
from pcq._registry import models as _registry  # noqa: E402

# v1.9: 모든 model 이 metadata-aware (explicit). factory 는 keyword 인자를 받아
# AtomRef.params 와 매핑 가능하게 정의. Trainer(model="name") 호환은 default
# 값으로 보장 (zero-arg 호출 가능).
_registry.register(
    "mlp",
    factory=lambda in_dim=3 * 32 * 32, hidden=None, out_dim=10: mlp(
        in_dim=in_dim,
        # default mutable 회피 — None 이면 [128] 사용
        hidden=hidden if hidden is not None else [128],
        out_dim=out_dim,
    ),
    meta={
        "tasks": ["classification"],
        "params": {
            "in_dim": {
                "type": "int",
                "default": 3 * 32 * 32,
                "min": 1,
                "description": "flattened input dim",
            },
            "hidden": {
                "type": "any",
                "default": [128],
                "description": "hidden layer sizes (list[int])",
            },
            "out_dim": {
                "type": "int",
                "default": 10,
                "min": 1,
                "description": "number of classes",
            },
        },
        "input_contract": {"x": ["B", "in_dim"]},
        "output_contract": {"logits": ["B", "out_dim"]},
        "smoke_safe": True,
        "description": (
            "Multi-layer perceptron (Linear+ReLU stack with input flatten). "
            "[reference example — register a project-local atom for production MLPs]"
        ),
    },
)
_registry.register(
    "small_cnn",
    factory=lambda in_channels=3, num_classes=10: small_cnn(
        in_channels=in_channels, num_classes=num_classes,
    ),
    meta={
        "tasks": ["classification"],
        "params": {
            "in_channels": {"type": "int", "default": 3, "min": 1},
            "num_classes": {"type": "int", "default": 10, "min": 1},
        },
        "input_contract": {"x": ["B", "in_channels", "H", "W"]},
        "output_contract": {"logits": ["B", "num_classes"]},
        "smoke_safe": True,
        "description": (
            "3-block Conv-BN-ReLU-MaxPool + global pooling + linear head. "
            "[reference example — register a project-local atom for production CNNs]"
        ),
    },
)
_registry.register(
    "resnet18",
    factory=lambda num_classes=10, pretrained=False: resnet18(
        num_classes=num_classes, pretrained=pretrained,
    ),
    meta={
        "tasks": ["classification"],
        "params": {
            "num_classes": {"type": "int", "default": 10, "min": 1},
            "pretrained": {
                "type": "bool",
                "default": False,
                "description": (
                    "load torchvision ImageNet pretrained weights"
                ),
            },
        },
        "input_contract": {"x": ["B", "3", "H", "W"]},
        "output_contract": {"logits": ["B", "num_classes"]},
        "requires_extras": ["vision"],
        "smoke_safe": True,
        "description": (
            "torchvision ResNet-18 with replaced final fc layer. "
            "[reference example — for production, register a project atom "
            "tuned to your dataset]"
        ),
    },
)
_registry.register(
    "text_classifier",
    factory=lambda vocab_size=1000, embed_dim=64, num_classes=2: text_classifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
    ),
    meta={
        "tasks": ["text_classification"],
        "params": {
            "vocab_size": {"type": "int", "default": 1000, "min": 2},
            "embed_dim": {"type": "int", "default": 64, "min": 1},
            "num_classes": {"type": "int", "default": 2, "min": 1},
        },
        "input_contract": {"x": ["B", "seq_len"]},
        "output_contract": {"logits": ["B", "num_classes"]},
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, "num_classes-1"],
        },
        "smoke_safe": True,
        "description": (
            "Embedding + mean pool + linear (tokenizer-free, integer sequences). "
            "[reference example — for production, register a project atom with "
            "real tokenizer + transformer]"
        ),
    },
)
_registry.register(
    "unet",
    factory=unet,
    meta={
        "tasks": ["segmentation"],
        "params": {
            "in_channels": {"type": "int", "default": 3, "min": 1},
            "num_classes": {"type": "int", "default": 21, "min": 1},
            "base_ch": {"type": "int", "default": 32, "min": 1},
        },
        "input_contract": {"x": ["B", "in_channels", "H", "W"]},
        "output_contract": {"logits": ["B", "num_classes", "H", "W"]},
        "smoke_safe": True,
        "description": (
            "Tiny 3-level UNet for segmentation. timm 의존 X. "
            "[reference example — for production, register a project atom with "
            "stronger encoder + skip connections]"
        ),
    },
)
_registry.register(
    "deeplab_v3",
    factory=lambda num_classes=21, pretrained=False: deeplab_v3(
        num_classes=num_classes, pretrained=pretrained,
    ),
    meta={
        "tasks": ["segmentation"],
        "params": {
            "num_classes": {"type": "int", "default": 21, "min": 1},
            "pretrained": {"type": "bool", "default": False},
        },
        "input_contract": {"x": ["B", "3", "H", "W"]},
        "output_contract": {"logits": ["B", "num_classes", "H", "W"]},
        "requires_extras": ["vision"],
        "smoke_safe": True,
        "description": (
            "torchvision DeepLabV3 (ResNet50 backbone) with custom head. "
            "[reference example — for production, register a project atom "
            "tuned to your seg task]"
        ),
    },
)
