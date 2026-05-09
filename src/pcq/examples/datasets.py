"""pcq.examples.datasets — reference example dataset atoms.

Contract examples for the dataset atom kind, NOT a maintained dataset catalog.
They cover smoke runs (``fake``, ``fake_text``, ``fake_seg``) and a few
torchvision wrappers (``cifar10``, ``mnist``, ``voc_seg``) used in onboarding
recipes. Real datasets should be registered as project-local atoms via
``pcq.register_dataset`` — see ``pcq atoms scaffold dataset NAME``.

``pcq.datasets`` re-exports these functions as a v2 compatibility facade.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class _FakeDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 128,
        num_classes: int = 10,
        image_size: int = 32,
        channels: int = 3,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
        # 재현성: 고정 generator로 동일 입력 → 동일 출력 보장
        g = torch.Generator().manual_seed(42)
        self.images = torch.randn(
            num_samples, channels, image_size, image_size, generator=g
        )
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]

    def __repr__(self) -> str:
        return (
            f"FakeDataset(n={self.num_samples}, "
            f"shape=({self.channels},{self.image_size},{self.image_size}), "
            f"classes={self.num_classes})"
        )


def fake(
    num_samples: int = 128,
    num_classes: int = 10,
    image_size: int = 32,
    channels: int = 3,
) -> Dataset:
    """Random tensor dataset for smoke tests."""
    return _FakeDataset(num_samples, num_classes, image_size, channels)


def cifar10(root: str | Path, train: bool = True, download: bool = True) -> Dataset:
    """CIFAR-10 wrapper (torchvision). Includes ToTensor + Normalize.

    Note: requires torchvision. Lazy import for cold start.
    """
    # cold start 줄이기 위해 함수 내부에서 import
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            ),
        ]
    )
    return datasets.CIFAR10(
        root=str(root), train=train, download=download, transform=transform
    )


def mnist(root: str | Path, train: bool = True, download: bool = True) -> Dataset:
    """MNIST wrapper (torchvision). ToTensor + 표준 정규화.

    Note: requires torchvision. Lazy import for cold start.
    """
    # cold start 줄이기 위해 함수 내부에서 import
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return datasets.MNIST(
        root=str(root), train=train, download=download, transform=transform
    )


class _FakeTextDataset(Dataset):
    """Random integer sequences with class labels. Tokenizer-free NLP smoke."""

    def __init__(
        self,
        num_samples: int = 128,
        num_classes: int = 2,
        vocab_size: int = 1000,
        seq_len: int = 32,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        # 재현성: 고정 generator로 동일 입력 → 동일 출력 보장
        g = torch.Generator().manual_seed(43)
        # 0 은 padding 으로 예약, 1..vocab_size-1 만 사용
        self.tokens = torch.randint(
            1, vocab_size, (num_samples, seq_len), generator=g
        )
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.labels[idx]

    def __repr__(self) -> str:
        return (
            f"FakeTextDataset(n={self.num_samples}, "
            f"vocab={self.vocab_size}, seq={self.seq_len}, "
            f"classes={self.num_classes})"
        )


def fake_text(
    num_samples: int = 128,
    num_classes: int = 2,
    vocab_size: int = 1000,
    seq_len: int = 32,
) -> Dataset:
    """Tokenizer-free fake text dataset for NLP recipe smoke."""
    return _FakeTextDataset(num_samples, num_classes, vocab_size, seq_len)


class _FakeSegDataset(Dataset):
    """Random images + random class masks. Tokenizer-free seg smoke.

    각 sample 은 (image: (C,H,W) float, mask: (H,W) int64).
    재현성: 고정 generator (seed=44).
    """

    def __init__(
        self,
        num_samples: int = 32,
        num_classes: int = 21,
        image_size: int = 64,
        channels: int = 3,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
        # 재현성 — fake_text(seed=43) 와 다른 seed
        g = torch.Generator().manual_seed(44)
        self.images = torch.randn(
            num_samples, channels, image_size, image_size, generator=g
        )
        self.masks = torch.randint(
            0, num_classes, (num_samples, image_size, image_size), generator=g
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.masks[idx]

    def __repr__(self) -> str:
        return (
            f"FakeSegDataset(n={self.num_samples}, "
            f"shape=({self.channels},{self.image_size},{self.image_size}), "
            f"classes={self.num_classes})"
        )


def fake_seg(
    num_samples: int = 32,
    num_classes: int = 21,
    image_size: int = 64,
    channels: int = 3,
) -> Dataset:
    """Random seg dataset for smoke (no torchvision required)."""
    return _FakeSegDataset(num_samples, num_classes, image_size, channels)


def voc_seg(
    root: str | Path,
    image_set: str = "train",
    download: bool = True,
    image_size: int = 256,
) -> Dataset:
    """Pascal VOC 2012 segmentation. pcq[vision] extras (torchvision) 권장.

    PIL 이미지/마스크 → tensor 변환 + ImageNet 정규화 + ignore_index=-1
    (VOC 의 255 = void label).

    Args:
        image_size: image 와 mask 를 (image_size, image_size) 로 리사이즈.
                    가변 크기는 default DataLoader collate 가 stack 못해 학습 실패.
                    image 는 bilinear, mask 는 nearest 로 동기 리사이즈.
    """
    # cold start 줄이기 위해 함수 내부에서 import
    import numpy as np
    from torchvision import datasets
    from torchvision.transforms import functional as TF
    from torchvision.transforms import InterpolationMode

    class _VOCSegWrapper(Dataset):
        def __init__(
            self,
            root: str | Path,
            image_set: str,
            download: bool,
            image_size: int,
        ) -> None:
            self.ds = datasets.VOCSegmentation(
                root=str(root), image_set=image_set, download=download
            )
            self.image_size = image_size
            self._image_set = image_set

        def __len__(self) -> int:
            return len(self.ds)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            img, mask = self.ds[idx]
            # 동기 리사이즈 — image 는 bilinear, mask 는 nearest (라벨 보존)
            img = TF.resize(
                img,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
            )
            mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )
            img = TF.to_tensor(img)
            img = TF.normalize(
                img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
            mask_arr = np.array(mask, dtype=np.int64)
            # VOC 의 255 (void) → ignore_index=-1
            mask_arr[mask_arr == 255] = -1
            mask_t = torch.from_numpy(mask_arr)
            return img, mask_t

        def __repr__(self) -> str:
            return (
                f"VOCSegmentation(image_size={self.image_size}, "
                f"image_set={self._image_set!r})"
            )

    return _VOCSegWrapper(root, image_set, download, image_size)


# Registry 자동 등록 (string-name lookup 용)
from pcq._registry import datasets as _registry  # noqa: E402


# Hybrid factories — split positional 또는 kwargs(num_samples=...) 둘 다 지원.
# Trainer(dataset="name") 경로: factory(split) 로 호출 → split 인자만 사용.
# AtomRef 경로: factory(**params) 로 호출 → split 무시, params 적용.
def _fake_factory(
    split: str | None = None,
    num_samples: int = 128,
    num_classes: int = 10,
    image_size: int = 32,
    channels: int = 3,
):
    return fake(
        num_samples=num_samples,
        num_classes=num_classes,
        image_size=image_size,
        channels=channels,
    )


def _fake_text_factory(
    split: str | None = None,
    num_samples: int = 128,
    num_classes: int = 2,
    vocab_size: int = 1000,
    seq_len: int = 32,
):
    return fake_text(
        num_samples=num_samples,
        num_classes=num_classes,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )


def _cifar10_factory(
    split: str | None = None,
    root: str = "data",
    train: bool | None = None,
    download: bool = True,
):
    # split 인자가 들어오면 자동으로 train 결정 (positional 호환)
    if train is None:
        train = split == "train"
    return cifar10(root=root, train=train, download=download)


def _mnist_factory(
    split: str | None = None,
    root: str = "data",
    train: bool | None = None,
    download: bool = True,
):
    if train is None:
        train = split == "train"
    return mnist(root=root, train=train, download=download)


def _fake_seg_factory(
    split: str | None = None,
    num_samples: int = 32,
    num_classes: int = 21,
    image_size: int = 64,
    channels: int = 3,
):
    return fake_seg(
        num_samples=num_samples,
        num_classes=num_classes,
        image_size=image_size,
        channels=channels,
    )


def _voc_seg_factory(
    split: str | None = None,
    root: str = "data",
    image_set: str | None = None,
    download: bool = True,
    image_size: int = 256,
):
    # split 인자가 들어오면 자동으로 image_set 결정
    if image_set is None:
        image_set = "train" if split == "train" else "val"
    return voc_seg(
        root=root,
        image_set=image_set,
        download=download,
        image_size=image_size,
    )


# v1.9: 모든 dataset 이 metadata-aware (explicit). hybrid factory 패턴 유지.
_registry.register(
    "fake",
    factory=_fake_factory,
    meta={
        "tasks": ["classification"],
        "params": {
            "num_samples": {"type": "int", "default": 128, "min": 1},
            "num_classes": {"type": "int", "default": 10, "min": 1},
            "image_size": {"type": "int", "default": 32, "min": 1},
            "channels": {"type": "int", "default": 3, "min": 1},
        },
        "output_contract": {
            "x": ["channels", "image_size", "image_size"],
            "y": [],
        },
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, "num_classes-1"],
        },
        "smoke_safe": True,
        "description": (
            "Random tensor dataset for smoke runs. "
            "[reference example — register a project atom for real datasets]"
        ),
    },
)
_registry.register(
    "fake_text",
    factory=_fake_text_factory,
    meta={
        "tasks": ["text_classification"],
        "params": {
            "num_samples": {"type": "int", "default": 128, "min": 1},
            "num_classes": {"type": "int", "default": 2, "min": 1},
            "vocab_size": {"type": "int", "default": 1000, "min": 2},
            "seq_len": {"type": "int", "default": 32, "min": 1},
        },
        "output_contract": {"x": ["seq_len"], "y": []},
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, "num_classes-1"],
        },
        "smoke_safe": True,
        "description": (
            "Random integer sequences for tokenizer-free NLP smoke. "
            "[reference example — for production, register a project atom "
            "with real tokenizer]"
        ),
    },
)
_registry.register(
    "cifar10",
    factory=_cifar10_factory,
    meta={
        "tasks": ["classification"],
        "params": {
            "root": {"type": "path", "default": "data"},
            "train": {"type": "bool", "default": True},
            "download": {"type": "bool", "default": True},
        },
        "output_contract": {"x": ["3", "32", "32"], "y": []},
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, 9],
        },
        "requires_extras": ["vision"],
        "smoke_safe": False,
        "description": (
            "CIFAR-10 (torchvision wrapper, ToTensor + Normalize). "
            "[reference example — onboarding wrapper, not a maintained dataset]"
        ),
    },
)
_registry.register(
    "mnist",
    factory=_mnist_factory,
    meta={
        "tasks": ["classification"],
        "params": {
            "root": {"type": "path", "default": "data"},
            "train": {"type": "bool", "default": True},
            "download": {"type": "bool", "default": True},
        },
        "output_contract": {"x": ["1", "28", "28"], "y": []},
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, 9],
        },
        "requires_extras": ["vision"],
        "smoke_safe": False,
        "description": (
            "MNIST (torchvision wrapper, ToTensor + Normalize). "
            "[reference example — onboarding wrapper, not a maintained dataset]"
        ),
    },
)
_registry.register(
    "fake_seg",
    factory=_fake_seg_factory,
    meta={
        "tasks": ["segmentation"],
        "params": {
            "num_samples": {"type": "int", "default": 32, "min": 1},
            "num_classes": {"type": "int", "default": 21, "min": 1},
            "image_size": {"type": "int", "default": 64, "min": 8},
            "channels": {"type": "int", "default": 3, "min": 1},
        },
        "output_contract": {
            "x": ["channels", "image_size", "image_size"],
            "y": ["image_size", "image_size"],
        },
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, "num_classes-1"],
        },
        "smoke_safe": True,
        "description": (
            "Random images + random masks for segmentation smoke. "
            "[reference example — register a project atom for real seg data]"
        ),
    },
)
_registry.register(
    "voc_seg",
    factory=_voc_seg_factory,
    meta={
        "tasks": ["segmentation"],
        "params": {
            "root": {
                "type": "path",
                "default": "data",
                "description": "data directory",
            },
            "image_set": {
                "type": "str",
                "default": "train",
                "choices": ["train", "val", "trainval"],
            },
            "download": {"type": "bool", "default": True},
            "image_size": {"type": "int", "default": 256, "min": 32},
        },
        "output_contract": {
            "x": ["3", "image_size", "image_size"],
            "y": ["image_size", "image_size"],
        },
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, 20],
            "ignore_index": -1,
        },
        "requires_extras": ["vision"],
        "smoke_safe": False,
        "description": (
            "Pascal VOC 2012 segmentation. Requires torchvision. "
            "void(255) labels are converted to -1 (ignore_index). "
            "[reference example — onboarding wrapper for the contract]"
        ),
    },
)
