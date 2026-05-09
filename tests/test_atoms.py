"""tests for cq atom catalog (models/datasets/optim/sched/loss)."""
from __future__ import annotations

import pytest
import torch
from torch import nn

import pcq


# ---------- models.mlp ----------


def test_mlp_forward_shape_returns_batch_x_out_dim():
    # MLP는 입력을 flatten 후 [B, out_dim]로 매핑한다
    x = torch.randn(8, 3, 32, 32)
    model = pcq.models.mlp(3 * 32 * 32, [64], 10)
    y = model(x)
    assert y.shape == (8, 10)


def test_mlp_repr_includes_dims():
    # __repr__에 입력/은닉/출력 차원 정보가 모두 포함돼야 한다
    model = pcq.models.mlp(784, [128, 64], 10)
    r = repr(model)
    assert "784" in r and "128" in r and "64" in r and "10" in r


def test_mlp_no_hidden_layers_works():
    # hidden=[]는 단일 Linear와 동등 (엣지 케이스)
    x = torch.randn(4, 10)
    model = pcq.models.mlp(10, [], 5)
    y = model(x)
    assert y.shape == (4, 5)


# ---------- models.small_cnn ----------


def test_small_cnn_forward_shape_default_returns_batch_x_10():
    x = torch.randn(8, 3, 32, 32)
    model = pcq.models.small_cnn()
    y = model(x)
    assert y.shape == (8, 10)


def test_small_cnn_grayscale_2class_handles_custom_channels():
    # in_channels=1, num_classes=2로 흑백 이진분류 가능해야 한다
    x = torch.randn(4, 1, 28, 28)
    model = pcq.models.small_cnn(in_channels=1, num_classes=2)
    y = model(x)
    assert y.shape == (4, 2)


def test_small_cnn_repr_includes_meta():
    model = pcq.models.small_cnn(in_channels=3, num_classes=100)
    r = repr(model)
    assert "in_channels=3" in r and "num_classes=100" in r


# ---------- datasets.fake ----------


def test_fake_dataset_len_and_item_shapes():
    ds = pcq.datasets.fake(num_samples=16)
    assert len(ds) == 16
    img, lbl = ds[0]
    assert img.shape == (3, 32, 32)
    # label은 정수 스칼라여야 한다
    assert isinstance(lbl.item(), int)


def test_fake_dataset_reproducibility_same_first_item():
    # 같은 파라미터로 두 번 만들면 첫 샘플이 동일해야 한다
    ds1 = pcq.datasets.fake(num_samples=8)
    ds2 = pcq.datasets.fake(num_samples=8)
    img1, lbl1 = ds1[0]
    img2, lbl2 = ds2[0]
    assert torch.equal(img1, img2)
    assert lbl1.item() == lbl2.item()


def test_fake_dataset_custom_shape():
    ds = pcq.datasets.fake(num_samples=4, channels=1, image_size=16, num_classes=3)
    img, lbl = ds[2]
    assert img.shape == (1, 16, 16)
    assert 0 <= lbl.item() < 3


# ---------- datasets.voc_seg ----------


def test_voc_seg_signature_exposes_image_size():
    """voc_seg 가 image_size 인자를 받고 기본값이 256 이어야 한다.

    실제 다운로드 없이 signature 만 검증 — 가변 크기 문제 회귀 방지.
    """
    import inspect

    sig = inspect.signature(pcq.datasets.voc_seg)
    assert "image_size" in sig.parameters
    assert sig.parameters["image_size"].default == 256


# ---------- datasets.cifar10 ----------


def test_cifar10_missing_path_raises_when_download_false(tmp_path):
    # 네트워크 없이 wrapper plumbing만 검증: 다운로드 없이 없는 경로 → 에러
    # torchvision 없는 환경(core-only)에서는 skip
    pytest.importorskip("torchvision")
    with pytest.raises((RuntimeError, FileNotFoundError)):
        pcq.datasets.cifar10(root=tmp_path / "nope", train=True, download=False)


# ---------- models.resnet18 ----------


def test_resnet18_forward_shape():
    # torchvision resnet18 wrapper. fc 가 num_classes 로 교체됨
    pytest.importorskip("torchvision")
    m = pcq.models.resnet18(num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    assert y.shape == (2, 5)


# ---------- models.text_classifier ----------


def test_text_classifier_forward_shape():
    # int64 token 시퀀스를 받아 (B, num_classes) 반환
    m = pcq.models.text_classifier(vocab_size=100, embed_dim=16, num_classes=3)
    x = torch.randint(0, 100, (4, 20))
    y = m(x)
    assert y.shape == (4, 3)


def test_text_classifier_repr_includes_meta():
    m = pcq.models.text_classifier(vocab_size=500, embed_dim=32, num_classes=5)
    r = repr(m)
    assert "vocab=500" in r and "embed=32" in r and "cls=5" in r


# ---------- datasets.fake_text ----------


def test_fake_text_dataset():
    ds = pcq.datasets.fake_text(
        num_samples=10, num_classes=3, vocab_size=50, seq_len=8
    )
    assert len(ds) == 10
    tokens, label = ds[0]
    assert tokens.shape == (8,)
    assert tokens.dtype == torch.int64
    # padding_idx=0 이 모델에서 예약되므로 dataset 도 1.. 부터
    assert tokens.min().item() >= 1
    assert 0 <= label.item() < 3


# ---------- datasets.mnist ----------


def test_mnist_smoke_no_download(tmp_path):
    """mnist 는 download=False + 데이터 없음 → RuntimeError 또는 FileNotFoundError."""
    pytest.importorskip("torchvision")
    with pytest.raises((RuntimeError, FileNotFoundError)):
        pcq.datasets.mnist(root=tmp_path / "nope", train=True, download=False)


# ---------- optim.adamw ----------


def test_adamw_step_updates_params():
    # 더미 손실에 대해 한 step 진행하면 파라미터가 변해야 한다
    model = nn.Linear(4, 2)
    before = model.weight.detach().clone()
    opt = pcq.optim.adamw(model.parameters(), lr=1e-2)

    x = torch.randn(8, 4)
    target = torch.randn(8, 2)
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    opt.step()

    assert not torch.equal(before, model.weight.detach())


def test_adamw_lr_and_weight_decay_passed():
    # factory가 lr/weight_decay를 그대로 forwarding 하는지 확인
    model = nn.Linear(2, 2)
    opt = pcq.optim.adamw(model.parameters(), lr=5e-4, weight_decay=0.1)
    g = opt.param_groups[0]
    assert g["lr"] == pytest.approx(5e-4)
    assert g["weight_decay"] == pytest.approx(0.1)


# ---------- sched.cosine ----------


def test_cosine_no_warmup_returns_pure_cosine():
    # warmup=0이면 첫 lr이 base_lr여야 한다 (warmup 없음)
    model = nn.Linear(2, 2)
    base_lr = 1e-3
    opt = pcq.optim.adamw(model.parameters(), lr=base_lr)
    sched = pcq.sched.cosine(opt, T_max=10, warmup=0)
    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr)
    sched.step()
    # 한 스텝 후에는 cosine 감소가 시작돼야 한다
    assert opt.param_groups[0]["lr"] < base_lr


def test_cosine_warmup_lr_curve_increases_then_decreases():
    # warmup=3 → 처음 3 스텝은 lr이 base_lr * 1e-3 → base_lr로 증가
    # 그 후 cosine이 감소
    model = nn.Linear(2, 2)
    base_lr = 1e-2
    opt = pcq.optim.adamw(model.parameters(), lr=base_lr)
    sched = pcq.sched.cosine(opt, T_max=10, warmup=3)

    # step 0 (warmup 시작): lr = base_lr * 1e-3 부근
    lr0 = opt.param_groups[0]["lr"]
    assert lr0 == pytest.approx(base_lr * 1e-3, rel=1e-2)

    sched.step()
    lr1 = opt.param_groups[0]["lr"]
    sched.step()
    lr2 = opt.param_groups[0]["lr"]

    # warmup 구간 lr 단조 증가
    assert lr1 > lr0
    assert lr2 > lr1

    # warmup 종료 후 cosine 감소
    sched.step()  # milestone 도달
    sched.step()
    lr_post = opt.param_groups[0]["lr"]
    sched.step()
    lr_post2 = opt.param_groups[0]["lr"]
    assert lr_post2 < lr_post


# ---------- loss.cross_entropy ----------


def test_cross_entropy_basic_returns_finite_loss():
    crit = pcq.loss.cross_entropy()
    logits = torch.randn(8, 10)
    labels = torch.randint(0, 10, (8,))
    loss = crit(logits, labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_cross_entropy_perfect_prediction_low_loss():
    # 정답 클래스에 매우 큰 logit을 주면 loss가 0에 가까워야 한다
    crit = pcq.loss.cross_entropy()
    logits = torch.full((4, 3), -10.0)
    logits[torch.arange(4), torch.tensor([0, 1, 2, 0])] = 10.0
    labels = torch.tensor([0, 1, 2, 0])
    loss = crit(logits, labels)
    assert loss.item() < 1e-3


def test_cross_entropy_ignore_index_excludes_target():
    """ignore_index=-1 적용 시 -1 라벨 픽셀이 loss 계산에서 제외돼야 한다.

    동일 logits 으로 (a) ignore_index=-1 + 1개 -1 라벨, (b) ignore 없이 같은 sample 만
    → loss 가 동일해야 한다 (수치 일치, atol=1e-5).
    """
    crit_ignore = pcq.loss.cross_entropy(ignore_index=-1)
    logits = torch.randn(4, 3)
    targets_with_ignore = torch.tensor([0, 1, -1, 2])  # idx 2 무시
    loss_with_ignore = crit_ignore(logits, targets_with_ignore)

    # ignore 가 적용되지 않은 동일 logits — idx 0,1,3 만 사용
    crit_plain = pcq.loss.cross_entropy()
    targets_subset = torch.tensor([0, 1, 2])
    loss_subset = crit_plain(logits[[0, 1, 3]], targets_subset)
    assert torch.allclose(loss_with_ignore, loss_subset, atol=1e-5)


def test_cross_entropy_weight_applies_class_weighting():
    """class weight 인자가 nn.CrossEntropyLoss 까지 forwarding 돼야 한다."""
    weight = torch.tensor([0.1, 0.5, 1.0])
    crit = pcq.loss.cross_entropy(weight=weight)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    loss = crit(logits, targets)
    assert torch.isfinite(loss)

    # 동일 logits/targets 를 unweighted 로 비교 — 값이 달라야 함 (weight 적용 증거)
    loss_unweighted = pcq.loss.cross_entropy()(logits, targets)
    assert not torch.allclose(loss, loss_unweighted)


# ---------- v1.3: models.unet ----------


def test_unet_forward_shape_preserves_hw():
    # UNet 은 입력과 동일한 H,W 유지 — (B,C,H,W) → (B,num_classes,H,W)
    m = pcq.models.unet(in_channels=3, num_classes=21, base_ch=16)
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    assert y.shape == (2, 21, 64, 64)


def test_unet_repr_includes_meta():
    m = pcq.models.unet(in_channels=3, num_classes=10, base_ch=8)
    r = repr(m)
    assert "in=3" in r and "classes=10" in r and "base=8" in r


def test_unet_grayscale_works():
    # 흑백 입력도 처리
    m = pcq.models.unet(in_channels=1, num_classes=2, base_ch=8)
    x = torch.randn(1, 1, 32, 32)
    y = m(x)
    assert y.shape == (1, 2, 32, 32)


# ---------- v1.3: datasets.fake_seg ----------


def test_fake_seg_dataset_basic():
    ds = pcq.datasets.fake_seg(num_samples=8, num_classes=5, image_size=32)
    assert len(ds) == 8
    img, mask = ds[0]
    assert img.shape == (3, 32, 32)
    assert mask.shape == (32, 32)
    assert mask.dtype == torch.int64
    # 라벨은 [0, num_classes) 범위
    assert mask.min().item() >= 0
    assert mask.max().item() < 5


def test_fake_seg_dataset_reproducibility():
    # 같은 파라미터 → 동일 첫 샘플 (seed=44 고정)
    ds1 = pcq.datasets.fake_seg(num_samples=4, num_classes=3, image_size=16)
    ds2 = pcq.datasets.fake_seg(num_samples=4, num_classes=3, image_size=16)
    img1, mask1 = ds1[0]
    img2, mask2 = ds2[0]
    assert torch.equal(img1, img2)
    assert torch.equal(mask1, mask2)


# ---------- v1.3: loss.dice / loss.focal ----------


def test_dice_loss_basic_finite():
    loss_fn = pcq.loss.dice()
    logits = torch.randn(2, 3, 4, 4)
    targets = torch.randint(0, 3, (2, 4, 4))
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    # dice loss = 1 - dice score → [0, 1] 범위
    assert 0.0 <= loss.item() <= 1.0


def test_dice_loss_perfect_prediction():
    # 정답 클래스에 매우 큰 logit → dice ~1, loss ~0
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = torch.full((1, 2, 4, 4), -10.0)
    logits[0, 0] = 10.0  # class 0 으로 강하게 예측
    loss = pcq.loss.dice()(logits, targets)
    assert loss.item() < 0.01


def test_dice_loss_ignores_index():
    # ignore_index=-1 인 픽셀은 loss 계산에서 제외
    targets = torch.full((1, 4, 4), -1, dtype=torch.long)
    targets[0, 0, 0] = 0  # 한 픽셀만 valid
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 0] = 10.0
    loss_fn = pcq.loss.dice(ignore_index=-1)
    loss = loss_fn(logits, targets)
    # ignored 픽셀이 dice 계산에 포함되지 않으므로 loss 매우 작음
    assert torch.isfinite(loss)
    assert loss.item() < 0.5


def test_focal_loss_basic_finite():
    loss_fn = pcq.loss.focal()
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,))
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)


def test_focal_loss_perfect_prediction():
    # 정답에 큰 logit → focal loss ~0
    crit = pcq.loss.focal(alpha=1.0, gamma=2.0)
    logits = torch.full((4, 3), -10.0)
    logits[torch.arange(4), torch.tensor([0, 1, 2, 0])] = 10.0
    labels = torch.tensor([0, 1, 2, 0])
    loss = crit(logits, labels)
    assert loss.item() < 1e-3
