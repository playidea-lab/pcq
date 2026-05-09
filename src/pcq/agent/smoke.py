"""pcq.agent.smoke — atom smoke contract.

Per-kind 1-step verification:
  model:   small input shape derived from input_contract → factory().forward()
  loss:    factory()(logits, target) → forward + backward
  dataset: factory().__getitem__(0) → check shape
  metric:  factory()(logits, target) → finite scalar
  optim:   factory(dummy_params).step() OK
  sched:   factory(dummy_optim, T_max=2) → 2 steps OK
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SmokeReport:
    schema_version: int = 1
    kind: str = ""
    name: str = ""
    passed: bool = False
    detail: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }
        if self.error:
            d["error"] = self.error
        return d


_DEFAULT_BATCH = 2
_DEFAULT_HW = 8
_DEFAULT_C = 3
# Top-5 metric 등 C>=5 가 필요한 atom 도 안전하게 통과하도록 default num_classes 는 10
_DEFAULT_NUM_CLASSES = 10


def _resolve_shape(contract: list, params: dict) -> tuple:
    """input_contract list → 구체 shape tuple. Symbol resolution rules:

    - "B"           → 2
    - "H" / "W"     → 8
    - "C"           → num_classes (or 3)
    - 정수 문자열   → int
    - "..."         → 생략 (skipped — 가변 dim, 자동 trimmed)
    - param 이름    → spec.params[name].default (int 가능 시), else 4
    - 알 수 없음    → 4
    """
    out: list[int] = []
    for token in contract:
        if isinstance(token, int):
            out.append(token)
            continue
        if not isinstance(token, str):
            out.append(_DEFAULT_BATCH)
            continue
        if token == "...":
            # 가변 차원 — smoke 에선 무시 (caller 가 결정)
            continue
        if token == "B":
            out.append(_DEFAULT_BATCH)
        elif token in ("H", "W"):
            out.append(_DEFAULT_HW)
        elif token == "C":
            # C 는 보통 num_classes 와 연결
            num_classes = _resolve_num_classes(params)
            out.append(num_classes)
        elif token.isdigit():
            out.append(int(token))
        elif token in params:
            pspec = params[token]
            d = getattr(pspec, "default", None)
            if isinstance(d, int) and d > 0:
                out.append(d)
            else:
                out.append(4)
        else:
            out.append(4)
    return tuple(out)


def _resolve_num_classes(params: dict) -> int:
    """params 에서 num_classes / out_dim 류 값 추출. 못 찾으면 default."""
    for key in ("num_classes", "out_dim", "C"):
        if key in params:
            d = getattr(params[key], "default", None)
            if isinstance(d, int) and d > 0:
                return d
    return _DEFAULT_NUM_CLASSES


def _build_loss_inputs(spec) -> tuple[torch.Tensor, torch.Tensor]:
    """loss spec.input_contract 기반으로 (logits, target) 합성.

    - logits shape: input_contract.logits 에서 추론 (없으면 (B,C))
    - target shape: input_contract.target 에서 추론 (없으면 (B,))
    - "..." 가 있으면 (B, C) (classification) 또는 (B, C, H, W) (seg) 로 추정
    """
    logits_contract = list(spec.input_contract.get("logits", []))
    target_contract = list(spec.input_contract.get("target", []))
    num_classes = _resolve_num_classes(spec.params)

    # "..." → seg 가정 시 H, W 추가; classification 가정 시 그대로
    if "..." in logits_contract:
        # classification baseline — (B, C)
        logits_shape: tuple = (_DEFAULT_BATCH, num_classes)
        target_shape: tuple = (_DEFAULT_BATCH,)
    else:
        logits_shape = _resolve_shape(logits_contract, spec.params)
        target_shape = _resolve_shape(target_contract, spec.params)
        if not logits_shape:
            logits_shape = (_DEFAULT_BATCH, num_classes)
        if not target_shape:
            target_shape = (_DEFAULT_BATCH,)

    logits = torch.randn(*logits_shape, requires_grad=True)
    target = torch.randint(0, num_classes, target_shape)
    return logits, target


def _build_metric_inputs(spec) -> tuple:
    """metric spec.input_contract 기반으로 (input1, input2) 합성.

    - classification: (B, C) logits + (B,) target
    - segmentation: (B, C, H, W) logits + (B, H, W) target
    - regression: (B,) predictions + (B,) target (predictions/target 동일 shape)
    """
    contract = spec.input_contract
    num_classes = _resolve_num_classes(spec.params)

    if "predictions" in contract and "target" in contract:
        # regression — same shape
        shape = (_DEFAULT_BATCH,)
        return (torch.randn(*shape), torch.randn(*shape))

    logits_contract = list(contract.get("logits", []))
    target_contract = list(contract.get("target", []))

    if "..." in logits_contract:
        logits_shape: tuple = (_DEFAULT_BATCH, num_classes)
        target_shape: tuple = (_DEFAULT_BATCH,)
    else:
        logits_shape = _resolve_shape(logits_contract, spec.params)
        target_shape = _resolve_shape(target_contract, spec.params)
        if not logits_shape:
            logits_shape = (_DEFAULT_BATCH, num_classes)
        if not target_shape:
            target_shape = (_DEFAULT_BATCH,)

    logits = torch.randn(*logits_shape)
    target = torch.randint(0, num_classes, target_shape)
    return (logits, target)


def smoke_atom(kind: str, name: str) -> SmokeReport:
    """Run a 1-step contract verification for one atom."""
    from pcq import registry

    REG_MAP: dict[str, Any] = {
        "model": registry.models,
        "dataset": registry.datasets,
        "loss": registry.losses,
        "optim": registry.optims,
        "sched": registry.scheds,
        "metric": registry.metrics,
    }

    report = SmokeReport(kind=kind, name=name)
    reg = REG_MAP.get(kind)
    if reg is None:
        report.error = f"unknown kind {kind!r}"
        return report

    try:
        spec = reg.get(name)
    except ValueError as e:
        report.error = str(e)
        return report

    try:
        if kind == "model":
            model = spec.factory()
            in_contract = spec.input_contract.get("x", [])
            shape = _resolve_shape(in_contract, spec.params)
            if not shape:
                shape = (_DEFAULT_BATCH, 3, _DEFAULT_HW, _DEFAULT_HW)
            # vocab_size 가 있으면 임베딩 입력으로 추정 → int64 token 입력
            if "vocab_size" in spec.params:
                vocab_pspec = spec.params["vocab_size"]
                vocab_default = getattr(vocab_pspec, "default", None) or 100
                x = torch.randint(0, max(2, int(vocab_default)), shape)
            else:
                x = torch.randn(*shape)
            with torch.no_grad():
                out = model(x)
            shape_str = (
                tuple(out.shape) if hasattr(out, "shape") else "n/a"
            )
            report.detail = f"forward OK, output shape: {shape_str}"
            report.passed = True

        elif kind == "loss":
            loss_fn = spec.factory()
            logits, target = _build_loss_inputs(spec)
            out = loss_fn(logits, target)
            out.backward()
            report.detail = (
                f"forward+backward OK, loss={float(out.detach()):.4f}"
            )
            report.passed = True

        elif kind == "dataset":
            ds = spec.factory()
            sample = ds[0]
            report.detail = (
                f"len={len(ds)}, sample_type={type(sample).__name__}"
            )
            report.passed = True

        elif kind == "metric":
            metric_fn = spec.factory()
            args = _build_metric_inputs(spec)
            val = metric_fn(*args)
            if hasattr(val, "isfinite"):
                if not bool(torch.isfinite(val).all()):
                    raise ValueError(f"metric returned non-finite: {val}")
            report.detail = f"value={float(val):.4f}"
            report.passed = True

        elif kind == "optim":
            param = torch.nn.Parameter(torch.randn(4))
            opt = spec.factory([param])
            param.grad = torch.randn_like(param)
            opt.step()
            report.detail = "step OK"
            report.passed = True

        elif kind == "sched":
            param = torch.nn.Parameter(torch.randn(4))
            opt = torch.optim.SGD([param], lr=1e-3)
            sched = spec.factory(opt, T_max=2)
            sched.step()
            sched.step()
            report.detail = "2 steps OK"
            report.passed = True

    except Exception as e:  # noqa: BLE001
        report.error = f"{type(e).__name__}: {e}"

    return report
