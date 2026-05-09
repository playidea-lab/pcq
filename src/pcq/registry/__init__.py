"""pcq.registry — metadata-aware atom registries.

기존 pcq._registry 대체. 호환 유지: pcq.register_model 등은 그대로.

확장 패턴 (v1.8+):
    pcq.register_loss(
        "boundary_dice",
        factory=lambda smooth=1.0: BoundaryDiceLoss(smooth=smooth),
        meta={
            "tasks": ["segmentation"],
            "params": {"smooth": {"type": "float", "default": 1.0, "min": 0.0}},
            "input_contract": {"logits": ["B","C","H","W"], "target": ["B","H","W"]},
        },
    )

기존 형태 (`pcq.register_X("name", factory)`)도 유지 — meta=None → inferred.
"""
from __future__ import annotations

from typing import Any, Callable

from pcq.registry.spec import AtomRef, AtomSpec, ParamSpec


class Registry:
    """Metadata-aware atom registry.

    - register(name, factory, meta=None) — direct
    - register(name, meta=None) — decorator (factory 가 함수에 부착됨)
    - get(name) -> AtomSpec
    - build(name, *args, **kwargs) -> object (with param validation if explicit)
    - meta(name) -> dict (JSON-safe)
    - build_ref(ref) -> object
    - validate_ref(ref) -> list[str]
    - list() -> sorted names
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, AtomSpec] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any] | None = None,
        meta: dict | None = None,
    ) -> Any:
        """register("name", factory, meta=...) 또는 @register("name", meta=...) 데코레이터.

        factory 가 None 이면 데코레이터 모드.
        """
        if factory is not None:
            spec = AtomSpec.from_meta(self.kind, name, factory, meta)
            self._items[name] = spec
            return factory

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            spec = AtomSpec.from_meta(self.kind, name, fn, meta)
            self._items[name] = spec
            return fn

        return decorator

    def get(self, name: str) -> AtomSpec:
        if name not in self._items:
            raise ValueError(
                f"unknown {self.kind} {name!r}; registered: {sorted(self._items)}"
            )
        return self._items[name]

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """factory 호출 + (메타가 explicit 이면) param validation."""
        spec = self.get(name)
        if spec.metadata_status == "explicit" and spec.params:
            errors = spec.validate_params(kwargs)
            if errors:
                raise ValueError(
                    f"{self.kind} {name!r} param errors: {errors}"
                )
        return spec.factory(*args, **kwargs)

    def meta(self, name: str) -> dict:
        return self.get(name).to_dict()

    def build_ref(self, ref: AtomRef) -> Any:
        if ref.kind != self.kind:
            raise ValueError(
                f"ref.kind {ref.kind!r} != registry.kind {self.kind!r}"
            )
        return self.build(ref.name, **ref.params)

    def validate_ref(self, ref: AtomRef) -> list[str]:
        if ref.kind != self.kind:
            return [
                f"kind mismatch: ref={ref.kind!r}, registry={self.kind!r}"
            ]
        try:
            spec = self.get(ref.name)
        except ValueError as e:
            return [str(e)]
        # inferred 메타는 strict param check 안 함 (legacy 호환)
        if spec.metadata_status == "inferred":
            return []
        return spec.validate_params(ref.params)

    def list(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


# 6 카테고리 단일 인스턴스 (atom 모듈 import 시 자동 등록)
models = Registry("model")
datasets = Registry("dataset")
losses = Registry("loss")
optims = Registry("optim")
scheds = Registry("sched")
metrics = Registry("metric")


# v1.12: project atom auto-discovery — re-export for pcq.registry.* convenience
from pcq.registry.loader import (  # noqa: E402
    LoadReport,
    list_sources,
    load_project_atoms,
)

__all__ = [
    "AtomRef",
    "AtomSpec",
    "LoadReport",
    "ParamSpec",
    "Registry",
    "datasets",
    "list_sources",
    "load_project_atoms",
    "losses",
    "metrics",
    "models",
    "optims",
    "scheds",
]
