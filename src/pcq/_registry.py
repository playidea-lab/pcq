"""Backward compat shim — actual implementation moved to pcq.registry package.

기존 `from pcq import _registry` 또는 `from pcq._registry import models` 패턴을
유지하기 위한 thin re-export. 신규 코드는 `from pcq.registry import ...` 사용.
"""
from pcq.registry import (  # noqa: F401
    AtomRef,
    AtomSpec,
    ParamSpec,
    Registry,
    datasets,
    losses,
    metrics,
    models,
    optims,
    scheds,
)
