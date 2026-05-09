"""pcq.registry.loader — project atom auto-discovery.

import 대상:
  - <path>/pcq_atoms.py  (단일 진입점, 권장)
  - <path>/atoms/*.py    (glob, 옵션)

Side effects:
  - sys.path 에 path 임시 추가 (with context manager 정리)
  - import 시점에 pcq.register_* 호출되며 registry 등록
  - 등록된 spec 의 source 는 자동으로 "project" 마킹
"""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from pcq.registry import datasets, losses, metrics, models, optims, scheds


@dataclass
class LoadReport:
    schema_version: int = 1
    project_root: str = ""
    modules_loaded: list[str] = field(default_factory=list)
    atoms_registered: list[dict] = field(default_factory=list)   # [{kind, name, module}]
    errors: list[dict] = field(default_factory=list)             # [{module, error}]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "project_root": self.project_root,
            "modules_loaded": self.modules_loaded,
            "atoms_registered": self.atoms_registered,
            "errors": self.errors,
        }


@contextmanager
def _temp_sys_path(path: Path):
    """sys.path 앞에 임시로 path 를 추가하는 context manager."""
    p = str(path)
    sys.path.insert(0, p)
    try:
        yield
    finally:
        try:
            sys.path.remove(p)
        except ValueError:
            pass


def _snapshot_registries() -> dict[str, set[str]]:
    """현재 6 개 registry 의 등록 atom 이름 snapshot."""
    return {
        "model": set(models.list()),
        "dataset": set(datasets.list()),
        "loss": set(losses.list()),
        "optim": set(optims.list()),
        "sched": set(scheds.list()),
        "metric": set(metrics.list()),
    }


def _diff_and_mark(before: dict[str, set[str]], module_name: str) -> list[dict]:
    """before snapshot 이후 새로 등록된 atom 들을 source='project' 로 마킹."""
    REGS = {
        "model": models, "dataset": datasets, "loss": losses,
        "optim": optims, "sched": scheds, "metric": metrics,
    }
    new_atoms: list[dict] = []
    for kind, reg in REGS.items():
        after = set(reg.list())
        for name in after - before[kind]:
            spec = reg.get(name)
            spec.source = "project"
            # module 은 factory.__module__ 으로 이미 채워져 있으나, 호환 강화
            if not spec.module:
                spec.module = module_name
            new_atoms.append({"kind": kind, "name": name, "module": spec.module or module_name})
    return new_atoms


def load_project_atoms(path: str | Path = ".") -> LoadReport:
    """Project-local atom modules 자동 import + registry 등록.

    우선순위:
      1. <path>/pcq_atoms.py 가 있으면 import (권장 단일 진입점)
      2. <path>/atoms/__init__.py 가 있으면 atoms 패키지 import
      3. 위가 없고 <path>/atoms/*.py 가 있으면 namespace 로 개별 import

    pcq.register_* 호출은 import 시 자동 실행. 등록된 spec 은 source="project" 로 마킹.
    """
    root = Path(path).resolve()
    report = LoadReport(project_root=str(root))

    if not root.exists() or not root.is_dir():
        return report

    candidates: list[tuple[str, Path]] = []
    cq_atoms_py = root / "pcq_atoms.py"
    if cq_atoms_py.exists():
        candidates.append(("pcq_atoms", cq_atoms_py))

    atoms_dir = root / "atoms"
    if atoms_dir.exists() and atoms_dir.is_dir():
        init_py = atoms_dir / "__init__.py"
        if init_py.exists():
            # __init__.py 가 있으면 패키지 + 자식 모듈 모두 import
            # (사용자가 __init__.py 에서 자식을 import 안 했어도 atom 등록 보장)
            candidates.append(("atoms", init_py))
        for py_file in sorted(atoms_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            mod_name = f"atoms.{py_file.stem}"
            # 중복 추가 방지
            if not any(c[0] == mod_name for c in candidates):
                candidates.append((mod_name, py_file))

    if not candidates:
        return report

    with _temp_sys_path(root):
        for mod_name, _mod_path in candidates:
            before = _snapshot_registries()
            try:
                if mod_name in sys.modules:
                    # 멱등성 — 이미 import 된 경우 reload 해서 register 재실행
                    importlib.reload(sys.modules[mod_name])
                else:
                    importlib.import_module(mod_name)
                report.modules_loaded.append(mod_name)
                new_atoms = _diff_and_mark(before, mod_name)
                report.atoms_registered.extend(new_atoms)
            except Exception as e:  # noqa: BLE001
                report.errors.append({
                    "module": mod_name,
                    "error": f"{type(e).__name__}: {e}",
                })

    return report


def list_sources() -> dict[str, list[str]]:
    """source 별 atom 분류. 키: builtin | project | generated | external."""
    REGS = {
        "model": models, "dataset": datasets, "loss": losses,
        "optim": optims, "sched": scheds, "metric": metrics,
    }
    out: dict[str, list[str]] = {
        "builtin": [], "project": [], "generated": [], "external": [],
    }
    for kind, reg in REGS.items():
        for name in reg.list():
            spec = reg.get(name)
            src = spec.source or "builtin"
            out.setdefault(src, []).append(f"{kind}/{name}")
    for src in out:
        out[src].sort()
    return out
