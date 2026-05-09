"""pcq.agent.scaffold — generate project-local atom file skeletons.

Per kind, produces minimal-but-runnable code that the loader can import and
the smoke contract can validate end-to-end.

- model:   nn.Sequential AdaptiveAvgPool + Linear (2D classification baseline)
- loss:    nn.Module fallback to cross_entropy
- dataset: random tensor Dataset with split arg
- metric:  callable returning accuracy
- optim:   AdamW factory (params, lr, weight_decay)
- sched:   CosineAnnealingLR factory (optimizer, T_max)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pcq.agent.schema import ValidationCheck, ValidationReport


_TEMPLATES: dict[str, str] = {
    "model": '''\
"""Project-local model atoms — managed by user/agent.

Edit the factory implementation. Keep the @pcq.register_model decorator and
its meta block stable so pcq validate / pcq atoms smoke can verify the
contract.
"""
import pcq
import torch.nn as nn


@pcq.register_model(
    "{name}",
    meta={{
        "tasks": ["classification"],          # TODO: change task
        "params": {{
            "in_channels": {{"type": "int", "default": 3, "min": 1}},
            "num_classes": {{"type": "int", "default": 10, "min": 1}},
        }},
        "input_contract": {{"x": ["B", "in_channels", "H", "W"]}},
        "output_contract": {{"logits": ["B", "num_classes"]}},
        "smoke_safe": True,
        "description": "TODO: describe {name}",
    }},
)
def factory(in_channels: int = 3, num_classes: int = 10):
    """TODO: replace with your real model."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(in_channels, num_classes),
    )
''',
    "loss": '''\
"""Project-local loss atoms — managed by user/agent."""
import pcq
import torch
import torch.nn as nn


class _{class_name}(nn.Module):
    """TODO: replace with your real loss."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, target, ignore_index=self.ignore_index,
        )


pcq.register_loss(
    "{name}",
    factory=lambda ignore_index=-100: _{class_name}(ignore_index=ignore_index),
    meta={{
        "tasks": ["classification"],
        "params": {{
            "ignore_index": {{"type": "int", "default": -100}},
        }},
        "input_contract": {{
            "logits": ["B", "C", "..."],
            "target": ["B", "..."],
        }},
        "label_contract": {{
            "target_dtype": "int64",
            "valid_range": [0, "C-1"],
            "ignore_index_param": "ignore_index",
        }},
        "smoke_safe": True,
        "description": "TODO: describe {name}",
    }},
)
''',
    "dataset": '''\
"""Project-local dataset atoms — managed by user/agent."""
import pcq
import torch
from torch.utils.data import Dataset


class _{class_name}(Dataset):
    """TODO: replace with your real dataset."""

    def __init__(
        self,
        split: str = "train",
        num_samples: int = 32,
        num_classes: int = 10,
    ) -> None:
        self.split = split
        self.num_samples = num_samples
        self.num_classes = num_classes
        g = torch.Generator().manual_seed(0 if split == "train" else 1)
        self.x = torch.randn(num_samples, 3, 32, 32, generator=g)
        self.y = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


pcq.register_dataset(
    "{name}",
    factory=lambda split="train", num_samples=32, num_classes=10: _{class_name}(
        split=split, num_samples=num_samples, num_classes=num_classes,
    ),
    meta={{
        "tasks": ["classification"],
        "params": {{
            "split": {{"type": "str", "default": "train", "choices": ["train", "val"]}},
            "num_samples": {{"type": "int", "default": 32, "min": 1}},
            "num_classes": {{"type": "int", "default": 10, "min": 1}},
        }},
        "output_contract": {{"x": ["3", "32", "32"], "y": []}},
        "label_contract": {{"target_dtype": "int64", "valid_range": [0, "num_classes-1"]}},
        "smoke_safe": True,
        "description": "TODO: describe {name}",
    }},
)
''',
    "metric": '''\
"""Project-local metric atoms — managed by user/agent."""
import pcq
import torch


def _{name}_impl(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TODO: replace with your real metric. Must return a scalar tensor."""
    return (logits.argmax(-1) == target).float().mean()


pcq.register_metric(
    "{name}",
    factory=lambda: _{name}_impl,
    meta={{
        "tasks": ["classification"],
        "params": {{}},
        "input_contract": {{"logits": ["B", "C"], "target": ["B"]}},
        "metric_contract": {{"mode": "max"}},      # TODO: min or max?
        "smoke_safe": True,
        "description": "TODO: describe {name}",
    }},
)
''',
    "optim": '''\
"""Project-local optimizer atoms — managed by user/agent."""
import pcq
import torch


pcq.register_optim(
    "{name}",
    factory=lambda params, lr=1e-3, weight_decay=0.0: torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay,
    ),
    meta={{
        "params": {{
            "lr": {{"type": "float", "default": 1e-3, "min": 0.0}},
            "weight_decay": {{"type": "float", "default": 0.0, "min": 0.0}},
        }},
        "smoke_safe": True,
        "description": "TODO: describe {name}",
    }},
)
''',
    "sched": '''\
"""Project-local scheduler atoms — managed by user/agent."""
import pcq
import torch


pcq.register_sched(
    "{name}",
    factory=lambda optimizer, T_max, warmup=0: torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max,
    ),
    meta={{
        "params": {{
            "T_max": {{"type": "int", "required": True, "min": 1}},
            "warmup": {{"type": "int", "default": 0, "min": 0}},
        }},
        "smoke_safe": True,
        "description": "TODO: describe {name}",
    }},
)
''',
}


@dataclass
class ScaffoldResult:
    schema_version: int = 1
    status: str = "created"               # "created" | "skipped" | "error"
    files_changed: list[str] = field(default_factory=list)
    atom: dict = field(default_factory=dict)
    next_checks: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "schema_version": self.schema_version,
            "status": self.status,
            "files_changed": self.files_changed,
            "atom": self.atom,
            "next_checks": self.next_checks,
        }
        if self.error:
            d["error"] = self.error
        return d


_PLURAL_MAP: dict[str, str] = {
    "model": "models",
    "loss": "losses",
    "dataset": "datasets",
    "metric": "metrics",
    "optim": "optims",
    "sched": "scheds",
}


def _resolve_default_path(kind: str, project_root: Path) -> Path:
    """기본 출력 경로: atoms/<plural>.py."""
    return project_root / "atoms" / f"{_PLURAL_MAP[kind]}.py"


def _is_valid_atom_name(name: str) -> bool:
    """Python identifier 검증 — 문자 시작 + (alnum | _) 만."""
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


def scaffold_atom(
    kind: str,
    name: str,
    output: str | Path | None = None,
    project_root: str | Path = ".",
    force: bool = False,
) -> ScaffoldResult:
    """Generate project-local atom file skeleton.

    Returns ScaffoldResult with files_changed and next_checks.
    """
    result = ScaffoldResult()
    if kind not in _TEMPLATES:
        result.status = "error"
        result.error = (
            f"unknown kind {kind!r}; supported: {sorted(_TEMPLATES)}"
        )
        return result

    if not _is_valid_atom_name(name):
        result.status = "error"
        result.error = (
            f"invalid atom name {name!r}; must be valid Python identifier"
        )
        return result

    project_root_p = Path(project_root).resolve()
    project_root_p.mkdir(parents=True, exist_ok=True)

    if output is None:
        output_path = _resolve_default_path(kind, project_root_p)
    else:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = project_root_p / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # atoms 디렉토리에 __init__.py 보장 (default 위치 사용 시)
    atoms_init = output_path.parent / "__init__.py"
    if not atoms_init.exists():
        atoms_init.write_text("", encoding="utf-8")
        try:
            rel = atoms_init.relative_to(project_root_p)
            result.files_changed.append(str(rel))
        except ValueError:
            result.files_changed.append(str(atoms_init))

    class_name = "".join(part.capitalize() for part in name.split("_") if part)

    new_block = _TEMPLATES[kind].format(name=name, class_name=class_name)

    if output_path.exists() and not force:
        existing = output_path.read_text(encoding="utf-8")
        if f'"{name}"' in existing:
            result.status = "skipped"
            result.error = (
                f"atom {name!r} already present in {output_path.name}; "
                "use --force to recreate"
            )
            return result
        # 기존 파일에 새 atom 블록 append (사용자가 import 중복 정리)
        output_path.write_text(
            existing.rstrip() + "\n\n\n" + new_block, encoding="utf-8"
        )
    else:
        output_path.write_text(new_block, encoding="utf-8")

    try:
        rel = output_path.relative_to(project_root_p)
        result.files_changed.append(str(rel))
    except ValueError:
        result.files_changed.append(str(output_path))
    result.atom = {"kind": kind, "name": name}
    result.next_checks = [
        f"pcq atoms validate-local {project_root_p}",
        f"pcq atoms smoke {kind} {name}",
        "pcq validate",
    ]
    return result


def validate_local_atoms(project_root: str | Path = ".") -> ValidationReport:
    """Project atom 전용 검증.

    Checks:
      - load_project_atoms 성공 (import error → fail)
      - 등록된 project atom 의 contract 완전성
        (model: input/output, dataset: output, loss: input, metric: mode)
    """
    from pcq import registry as registry_pkg
    from pcq.registry.loader import load_project_atoms

    report = ValidationReport()
    load_report = load_project_atoms(project_root)

    if not load_report.modules_loaded and not load_report.errors:
        report.add(ValidationCheck(
            id="project_atoms_present",
            status="warn",
            severity="info",
            detail="no pcq_atoms.py or atoms/ directory found",
        ))
        return report

    # import errors
    for err in load_report.errors:
        report.add(ValidationCheck(
            id="project_atom_import",
            status="fail",
            severity="blocking",
            detail=f"{err['module']}: {err['error']}",
        ))

    REG_MAP = {
        "model": registry_pkg.models,
        "dataset": registry_pkg.datasets,
        "loss": registry_pkg.losses,
        "optim": registry_pkg.optims,
        "sched": registry_pkg.scheds,
        "metric": registry_pkg.metrics,
    }

    for entry in load_report.atoms_registered:
        kind = entry["kind"]
        name = entry["name"]
        module = entry["module"]
        spec = REG_MAP[kind].get(name)

        # tasks 권장 (model/dataset/loss/metric)
        if kind in ("model", "dataset", "loss", "metric") and not spec.tasks:
            report.add(ValidationCheck(
                id="project_atom_tasks",
                status="warn",
                severity="warning",
                detail=f"{kind}/{name}: tasks list is empty",
                suggested_fix=f"add tasks to {module} register meta",
            ))

        if kind == "model":
            if not spec.input_contract or not spec.output_contract:
                report.add(ValidationCheck(
                    id="project_atom_model_contract",
                    status="fail",
                    severity="blocking",
                    detail=(
                        f"model/{name}: missing input_contract or "
                        "output_contract"
                    ),
                    suggested_fix=(
                        f"add input_contract and output_contract to "
                        f"{module} meta"
                    ),
                ))
        elif kind == "dataset":
            if not spec.output_contract:
                report.add(ValidationCheck(
                    id="project_atom_dataset_contract",
                    status="fail",
                    severity="blocking",
                    detail=f"dataset/{name}: missing output_contract",
                ))
        elif kind == "loss":
            if not spec.input_contract:
                report.add(ValidationCheck(
                    id="project_atom_loss_contract",
                    status="warn",
                    severity="warning",
                    detail=f"loss/{name}: missing input_contract",
                ))
        elif kind == "metric":
            if not spec.metric_contract.get("mode"):
                report.add(ValidationCheck(
                    id="project_atom_metric_mode",
                    status="warn",
                    severity="warning",
                    detail=(
                        f"metric/{name}: missing metric_contract.mode (min/max)"
                    ),
                ))

    if not report.checks:
        report.add(ValidationCheck(
            id="project_atoms_loaded",
            status="pass",
            severity="info",
            detail=(
                f"loaded {len(load_report.atoms_registered)} project atoms "
                "with valid contracts"
            ),
        ))

    return report
