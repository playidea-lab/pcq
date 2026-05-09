"""v2.4: AtomSpec.role field — builtin = reference_example, project = user.

기존 source 필드는 그대로 유지하되, role 이 atom 의 의도적 위치(목적)를 나타낸다.
- builtin atoms (pcq 내부 등록) → role="reference_example"
- 사용자/project atom (meta 에 source="project") → role="user"
- meta 에 명시적으로 role 지정 가능.
"""
from __future__ import annotations

from pcq import registry


def test_builtin_model_has_reference_example_role():
    spec = registry.models.get("mlp")
    assert spec.source == "builtin"
    assert spec.role == "reference_example"


def test_builtin_loss_has_reference_example_role():
    spec = registry.losses.get("cross_entropy")
    assert spec.source == "builtin"
    assert spec.role == "reference_example"


def test_builtin_dataset_has_reference_example_role():
    spec = registry.datasets.get("fake")
    assert spec.role == "reference_example"


def test_builtin_optim_has_reference_example_role():
    spec = registry.optims.get("adamw")
    assert spec.role == "reference_example"


def test_builtin_sched_has_reference_example_role():
    spec = registry.scheds.get("cosine")
    assert spec.role == "reference_example"


def test_builtin_metric_has_reference_example_role():
    spec = registry.metrics.get("accuracy")
    assert spec.role == "reference_example"


def test_to_dict_exposes_role():
    spec = registry.losses.get("cross_entropy")
    d = spec.to_dict()
    assert d["role"] == "reference_example"


def test_all_builtin_atoms_have_reference_example_role():
    """모든 6 카테고리의 builtin atom 이 role='reference_example'."""
    REGS = {
        "model": registry.models,
        "dataset": registry.datasets,
        "loss": registry.losses,
        "optim": registry.optims,
        "sched": registry.scheds,
        "metric": registry.metrics,
    }
    for kind, reg in REGS.items():
        for name in reg.list():
            spec = reg.get(name)
            assert spec.role == "reference_example", (
                f"{kind}/{name} role={spec.role!r}, expected 'reference_example' "
                f"(source={spec.source!r})"
            )


def test_explicit_user_role_via_meta():
    """meta 에서 role='user' 명시 가능 (project atom path)."""
    from pcq.registry import Registry
    reg = Registry("loss")
    reg.register(
        "user_atom",
        factory=lambda: None,
        meta={"source": "project", "role": "user"},
    )
    spec = reg.get("user_atom")
    assert spec.source == "project"
    assert spec.role == "user"


def test_project_source_defaults_role_user():
    """source='project' → role 미지정 시 'user' default."""
    from pcq.registry import Registry
    reg = Registry("loss")
    reg.register(
        "project_atom_no_role",
        factory=lambda: None,
        meta={"source": "project"},
    )
    spec = reg.get("project_atom_no_role")
    assert spec.source == "project"
    assert spec.role == "user"


def test_inferred_atom_role_inferred_from_source():
    """meta=None (legacy) → source='builtin' default → role='reference_example'."""
    from pcq.registry import Registry
    reg = Registry("loss")
    reg.register("legacy_atom", factory=lambda: None, meta=None)
    spec = reg.get("legacy_atom")
    assert spec.metadata_status == "inferred"
    assert spec.source == "builtin"
    assert spec.role == "reference_example"
