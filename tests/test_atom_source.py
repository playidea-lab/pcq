"""AtomSpec.source + module field 검증 (v1.12)."""
from pcq import registry


def test_builtin_loss_has_source_builtin():
    spec = registry.losses.get("cross_entropy")
    assert spec.source == "builtin"
    # pcq.loss 모듈에서 등록됨
    assert "loss" in spec.module


def test_builtin_model_has_source_builtin():
    spec = registry.models.get("mlp")
    assert spec.source == "builtin"
    assert "models" in spec.module


def test_atomspec_to_dict_includes_source_and_module():
    spec = registry.models.get("mlp")
    d = spec.to_dict()
    assert d["source"] == "builtin"
    assert "module" in d
    assert d["module"]


def test_all_builtin_atoms_marked_builtin():
    """모든 6 카테고리의 등록 atom 이 source='builtin'."""
    REGS = {
        "model": registry.models, "dataset": registry.datasets,
        "loss": registry.losses, "optim": registry.optims,
        "sched": registry.scheds, "metric": registry.metrics,
    }
    for kind, reg in REGS.items():
        for name in reg.list():
            spec = reg.get(name)
            assert spec.source == "builtin", (
                f"{kind}/{name} has source={spec.source!r}, expected 'builtin'"
            )


def test_inferred_atom_still_has_source_default():
    """meta=None 으로 등록된 atom 도 source='builtin' default."""
    from pcq.registry import Registry
    reg = Registry("loss")
    reg.register("legacy_test_atom", factory=lambda: None, meta=None)
    spec = reg.get("legacy_test_atom")
    assert spec.metadata_status == "inferred"
    assert spec.source == "builtin"
