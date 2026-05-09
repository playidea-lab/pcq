"""pcq.examples reference namespace.

Reference implementations live in pcq.examples.{models,datasets,optim,sched};
pcq.{models,datasets,optim,sched} re-export them as compatibility facades.
Loss and metric categories currently remain module aliases.
"""
from __future__ import annotations


def test_examples_models_is_reference_module_not_facade():
    import pcq
    import pcq.examples
    assert pcq.examples.models is not pcq.models
    assert pcq.examples.models.mlp is pcq.models.mlp
    assert pcq.examples.models.small_cnn is pcq.models.small_cnn


def test_examples_datasets_is_reference_module_not_facade():
    import pcq
    import pcq.examples
    assert pcq.examples.datasets is not pcq.datasets
    assert pcq.examples.datasets.fake is pcq.datasets.fake
    assert pcq.examples.datasets.voc_seg is pcq.datasets.voc_seg


def test_examples_loss_mirrors_cq_loss():
    import pcq
    import pcq.examples
    assert pcq.examples.loss is pcq.loss


def test_examples_optim_is_reference_module_not_facade():
    import pcq
    import pcq.examples
    assert pcq.examples.optim is not pcq.optim
    assert pcq.examples.optim.adamw is pcq.optim.adamw


def test_examples_sched_is_reference_module_not_facade():
    import pcq
    import pcq.examples
    assert pcq.examples.sched is not pcq.sched
    assert pcq.examples.sched.cosine is pcq.sched.cosine


def test_examples_metric_mirrors_cq_metric():
    import pcq
    import pcq.examples
    assert pcq.examples.metric is pcq.metric


def test_examples_all_modules_mirrored():
    """Reference modules are separate; loss/metric remain aliases."""
    import pcq
    import pcq.examples
    for kind in ("models", "datasets", "optim", "sched"):
        assert getattr(pcq.examples, kind) is not getattr(pcq, kind), (
            f"pcq.examples.{kind} should be the implementation module"
        )
    for kind in ("loss", "metric"):
        assert getattr(pcq.examples, kind) is getattr(pcq, kind), (
            f"pcq.examples.{kind} not aliased to pcq.{kind}"
        )


def test_examples_atom_call_works():
    """pcq.examples.models.mlp(...) ≡ pcq.models.mlp(...) — same factory call."""
    import pcq
    import pcq.examples
    m1 = pcq.models.mlp(10, [4], 2)
    m2 = pcq.examples.models.mlp(10, [4], 2)
    # 같은 factory → 같은 클래스
    assert type(m1) is type(m2)


def test_examples_loss_call_works():
    """pcq.examples.loss.cross_entropy() returns same module type."""
    import pcq
    import pcq.examples
    l1 = pcq.loss.cross_entropy(ignore_index=-1)
    l2 = pcq.examples.loss.cross_entropy(ignore_index=-1)
    assert type(l1) is type(l2)


def test_examples_dataset_call_works():
    """pcq.examples.datasets.fake(...) uses the same factory as pcq.datasets.fake(...)."""
    import pcq
    import pcq.examples
    d1 = pcq.datasets.fake(num_samples=4)
    d2 = pcq.examples.datasets.fake(num_samples=4)
    assert type(d1) is type(d2)


def test_examples_optim_and_sched_call_works():
    """Optimizer/scheduler reference examples share the facade factories."""
    import torch.nn as nn
    import pcq
    import pcq.examples

    model = nn.Linear(2, 2)
    opt = pcq.examples.optim.adamw(model.parameters(), lr=1e-3)
    sched = pcq.examples.sched.cosine(opt, T_max=2)
    assert type(opt) is type(pcq.optim.adamw(model.parameters(), lr=1e-3))
    assert type(sched) is type(pcq.sched.cosine(opt, T_max=2))


def test_examples_exposed_via_cq():
    """import pcq → pcq.examples 접근 가능 (pcq.__init__ 에서 export)."""
    import pcq
    assert hasattr(pcq, "examples")
    assert pcq.examples.models.mlp is pcq.models.mlp
    assert pcq.examples.datasets.fake is pcq.datasets.fake


def test_examples_in_all():
    """pcq.__all__ 에 'examples' 포함."""
    import pcq
    assert "examples" in pcq.__all__
