"""pcq — ML authoring library on top of cq.yaml runtime contract.

이 패키지는 cq.yaml 런타임 계약(CQ_CONFIG_JSON, stdout @key=value, output_dir)을
감싸는 편의 계층이다.

Note: cq.yaml / CQ_CONFIG_JSON / cq:// URI 스킴은 CQ Go service contract
specification 으로, pcq 라이브러리 이름과 무관하게 그대로 유지된다.
"""
from pcq import _registry, agent, datasets, loss, metric, models, optim, sched, testing
# v2.7: examples namespace — reference example implementations live under
# pcq.examples.*. pcq.{models,datasets,optim,sched} remain compatibility facades.
from pcq import examples
from pcq.agent import (
    ResolvedConfig,
    RunContext,
    compare_runs,
    describe_run,
    diff_recipes,
    inspect_project,
    list_meta,
    recipe_meta,
    resolve_project,
    resolve_run_context,
    summarize_run,
    validate_project,
)
from pcq.contract import (
    finalize_run,
    save_all,
    save_config_snapshot,
    save_manifest,
    save_metrics,
    save_partial_run_record,
    save_run_summary,
)
from pcq.core import config, input_dir, log, output_dir, seed_everything
from pcq.experiment import Experiment
from pcq.registry.spec import AtomRef, AtomSpec, ParamSpec
from pcq.trainer import Trainer

# 확장 API: Registry 함수/데코레이터 폼
register_model = _registry.models.register
register_dataset = _registry.datasets.register
register_loss = _registry.losses.register
register_optim = _registry.optims.register
register_sched = _registry.scheds.register
register_metric = _registry.metrics.register


# Ref constructors (v1.8+) — agent 가 직렬화 가능한 atom 참조 생성
def model_ref(name: str, params: dict | None = None) -> AtomRef:
    """model AtomRef 생성. registry 의 model atom 을 직렬화 가능한 형태로 참조."""
    return AtomRef(kind="model", name=name, params=dict(params or {}))


def dataset_ref(name: str, params: dict | None = None) -> AtomRef:
    """dataset AtomRef 생성."""
    return AtomRef(kind="dataset", name=name, params=dict(params or {}))


def loss_ref(name: str, params: dict | None = None) -> AtomRef:
    """loss AtomRef 생성."""
    return AtomRef(kind="loss", name=name, params=dict(params or {}))


def optim_ref(name: str, params: dict | None = None) -> AtomRef:
    """optimizer AtomRef 생성. params 에는 lr 등 hyperparam 만 (model.parameters() 제외)."""
    return AtomRef(kind="optim", name=name, params=dict(params or {}))


def sched_ref(name: str, params: dict | None = None) -> AtomRef:
    """scheduler AtomRef 생성. params 에는 T_max 등만 (optimizer 제외)."""
    return AtomRef(kind="sched", name=name, params=dict(params or {}))


def metric_ref(name: str, params: dict | None = None) -> AtomRef:
    """metric AtomRef 생성."""
    return AtomRef(kind="metric", name=name, params=dict(params or {}))


__version__ = "3.0.4"
__all__ = [
    "AtomRef",
    "AtomSpec",
    "Experiment",
    "ParamSpec",
    "ResolvedConfig",
    "RunContext",
    "Trainer",
    "agent",
    "compare_runs",
    "config",
    "dataset_ref",
    "datasets",
    "describe_run",
    "diff_recipes",
    "examples",
    "finalize_run",
    "input_dir",
    "inspect_project",
    "list_meta",
    "log",
    "loss",
    "loss_ref",
    "metric",
    "metric_ref",
    "model_ref",
    "models",
    "optim",
    "optim_ref",
    "output_dir",
    "recipe_meta",
    "register_dataset",
    "register_loss",
    "register_metric",
    "register_model",
    "register_optim",
    "register_sched",
    "resolve_project",
    "resolve_run_context",
    "save_all",
    "save_config_snapshot",
    "save_manifest",
    "save_metrics",
    "save_partial_run_record",
    "save_run_summary",
    "sched",
    "sched_ref",
    "seed_everything",
    "summarize_run",
    "testing",
    "validate_project",
]
