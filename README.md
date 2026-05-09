# pcq

> Apache-2.0 Python library for agent-operable, reproducible ML experiments.
> Bring any training code; `pcq` gives it a standard contract, artifacts, and
> machine-readable run evidence.

`pcq` turns a project with `cq.yaml` into a reproducible experiment unit. It
loads config, resolves output paths, emits declared metrics, writes standard
artifacts, and finalizes a `run_record.json` that an agent, CI job, notebook,
or CQ service worker can inspect later.

It is **not** a new training framework and it is **not** CQ-only. Use PyTorch,
Hugging Face Trainer, sklearn, TabPFN, PyCaret, XGBoost, or project-local custom
code. As long as the run honors the contract, `pcq` can collect the result.

**v3 single-name line** — PyPI package, Python import, CLI command, GitHub
repository, runtime workspace, and JSON contract names are all `pcq`. The CQ
runtime names `cq.yaml`, `CQ_CONFIG_JSON`, and `cq://` remain unchanged.

[SITE](https://playidea-lab.github.io/pcq/) | [INTRODUCTION](docs/INTRODUCTION.md) | [CHANGELOG](CHANGELOG.md) |
[SPEC](docs/SPEC.md) | [VISION](docs/VISION.md) |
[AGENT_OPERABILITY](docs/AGENT_OPERABILITY.md) | [ATOM_REGISTRY](docs/ATOM_REGISTRY.md) |
[RUN_RECORD](docs/RUN_RECORD.md) | [CQ_YAML_RUNTIME_CONTRACT](docs/CQ_YAML_RUNTIME_CONTRACT.md) |
[JSON_CONTRACTS](docs/JSON_CONTRACTS.md) |
[STRICTNESS](docs/STRICTNESS.md) |
[WORKER_EXECUTION_FLOW](docs/WORKER_EXECUTION_FLOW.md) |
[AGENT_OPERATING_GUIDE](docs/AGENT_OPERATING_GUIDE.md) |
[COMPLETION_ROADMAP](docs/PCQ_COMPLETION_ROADMAP.md) |
[CASE_STUDY: MNIST Dogfood](docs/case-studies/mnist-dogfood-2026-05-08.md) |
[CASE_STUDY: Tabular Dogfood](docs/case-studies/tabular-dogfood-2026-05-09.md)

```text
pcq = open-source agent-operable experiment contract
cq   = managed execution + orchestration + result loop
```

## Why pcq

- **One contract for many frameworks** — keep your preferred ML stack and add
  a thin evidence layer instead of adopting a new trainer.
- **Agent-readable by default** — CLI JSON, strictness gates, manifests,
  lineage, and run summaries are designed for coding agents and services.
- **Reproducible run boundary** — config, metrics, source, environment, inputs,
  artifacts, validation, and best/last results converge into `run_record.json`.
- **Local first, service ready** — the same project can run locally, in CI, in a
  notebook, or inside the managed CQ worker.

## Installation

> **PyPI distribution name: `pcq`**. The short `cq` name is already occupied
> on PyPI and also denotes the managed CQ service boundary, so the open-source
> contract library uses `pcq`. Python `import pcq` and the `pcq` CLI command
> are the public runtime surfaces.

```bash
uv add pcq
# Optional extras
uv add 'pcq[vision]'   # timm + torchvision
uv add 'pcq[dist]'     # accelerate (multi-GPU)
uv add 'pcq[nlp]'      # transformers
```

`pyproject.toml`:

```toml
[project]
dependencies = ["pcq"]
```

`pcq init-experiment --with-pyproject` generates a fresh `pyproject.toml`
with `pcq` dependencies for you automatically.

If you need to pin a specific git tag/branch (pre-release / private fork /
patch under review), use a git source:

```toml
[tool.uv.sources]
pcq = { git = "https://github.com/playidea-lab/pcq.git", tag = "v3.0.2" }
```

### Current Scope (v3.x)

- **lineage summary**: `pcq lineage` focuses on the head run's best value; use
  `pcq describe-run` on ancestors when an agent needs full per-run detail.
- **compare-runs signal**: if both runs select the same best epoch and value,
  `metric_direction` is `tied`; agents should also inspect config/source diffs.
- **validate --plan**: plan validation catches structural issues first; full
  label-contract checks run through project/recipe validation.
- **catalog scope**: contract examples live under
  `pcq.examples.{models,datasets,optim,sched}` plus helper examples
  (`pcq.loss`, `pcq.metric`). They are **reference examples** for contract
  verification + onboarding + smoke, not a production catalog. Real research
  atoms live in project-local `atoms/` (see [Project-Local
  Atoms](#확장--project-local-atoms-v112) below). `pcq.{models,datasets,optim,sched}`
  remain compatibility facades.

## Agent / Worker Docs

For agent and CQ service integration, read these in order:

1. [CQ YAML Runtime Contract](docs/CQ_YAML_RUNTIME_CONTRACT.md) — how
   `cq.yaml`, `CQ_CONFIG_JSON`, metrics, inputs, and `output_dir` resolve.
2. [Worker Execution Flow](docs/WORKER_EXECUTION_FLOW.md) — how a CQ worker
   executes a project and collects standard artifacts.
3. [Agent Operating Guide](docs/AGENT_OPERATING_GUIDE.md) — how a coding agent
   should choose script, Trainer, Experiment, or project-local atoms.
4. [JSON Contracts](docs/JSON_CONTRACTS.md) — stable machine-readable CLI
   output contracts for agents and services.
5. [Strictness Evidence Matrix](docs/STRICTNESS.md) — evidence required at
   strictness levels 0..4.
6. [CQ MCP Tool Spec](docs/CQ_MCP_SPEC.md) — service-facing tool surface for
   structured agent operation.
7. [Agent Acceptance Checklist](docs/AGENT_ACCEPTANCE_CHECKLIST.md) — release
   gates for agent-operable behavior.
8. [pcq Completion Roadmap](docs/PCQ_COMPLETION_ROADMAP.md) — remaining
   strictness, evidence, E2E, agent-runtime, MCP, and release-hardening work.

Reusable agent assets:

- [templates/AGENTS.pcq.md](templates/AGENTS.pcq.md)
- [skills/pcq/SKILL.md](skills/pcq/SKILL.md)

Install those assets into a project explicitly:

```bash
# Codex: AGENTS.md + .agents/skills/pcq/SKILL.md
pcq agent install --target codex --path .

# Claude Code: CLAUDE.md + .claude/skills/pcq/SKILL.md
pcq agent install --target claude --path .

# Preview without writing
pcq agent install --target both --path . --dry-run --json

# Inspect installed/missing/stale assets without writing
pcq agent status --target both --path . --json
```

`pcq` never modifies project agent files during package installation. Agent
runtime files are created only by `pcq agent install` or
`pcq init-experiment --agent codex|claude|both`.

## 3-Tier API

pcq 은 3 레이어 API 를 제공합니다. 동일한 contract 위에서 어느 레이어든
선택할 수 있고, 위 레이어에서 아래 레이어로 자연스럽게 내려갈 수 있습니다.

> **Note**: pcq 내부 atom 들 (`pcq.examples.models.mlp`,
> `pcq.examples.datasets.fake`, `pcq.examples.optim.adamw`,
> `pcq.examples.sched.cosine`, `pcq.loss.cross_entropy` 등) 은 **reference
> examples** — contract 검증 + 온보딩 + smoke baseline 용 예시입니다.
> production catalog 가 아닙니다. 실제 atom 은 project-local 작성
> ([Project-Local Atoms](#확장--project-local-atoms-v112) 참조).
> `pcq.{models,datasets,optim,sched}` 는 compatibility facade 로 남아 있지만,
> 문서와 신규 코드는 `pcq.examples.*` 또는 project-local atoms 를 사용합니다.

### 저레벨 — cq.yaml 컨트랙트 API

```python
import pcq

cfg = pcq.config()                 # CQ_CONFIG_JSON 파싱
pcq.seed_everything(cfg["seed"])
out = pcq.output_dir()             # output/ 디렉토리 보장
pcq.log(epoch=1, train_loss=0.42)  # stdout @key=value
```

계약 스크립트는 Trainer/Experiment 없이도 표준 artifact 를 만들 수 있습니다.
HF Trainer, TabPFN, PyCaret, sklearn, XGBoost, custom code 모두 이 경로로
연결합니다. framework adapter 는 필요하지 않습니다.
실행 가능한 최소 예제는
[examples/contract_numpy.py](examples/contract_numpy.py) 입니다.

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()
pcq.seed_everything(cfg.get("seed", 42))

# Use any ML library here.
score = 0.74

pcq.log(epoch=0, eval_score=score)
pcq.save_all(
    history=[{"epoch": 0, "eval_score": score}],
    status="completed",
    artifacts={"model": "model.pkl"},
)
```

### 중레벨 — Lightning 스타일 Experiment

```python
import pcq
from pcq import Experiment

class MyExp(Experiment):
    def build_dataset(self, split): return pcq.examples.datasets.fake(num_samples=128)
    def build_model(self): return pcq.examples.models.small_cnn(3, 10)
    def build_loss(self): return pcq.loss.cross_entropy()
    def build_optimizer(self, params): return pcq.examples.optim.adamw(params, lr=1e-3)

    def training_step(self, batch):
        # 반환: (loss_tensor, metrics_dict)
        # loss_tensor 는 backward 용 grad 살아있는 tensor, metrics 는 logging 전용.
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        return loss, {"acc": pcq.metric.accuracy(logits, y).item()}

    def eval_step(self, batch):
        # 반환: metrics_dict (backward 없음)
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        return {"loss": loss.item(), "acc": pcq.metric.accuracy(logits, y).item()}

MyExp().fit()  # accelerate 자동 감지, 5종 아티팩트 저장
```

### 고레벨 — preset + atom override

```python
import pcq

# Case A: preset 한 줄
pcq.Trainer(preset="vision/cifar10_smallcnn_baseline").fit()

# Case B: preset + atom override
pcq.Trainer(
    preset="vision/cifar10_smallcnn_baseline",
    sched_factory=lambda o: pcq.examples.sched.cosine(o, T_max=20, warmup=1000),
).fit()

# Case C: preset 없이 atom 이름으로
pcq.Trainer(task="classification", dataset="fake", model="mlp").fit()
```

## cq.yaml 통합

cq worker 가 cq.yaml 의 `cmd` 를 실행하면서 inline configs 를 `CQ_CONFIG_JSON`
환경변수에 JSON 파일 경로로 전달합니다. 라이브러리는 이 컨트랙트만 알고
있고, cq 의 Hub/Drive 를 직접 호출하지 않습니다.

```yaml
# examples/cq.yaml
name: cq-python-smoke
cmd: uv run python examples/train.py

configs:
  output_dir: output
  epochs: 2
  batch_size: 32
  lr: 0.001
  seed: 42

metrics:
  - epoch
  - train_loss
  - train_acc
  - eval_loss
  - eval_acc

artifacts:
  - output/
```

```python
# examples/train.py — 9 줄
import pcq
cfg = pcq.config()
pcq.seed_everything(cfg["seed"])
pcq.Trainer(task="classification", dataset="fake", model="mlp", cfg=cfg).fit()
```

## Structured cq.yaml (v1.15+)

기존 list-style metrics 는 영구 호환. agent 가 풍부한 메타데이터를 원할 때
dict-style `metrics:` 와 신규 `inputs:` 섹션을 사용한다. pcq 은 cq URI 를
**opaque string** 으로 취급한다 — parse / fetch 는 CQ service 의 책임.

### `inputs:` 섹션 — dataset / input identity

```yaml
inputs:
  dataset:
    name: dental
    version: v12
    uri: cq://datasets/dental/v12
    split: train-val-2026-05-01
    sha256: "abc123..."
```

### `metrics:` dict-style — 각 metric 의 schema

```yaml
# legacy (영구 호환)
metrics:
  - eval_iou
  - eval_loss

# v1.15+ structured (agent-friendly)
metrics:
  eval_iou:
    mode: max
    split: val
    aggregation: macro
    sample_count: 1240
  eval_loss:
    mode: min
    split: val
```

`pcq validate` 가 dict-style 일 때 추가 gate:

- `metric_schema_complete` / `metric_schema_mode` / `metric_schema_mode_value` —
  각 metric 의 `mode` 필드가 `min` / `max` 인지
- `monitor_in_metric_schema` / `monitor_mode_consistency` —
  `configs.monitor` 가 schema key 인지 + `configs.mode` 와 일치하는지
- `inputs_declared` / `input_identity` —
  inputs 섹션의 각 항목에 `name` 필드가 있는지

`pcq inspect --json` 출력의 `cq_yaml` 에 `metrics_schema` 와 `inputs` 가
포함된다 (둘 다 미사용이면 출력에서 생략).

## RunRecord — Single Source of Truth (v1.16+)

`run_record.json` 은 한 실험 run 의 모든 evidence 를 담은 SSOT. CQ service 가
비교 / lineage / 재현성에 사용한다.

```json
{
  "schema_version": 1,
  "run": {"id": "run_20260506_120000_abc", "status": "completed", "...": "..."},
  "execution": {"cmd": "...", "config_path": "cq.yaml"},
  "source": {"git_sha": "...", "dirty": false},
  "environment": {
    "python": "3.12.10",
    "platform": "Darwin-arm64",
    "lockfile": "uv.lock",
    "lockfile_sha256": "..."
  },
  "inputs": {"dataset": {"name": "dental", "uri": "cq://...", "...": "..."}},
  "metrics": {
    "declared": [{"name": "eval_iou", "mode": "max", "split": "val"}],
    "history_path": "metrics.json"
  },
  "artifacts": [{"path": "model.pt", "sha256": "...", "size_bytes": 0}],
  "summary": {"target_metric": "eval_iou", "best": {"epoch": 12}},
  "agent": {"plan_id": "exp-001", "recipe": "vision/seg/voc_unet"},
  "validation": {"status": "pass", "report_path": "validation_report.json"}
}
```

### 자동 생성 경로

- **Trainer / Experiment**: `fit()` 끝나면 자동 작성.
- **Contract script**: `pcq.save_all(history=[...])` 가 자동 finalize 포함
  (opt-out: `finalize=False`).
- **명시 호출**: `pcq.finalize_run(history=[...])` 또는 CLI `pcq finalize`.
- **Post-run validation**: `pcq validate-run [OUTPUT_DIR]`.

### Source / Environment Snapshot

- `source.git_sha` — `git rev-parse HEAD` (자동).
- `source.dirty` — `git status --porcelain` 비어있지 않으면 true.
- `source.patch_sha256`, `source.changed_files` — `cfg["record_patch"]: true` 시 opt-in.
- `environment.python` — `sys.version`.
- `environment.platform` — `system + machine`.
- `environment.lockfile_sha256` — `uv.lock` / `poetry.lock` / `Pipfile.lock` 자동 감지.

### CQ URI 는 opaque

`inputs.dataset.uri: cq://datasets/...` 같은 CQ-aware URI 는 pcq 이 **parse / fetch
하지 않는다**. CQ service 가 dataset fetch 를 담당. pcq 은 URI 를 string 그대로
record.

### CLI

```bash
pcq finalize [OUTPUT_DIR]            # run_record.json + validation_report.json 작성
pcq validate-run [OUTPUT_DIR]         # post-run gates
                                       #   - manifest sha256 round-trip
                                       #   - metrics well-formed
                                       #   - run_summary best/last consistent
                                       #   - run_record completeness
```

`Experiment.fit()` / `pcq.save_all()` 사용 시 자동 — 외부 trigger 로 재생성 시에만
명시 호출.

## ResolvedConfig — Single Source of Truth (v2.2+)

`cq.yaml` + `CQ_CONFIG_JSON` env 를 한 번 해석한 **단일 view**. inspect,
validate, finalize_run, RunRecord 가 모두 같은 resolver 를 거쳐 cq.yaml 해석
일관성을 보장한다.

```python
import pcq

rc = pcq.resolve_project(path=".")
print(rc.cfg)              # configs section (cq.yaml + env merged, env wins)
print(rc.declared_metrics) # always list[str], list/dict-style 둘 다 정규화
print(rc.metrics_schema)   # dict-style metrics (mode/split/...) if present
print(rc.inputs)           # opaque dict (cq URI 는 그대로 보존)
print(rc.output_dir)       # absolute Path, project_root-rooted, mkdir-safe
```

CLI:

```bash
pcq resolve --json
```

cwd 의존 제거: `resolve_project()` 인자 없이 호출 시 cwd 기준 ancestor walk-up
(8 levels max, `.git` / `pyproject.toml` 만나면 stop). agent 가 어디서 실행하든
같은 cq.yaml 해석.

명시적 우선순위: `CQ_CONFIG_JSON` 의 cfg 가 `cq.yaml.configs` 위에 merge —
env 가 동일 키를 override.

## Run Comparison & Description (v1.17+)

agent 가 RunRecord 만으로 다음 실험을 결정할 수 있는 read-side 도구. side-effect
없음.

### `pcq describe-run` — compact run summary

```bash
pcq describe-run output/ --json
```

```json
{
  "schema_version": 1,
  "run_id": "run_20260506_120000_abc",
  "status": "completed",
  "target_metric": "eval_iou",
  "mode": "max",
  "best": {"epoch": 4, "value": 0.78, "checkpoint": "best.ckpt"},
  "best_value": 0.78,
  "best_epoch": 4,
  "last": {"epoch": 4, "value": 0.78, "checkpoint": "last.ckpt"},
  "last_value": 0.78,
  "last_epoch": 4,
  "epochs_completed": 5,
  "python": "3.12.10",
  "platform": "Darwin-arm64",
  "metrics_declared": [{"name": "eval_iou", "mode": "max"}],
  "artifacts": [{"path": "model.pt", "kind": "model"}],
  "artifacts_summary": {"model": 1, "checkpoint": 2, "metrics": 1, "summary": 1},
  "plan_id": "plan-b-001",
  "recipe": "improved",
  "validation_status": "pass",
  "reproducibility_evidence": {
    "source": {"git_sha": "...", "dirty": false},
    "environment": {"python": "3.12.10", "lockfile": "uv.lock"},
    "config": {"seed": 42, "strictness": 3}
  },
  "decision_facts": {
    "run_completed": true,
    "validation_passed": true,
    "has_best": true,
    "artifact_count": 4
  }
}
```

`output/` 에 `run_record.json` 이 없으면 `status: "no_record"` 로 응답 (rc=0).
`describe-run` 은 다음 실험을 추천하지 않습니다. agent 가 판단할 수 있는
status / metric / artifact / validation / reproducibility facts 만 제공합니다.

### `pcq compare-runs A B` — diff two runs

```bash
pcq compare-runs run_a/ run_b/ --json
```

```json
{
  "schema_version": 1,
  "a_run_id": "run_20260505_160645_a54fdc",
  "b_run_id": "run_20260505_160645_ccf92a",
  "target_metric": "eval_iou",
  "a_target_metric": "eval_iou",
  "b_target_metric": "eval_iou",
  "mode": "max",
  "best": {"a": 0.66, "b": 0.78, "delta": 0.12, "direction": "improved"},
  "metric_delta": 0.12,
  "metric_direction": "improved",
  "last": {"a": 0.63, "b": 0.76, "delta": 0.13, "direction": "improved"},
  "config_changes": [
    {"key": "_overrides_keys", "a": ["lr"], "b": ["batch_size", "lr"]},
    {"key": "recipe", "a": "baseline", "b": "improved"}
  ],
  "validation": {"a": "pass", "b": "pass", "same": true},
  "artifacts": {"a_count": 4, "b_count": 5},
  "source": {"same_git_sha": false, "same_cq_yaml_sha256": false},
  "decision_facts": {
    "comparable": true,
    "same_target_metric": true,
    "best_improved": true,
    "candidate_validated": true,
    "config_changed": true,
    "source_changed": true
  },
  "a_status": "completed",
  "b_status": "completed"
}
```

`metric_direction` 은 `metrics.declared[].mode` (없으면 `min`) 를 보고 결정:

| mode  | delta < 0  | delta == 0 | delta > 0  |
|-------|------------|------------|------------|
| `max` | regressed  | tied       | improved   |
| `min` | improved   | tied       | regressed  |

`compare-runs` 도 추천하지 않습니다. agent 가 다음 실험을 고를 수 있도록 metric,
trajectory, config/input/source/artifact/validation/failure 차이와
`decision_facts` 만 제공합니다.

A/B 인자는 `run_record.json` 파일 직접 또는 output 디렉토리 둘 다 수용.

### Failure Classifier

`run_summary.json.failure.category` 가 자동 분류되어 agent 가 다음 행동 결정:

| category               | 처방 예시                          |
|------------------------|-----------------------------------|
| `oom`                  | batch_size 줄이기, gradient checkpointing |
| `nan_loss`             | lr 줄이기 / gradient clip 적용     |
| `dataset_shape`        | resize / collate_fn 점검           |
| `dataset_missing`      | input identity / 데이터셋 pull 확인 |
| `missing_dependency`   | `uv add ...`                       |
| `label_contract`       | ignore_index / label range 확인    |
| `loss_contract`        | loss signature / shape contract    |
| `metric_contract`      | declared metric vs emit mismatch   |
| `timeout`              | timeout_sec 늘리기 / 데이터 캐시   |
| `distributed_write_race` | manifest 쓰기 race — rank 분리   |
| `config_error`         | CQ_CONFIG_JSON / cfg key 확인      |
| `unknown_exception`    | logs 직접 확인                     |

명시 카테고리(non-`unknown_exception`)가 이미 있으면 보존, 없거나 unknown 이면
`failure.message` 를 휴리스틱으로 분류.

## Lineage Tracking (v1.18+)

agent 가 multi-run 진화를 추적하기 위한 RunRecord parent chain. RunRecord 의
`run.parent_run_id` + `run.parent_run_path` 두 필드로 연결한다 — semantic id
(검증) + path string (resolution).

### Plan-driven (agent path)

ExperimentPlan 에 `parent_run_id` + `parent_run_path` 추가:

```json
{
  "schema_version": 1,
  "id": "exp-005",
  "intent": "Tighten regularization",
  "base": {"preset": "vision/seg/voc_unet"},
  "parent_run_id": "run_20260506_120000_abc",
  "parent_run_path": "../run_004/output",
  "changes": [
    {"op": "set_config", "key": "weight_decay", "value": 5e-4}
  ]
}
```

`pcq apply-plan` 이 자동으로 `cq.yaml.configs._parent_run_id` /
`_parent_run_path` 를 주입. 다음 fit() / save_all() 종료 시 finalize_run 이
`RunRecord.run.parent_run_id` / `parent_run_path` 에 기록한다.

### Script path (explicit)

```python
pcq.save_all(
    history=...,
    parent_run_id="run_20260506_120000_abc",
    parent_run_path="../run_004/output",
)
```

또는 cq.yaml configs 에 직접:

```yaml
configs:
  _parent_run_id: run_20260506_120000_abc
  _parent_run_path: ../run_004/output
```

### Lineage 조회

```bash
pcq lineage output/ --json
```

```json
{
  "schema_version": 1,
  "chain": [
    {"run_id": "current",     "depth": 0, "best_value": 0.78, "target_metric": "eval_iou"},
    {"run_id": "parent",      "depth": 1, "best_value": 0.74, "target_metric": "eval_iou"},
    {"run_id": "grandparent", "depth": 2, "best_value": 0.71, "target_metric": "eval_iou"}
  ],
  "truncated": false,
  "notes": []
}
```

`compare-runs` 도 lineage 인지 — 결과에 `a_is_ancestor_of_b` /
`b_is_ancestor_of_a` 필드가 포함된다.

### 안정성 보장

- **순환 감지**: 같은 `run_id` 가 다시 등장하면 chain 종료 + note 기록.
- **Missing parent**: `parent_run_path` 가 가리키는 곳에 `run_record.json` 이
  없으면 placeholder node 추가 + note. 빈 chain 반환하지 않음.
- **max_depth**: default 100. 초과 시 `truncated: true`.
- **상대 경로**: child 의 output_dir 기준으로 해석되어 working tree 이동에
  안전.

### CQ URI 호환

`parent_run_path` 가 `cq://...` URI 여도 pcq 은 **opaque string** 으로 보존만
한다. pcq 의 `lineage()` 는 해당 노드를 follow 하지 않고 placeholder 추가
(note 에 "remote URI not followed"). URI resolve 책임은 CQ service.

## Atom vs Contract Script — When To Use Which (v1.13)

pcq 은 두 가지 실험 작성 경로를 제공합니다. 어느 쪽이든 cq.yaml 위에서 동일하게
실행됩니다 — agent 는 두 경로를 모두 지원합니다.

### Atom + RecipeSpec (PyTorch path)

실험 컴포넌트가 select / replace / validate / reuse 되어야 할 때.

```python
import pcq
pcq.Trainer(
    preset="vision/seg/voc_unet",
    sched=pcq.sched_ref("cosine", {"T_max": 80, "warmup": 1000}),
).fit()
```

`apply-plan` 의 `set_atom` 으로 atom 단위 swap 가능. validate 가 ignore_index /
in_channels / extras 일관성 검사. project-local atom 등록 가능.

### Contract Script (any framework)

framework 가 자체 training flow 를 갖고 있을 때 (HF Trainer, sklearn, XGBoost,
TabPFN, PyCaret, custom code).

```python
import pcq
from sklearn.ensemble import RandomForestClassifier

cfg = pcq.config()
out = pcq.output_dir()

model = RandomForestClassifier(n_estimators=cfg["n_estimators"])
model.fit(X_tr, y_tr)
acc = model.score(X_te, y_te)

pcq.log(epoch=0, eval_acc=acc)
pcq.save_all(
    history=[{"epoch": 0, "eval_acc": acc}],
    artifacts={"model": "model.pkl"},
)
```

agent 는 `set_config` 로만 hyperparameter 변경 (`set_atom` 거부). atom 시스템은
경유하지 않음 — framework 자체가 모델/loss/optim 을 책임.

### Decision tree

```
컴포넌트 swap 이 실험 핵심?
├─ YES → Atom + RecipeSpec (PyTorch)
└─ NO, framework 자체로 training flow
   └─ Contract Script (any library)
```

`pcq init-experiment --style script|trainer|experiment` 로 시작.
[examples/contract_numpy.py](examples/contract_numpy.py) 는 core dependency 만으로
항상 실행되는 framework-neutral reference 이고,
[examples/contract_sklearn.py](examples/contract_sklearn.py) 는 sklearn-iris
optional reference 입니다.

## 확장 — Project-Local Atoms (v1.12+)

> **포지셔닝 (v2.7)**: pcq 의 built-in atom 들
> (`pcq.examples.{models,datasets,optim,sched}`, `pcq.loss`, `pcq.metric`) 은
> **reference examples** — contract 검증 + 온보딩 + smoke baseline 용 예시
> 구현입니다. **production catalog 가 아닙니다.** 실제 연구 atom 은
> project-local `atoms/` 디렉토리에 등록. `pcq.{models,datasets,optim,sched}`
> 는 `pcq.examples.*` 를 다시 내보내는 v2 compatibility facade 입니다.
> AtomSpec 의 `role="reference_example"` 필드와 `pcq atoms list` 인간 출력의
> `[reference example]` 태그로도 구분됩니다.

실제 실험은 project-local atom 으로 작성:

```bash
# 1. atom skeleton 생성
pcq atoms scaffold model dental_unet
# → atoms/models.py 생성, @pcq.register_model + meta 채워둠

# 2. implementation 채우기 (수동 또는 agent)
# atoms/models.py 편집

# 3. 검증
pcq atoms validate-local         # metadata + contract 완전성
pcq atoms smoke model dental_unet --load-project .   # forward 1-step 검증

# 4. recipe 에서 사용
pcq.model_ref("dental_unet", {"num_classes": 4})
```

`init-experiment` 가 `cq_atoms.py` + `atoms/__init__.py` 를 함께 생성합니다.
`train.py` 는 자동으로 `cq_atoms` 를 import → 추가 작업 없이 atom 등록.

`pcq inspect` 는 기본적으로 project atom 을 import 하지 않는 read-only
모드입니다. project atom 까지 registry view 에 포함하려면 명시적으로
`--load-project-atoms` 를 사용합니다.

```bash
pcq inspect . --load-project-atoms --json
pcq atoms list --source project --json   # project atom 만
pcq atoms list --source builtin --json   # builtin 만
pcq atoms list --load-project /path/to/project --json
```

### Atom Sources

| Source     | 위치                              | 추가 정책                                | role (v2.4)         |
|------------|-----------------------------------|------------------------------------------|---------------------|
| `builtin`  | `pcq` 내부                        | 추가 X — contract example only            | `reference_example` |
| `project`  | `cq_atoms.py`, `atoms/*.py`        | 사용자/agent 자유 작성 — **primary path** | `user`              |
| `generated`| agent 실험 루프 산출               | project atom 과 동일 메커니즘             | `user`              |
| `external` | 외부 패키지 등록                   | (예약)                                   | `user`              |

`role` 은 v2.4 추가 메타데이터. `pcq atoms list --json` 결과의 각 entry 에
포함되며, builtin atom 이 production catalog 가 아니라 contract example 임을
명시합니다.

## 확장 (atoms / recipes)

pcq 은 두 레이어로 확장합니다.

### Atom — Python 함수

pcq 이 제공하는 계약 예제(`pcq.examples.models`, `pcq.examples.datasets`,
`pcq.examples.optim`, `pcq.examples.sched`)와 helper atom (`pcq.loss`,
`pcq.metric`)을 직접 호출하거나 사용자 코드에 자기 함수를 작성. 직접 사용 시
등록 불필요:

```python
import pcq
my_model = pcq.examples.models.small_cnn(3, 10)
dataset = pcq.examples.datasets.fake(num_samples=32)
optim = pcq.examples.optim.adamw(my_model.parameters(), lr=1e-3)
acc = pcq.metric.accuracy(logits, y)
```

`pcq.{models,datasets,optim,sched}` 는 v2 호환용 facade 입니다. 신규 코드와
문서는 `pcq.examples.*` 또는 project-local atom 을 우선 사용합니다.

### String-name 등록 (`Trainer(model="name")` 용)

string lookup 을 쓰려면 registry 에 등록:

```python
import pcq

# 함수 형태
pcq.register_model("vit_b16", lambda: my_vit_b16(num_classes=10))
pcq.register_dataset("my_data", lambda split: MyDataset(train=(split == "train")))
pcq.register_metric("f1", lambda l, y: my_f1_impl(l, y))

# 데코레이터 형태
@pcq.register_model("custom_resnet")
def factory():
    return torchvision.models.resnet50(num_classes=100)

pcq.Trainer(model="vit_b16", dataset="my_data").fit()

pcq.Trainer.list_models()    # 등록된 모델
pcq.Trainer.list_datasets()  # 등록된 데이터셋
pcq.Trainer.list_metrics()   # 등록된 메트릭
pcq.Trainer.list_presets()   # 등록된 recipe
```

내장 atom 들은 모듈 import 시 자동 등록됩니다.

### Atom Registry — Metadata-Aware (v1.8+)

atom 등록 시 `meta=...` 로 ParamSpec / contracts / extras 를 함께 선언하면
agent 가 검증·검색할 수 있습니다:

```python
import pcq

pcq.register_loss(
    "boundary_dice",
    factory=lambda smooth=1.0: BoundaryDiceLoss(smooth=smooth),
    meta={
        "tasks": ["segmentation"],
        "params": {"smooth": {"type": "float", "default": 1.0, "min": 0.0}},
        "input_contract": {"logits": ["B","C","H","W"], "target": ["B","H","W"]},
        "label_contract": {"target_dtype": "int64", "ignore_index_param": "ignore_index"},
    },
)
```

기존 단순 형태 (`pcq.register_X("name", factory)`) 도 그대로 — `metadata_status="inferred"`
로 wrap. v1.8 에서 cross_entropy / unet / fake_seg / voc_seg / iou 가 explicit.

### AtomRef — Agent-Operable References (v1.8+)

agent 는 직렬화 가능한 ref 로 atom 을 사용할 수 있습니다:

```python
ref = pcq.loss_ref("cross_entropy", {"ignore_index": -1})
# ref.to_dict() → {"kind": "loss", "name": "cross_entropy", "params": {"ignore_index": -1}}
```

CLI 로 카탈로그·검증:

```bash
pcq atoms list --json
pcq atoms list --kind loss --json
pcq atoms show loss cross_entropy --json
pcq atoms validate-ref ref.json --json
```

### RecipeSpec — Metadata-First Recipes (v1.8+)

`vision/seg/*` recipe 는 v1.8 부터 `RecipeSpec` 기반:

```python
from pcq.agent.schema import RecipeSpec
import pcq

SPEC = RecipeSpec(
    name="vision/seg/my_recipe",
    task="segmentation",
    metrics=["epoch", "train_loss", "eval_iou"],
    monitor_candidates=[{"name": "eval_iou", "mode": "max"}],
    requires_extras=["vision"],
    smoke_safe=False,
    atoms={
        "model":         pcq.model_ref("unet", {"num_classes": 21}),
        "dataset_train": pcq.dataset_ref("voc_seg", {"image_set": "train"}),
        "dataset_eval":  pcq.dataset_ref("voc_seg", {"image_set": "val"}),
        "loss":          pcq.loss_ref("cross_entropy", {"ignore_index": -1}),
        "optim":         pcq.optim_ref("adamw", {"lr": 1e-3}),
        "sched":         pcq.sched_ref("cosine", {"T_max": 50, "warmup": 500}),
    },
    defaults={"epochs": 50, "batch_size": 16},
)

def my_recipe() -> dict:
    return SPEC.build()
```

기존 dict recipe (`vision/fake_smoke`, `vision/cifar10_*`, `nlp/*` 등) 은 그대로 호환.

### Recipe — atom 조합의 named alias

`src/pcq/recipes/<group>/<name>.py` 에 함수만 작성하면 `Trainer.list_presets()` 이
pkgutil 로 자동 발견. 데코레이터 등록 불필요:

```python
# src/pcq/recipes/vision/my_recipe.py
from pcq import datasets, loss, optim
from pcq.examples import models as example_models

def my_recipe() -> dict:
    return {
        "model": example_models.small_cnn(3, 10),
        "dataset_train": lambda _split: datasets.cifar10("data", train=True, download=True),
        "dataset_eval":  lambda _split: datasets.cifar10("data", train=False, download=True),
        "loss": loss.cross_entropy(),
        "optim_factory": lambda p: optim.adamw(p, lr=1e-3),
        "epochs": 30, "batch_size": 128,
    }
```

## Contract Behavior (v1.6)

agent-operability 를 위한 5가지 contract gap 을 v1.6 에서 닫았습니다.
docs/AGENT_OPERABILITY.md §"Phase A: Fix Current Contract Gaps" 참조.

### Declared Metrics 자동 로드

`pcq.log()` 는 다음 우선순위로 declared metrics 를 검증합니다:

1. `CQ_DECLARED_METRICS` 환경변수 (콤마 구분).
2. `CQ_CONFIG_JSON` 이 가리키는 JSON 파일의 `_metrics_declared` 키 (자동).

이전엔 사용자가 직접 env 를 설정하거나 cfg 인자를 줘야 했지만, v1.6 부터
`pcq.log()` 가 `CQ_CONFIG_JSON` 을 자동으로 1회 읽어 캐시합니다. 이미
`cq.yaml.metrics` 에 메트릭을 선언했다면 추가 설정 불필요.

### `cross_entropy(ignore_index=...)`

```python
pcq.loss.cross_entropy(ignore_index=-1)             # segmentation: void/-1 무시
pcq.loss.cross_entropy(weight=torch.tensor([0.1, 0.5, 1.0]))  # 클래스 가중
```

VOC `voc_seg` 가 void(255) 를 -1 로 변환하므로 `voc_unet` recipe 는
`ignore_index=-1` 로 학습합니다.

### `voc_seg(image_size=...)`

`torchvision.VOCSegmentation` 은 가변 크기 PIL 이미지를 반환하므로 default
DataLoader collate 가 stack 못 합니다. v1.6 부터 `image_size=256` (default) 로
image 와 mask 를 동기 리사이즈합니다 (image=bilinear, mask=nearest).

### Best Checkpoint Monitor — 사전 검증

`fit()` 시작 시 monitor key 가 declared metrics 안에 없으면 stderr 에 즉시
경고합니다. agent/사용자가 학습 시작 전에 best.ckpt 가 안 만들어질 수 있다는
사실을 인지할 수 있습니다.

### Multi-process Safety (accelerate)

`accelerate` 활성화 시 모든 IO (stdout `@key=value` 메트릭, checkpoint,
최종 artifact) 는 main process 에서만 수행합니다. multi-GPU 환경에서 동일
파일에 race write 발생 안 함. epoch 종료 후 `wait_for_everyone()` 으로
process 동기화.

## Recipe 카탈로그 (v1.4)

| Recipe | dataset × model | dep |
|--------|-----------------|-----|
| `vision/fake_smoke` | fake × mlp | torch only |
| `vision/mnist_mlp` | MNIST × mlp | + torchvision |
| `vision/cifar10_smallcnn_baseline` | CIFAR-10 × small_cnn | + torchvision |
| `vision/cifar10_resnet18` | CIFAR-10 × ResNet-18 | + torchvision |
| `vision/seg/fake_seg_smoke` | fake_seg × UNet | torch only |
| `vision/seg/voc_unet` | Pascal VOC × UNet | + torchvision |
| `nlp/fake_text_classifier` | fake_text × Embed+Mean+Linear | torch only |

`pcq.Trainer.list_presets()` 로 전체 목록 확인. `pcq.Trainer.print_recipe(name)`
으로 atom 조합 검사 가능. Recipe 모듈 경로는 임의 깊이 nested group 을 지원합니다
(`vision/seg/<name>` 처럼).

## 메트릭 schema 경고

`cq.yaml.metrics` 에 선언되지 않은 key 를 `pcq.log()` 로 출력하면 첫 1회
stderr 경고와 종료 시 누적 요약을 출력합니다. CI 에서는 `pcq.log(strict=True)`
로 즉시 실패하게 만들 수 있습니다.

## 분산 학습

`pcq[dist]` 로 `accelerate` 를 설치하면 `Experiment.fit()` 이 자동으로
multi-GPU/DDP/FSDP 경로로 분기합니다.

**Device 우선순위**: `cfg["device"]` (명시) > CUDA > MPS (Apple Silicon) > CPU.

`cfg["device"]` 를 명시하면 accelerate 분기를 우회합니다 (사용자가 명시한
device 를 accelerate 가 덮어쓰지 않도록). multi-GPU 를 원하면 device 를
명시하지 않고 `pcq[dist]` 만 설치하세요.

```yaml
configs:
  device: cpu   # accelerate 우회 + 강제 CPU
```

## Resume

```yaml
configs:
  resume_from: output/last.ckpt   # 명시적 path
  # 또는
  resume: true                     # 자동: output_dir/last.ckpt 발견 시 resume, 없으면 fresh
```

`resume_from` 이 명시적 우선. `resume: true` 만 있으면 `output_dir` 에서
`last.ckpt` 를 찾아 자동 resume (없으면 silent fresh start). `resume_from`
명시 시 그 path 가 없으면 여전히 `FileNotFoundError` 가 발생합니다 (auto fallback X).

## Best Checkpoint Monitor

기본 `best.ckpt` 는 `eval_loss` 최소 epoch 을 저장합니다. cfg 로 다른 메트릭/방향 지정:

```yaml
configs:
  monitor: eval_acc   # history entry 키 (epoch 별 metrics)
  mode: max           # min | max
```

monitor key 가 metrics 에 없으면 stderr 1회 경고 후 `best.ckpt` 저장을
건너뜁니다 (`last.ckpt` 는 그대로 매 epoch 저장).

## AMP + Gradient Accumulation

```yaml
configs:
  amp: true            # autocast + GradScaler (cuda+fp16) 또는 autocast 만 (bf16/cpu)
  amp_dtype: fp16      # fp16 | bf16
  grad_accum: 4        # effective batch_size = batch_size * grad_accum
```

- fp16 + cuda: `torch.amp.GradScaler` 활성화 (overflow 방지)
- bf16 또는 cpu/mps: autocast 만 사용 (GradScaler 불필요)
- `accelerate` 활성화 시 pcq 의 AMP 경로는 우회 (accelerate 가 처리)
- grad_accum 은 마지막 batch 또는 `accum_steps` 마다 `optimizer.step()` 실행

## Early Stopping

monitor 인프라를 그대로 재사용. 개선 없는 epoch 이 patience 초과 시 중단:

```yaml
configs:
  monitor: eval_loss
  mode: min
  early_stop_patience: 5     # 5 epoch 무개선 → 중단
  early_stop_min_delta: 0.001
```

- `early_stop_patience: 0` (default) → 비활성화
- `min_delta` 만큼 개선되어야 'improved' 로 카운트
- `no_improve_count` 는 ckpt 에 저장 → resume 시 누적 카운트 복원
- 중단 시 `metrics.json` 에 `early_stopped_at_epoch` 기록

## Metric Aggregation

기본은 batch 단위 단순 평균. variable batch size 환경 (마지막 drop_last=False)
에서 last-batch bias 가 신경 쓰이면 weighted_mean 사용:

```yaml
configs:
  metrics_aggregation: weighted_mean   # batch_size 가중 평균 (default: mean)
```

정확한 sample-weighted aggregation 이 필요하면 `pcq.metric.stateful` 의 클래스
사용 (v1.3 은 사용자가 직접 update/compute, v1.4 에서 Experiment 자동 통합 예정):

```python
acc = pcq.metric.stateful.Accuracy()
for batch in loader:
    x, y = batch
    acc.update(model(x), y)
print(acc.compute())   # 전체 sample 기준 정확도
```

## Artifacts

`fit()` 종료 시 `output_dir` 에 다음 파일들이 생성됩니다:

| 파일 | 설명 |
|---|---|
| `model.pt` | 최종 `model.state_dict()` |
| `config.json` | 정규화된 cfg + git SHA + pcq version + `_recipe`/`_overrides` |
| `metrics.json` | `{"history": [...]}` epoch 별 history |
| `run_summary.json` | best/last/target/provenance 요약 |
| `last.ckpt` | 매 epoch 저장 (resume 용) |
| `best.ckpt` | monitor 기준 best epoch (default: `eval_loss` min) |
| `manifest.json` | output 파일 인덱스 — schema v2 (sha256 + size_bytes + created_at) |

### Manifest schema v2 (v1.14+)

v1.14 부터 `manifest.json` 은 단순 file list 가 아니라 **artifact evidence**
를 기록합니다. 각 entry 는 sha256, size_bytes, created_at 을 포함하므로
CQ worker / agent 가 artifact 무결성을 round-trip 검증할 수 있습니다.

```json
{
  "schema_version": 2,
  "files": [
    {
      "path": "model.pt",
      "kind": "weights",
      "sha256": "8bb2ec07...",
      "size_bytes": 18423920,
      "created_at": "2026-05-05T10:58:00Z"
    }
  ]
}
```

대형 weight 가 많아 sha256 비용이 부담될 때는 cfg 에서 opt-out 가능
(schema v1 fallback — path/kind 만):

```yaml
configs:
  manifest_checksums: false
```

`pcq validate` 의 post-run gate `manifest_evidence` 가 manifest entries
가 실제 file 을 가리키고 sha256 이 일치하는지 검증합니다. v1 legacy manifest
는 file 존재 검증만 수행합니다.

자세한 표준은 [docs/RUN_RECORD.md](docs/RUN_RECORD.md)를 참고하세요.

`config.json` 의 provenance 필드 (v1.4):

| 필드 | 의미 |
|---|---|
| `_git_sha` | 학습 시점 HEAD commit SHA |
| `_pcq_version` | 사용된 pcq 버전 |
| `_recipe` | base preset 이름 (없으면 atom-only 모드) |
| `_overrides` | 사용자가 override 한 atom 키 정렬 목록 |

## Recipe Acceptance Framework (v1.4)

모든 등록된 recipe 는 SPEC §"Recipe Acceptance Criteria" 7항목을 자동 검증
합니다. 새 recipe 를 catalog 에 추가하면 `tests/test_acceptance.py` 의 parametrize
가 자동으로 검증 항목을 추가합니다.

```python
from pcq.testing import recipe_smoke, list_failures

# 단일 recipe 검증
report = recipe_smoke("vision/cifar10_smallcnn_baseline")
print(report)
# Recipe: vision/cifar10_smallcnn_baseline
# Status: PASS
#   [PASS] import: recipe module importable without training
#   [PASS] inspect: recipe() returned dict (12 keys)
#   [PASS] smoke_path: using 2 override(s): ['dataset_eval', 'dataset_train']
#   [PASS] fit_smoke: one epoch completed
#   [PASS] declared_metrics: all 5 emitted
#   [PASS] artifacts: all 5 present
#   [PASS] resume: resume from last.ckpt succeeded

# 모든 recipe 일괄 검증 — 실패한 것만
fails = list_failures()
```

Recipe 작성자는 metadata 키로 acceptance 동작을 제어합니다:

```python
def my_recipe() -> dict:
    return {
        # 기존 atom 키
        "model": ..., "dataset_train": ..., "loss": ..., "optim_factory": ...,
        # v1.4 옵션 metadata
        "metrics": ["epoch", "train_loss", "eval_acc"],
        "requires_extras": ["vision"],   # pcq[vision] for torchvision
        "smoke_safe": False,             # 외부 dataset/network 필요
        "smoke_overrides": {             # acceptance 가 substitute 할 atom
            "dataset_train": lambda _split: pcq.datasets.fake(...),
            "dataset_eval":  lambda _split: pcq.datasets.fake(...),
        },
    }
```

- `smoke_safe=True` → recipe atoms 그대로 1-epoch 학습
- `smoke_safe=False` + `smoke_overrides` → override 로 atom 치환 후 학습
- `smoke_safe=False` + overrides 없음 → acceptance fail
- `metrics` 비어있으면 declared metrics 검증 skip
- `requires_extras` 의 패키지 미설치 시 해당 recipe 는 skip (fail X)

## Agent-Native Helpers (v1.4)

VISION §"Agent-Native Future" 시작점. agent 가 recipe 를 선택·diff·검증하기
위한 inspection API. 모두 학습 부작용 없음:

```python
import pcq

# Recipe metadata — atom 조합 + 선언된 metrics + extras 요구사항
pcq.recipe_meta("vision/seg/fake_seg_smoke")
# {'name': 'vision/seg/fake_seg_smoke', 'task': 'segmentation',
#  'declared_metrics': ['epoch', 'train_loss', 'train_iou', ...],
#  'requires_extras': [], 'atoms': {'model': '_UNet (instance)', ...}, ...}

# Recipe diff — 두 baseline 의 atom/설정 비교
pcq.diff_recipes("vision/cifar10_smallcnn_baseline", "vision/cifar10_resnet18")

# Catalog 둘러보기
metas = pcq.agent.list_meta()
for m in metas:
    print(m["name"], "→", m["task"], "extras:", m["requires_extras"])

# Dry run — 조립 plan 만 보여주고 학습 X
plan = pcq.Trainer(
    preset="vision/cifar10_resnet18",
    sched_factory=lambda o: pcq.sched.cosine(o, T_max=10),
).dry_run()
print(plan["preset"], plan["overrides"], plan["expected_artifacts"])
```

## CLI (Agent JSON Interface, v1.7+)

pcq 은 agent 가 Python import 없이 사용할 수 있는 JSON CLI 를 제공합니다.

```bash
pcq resolve . --json                       # ResolvedConfig JSON
pcq resolve --cq-yaml ./cq.yaml --json     # explicit cq.yaml
pcq inspect . --json                       # ProjectInspection JSON (read-only)
pcq inspect . --load-project-atoms --json  # opt-in project atom import
pcq recipe-meta vision/fake_smoke --json   # RecipeMeta JSON
pcq dry-run . --json                       # 조립된 plan (학습 X)
pcq validate . --strictness 2 --json       # ValidationReport (fail → exit 1)
pcq validate-run output --strictness 3 --json
pcq summarize-run output/ --json           # RunSummary JSON
pcq run --path . --json                    # execute cq.yaml.cmd; pure JSON envelope
pcq init-experiment --style script --output ./tabular-exp --json
pcq agent install --target codex --path ./tabular-exp --json
pcq agent status --target codex --path ./tabular-exp --json
```

학습 후 `output/run_summary.json` 이 자동 생성되며 `summarize-run` 이 그대로
읽습니다. 파일이 없으면 `metrics.json`+`config.json` 에서 합성합니다.

모든 JSON 출력은 `schema_version` 필드를 포함합니다. 핵심 surface 는
[JSON Contracts](docs/JSON_CONTRACTS.md) 와
`pcq.agent.json_contracts.JSON_CONTRACTS` 에서 최소 안정 필드를 고정합니다.
검증 강도는 [Strictness Evidence Matrix](docs/STRICTNESS.md) 와
`pcq.agent.strictness.STRICTNESS_EVIDENCE_MATRIX` 에서 레벨별 필수 evidence 를
고정합니다.
`pcq run --json` 은
stdout 에 **envelope JSON 만** 출력하고, child process stdout/stderr 는
`.pcq/run_stdout.log` / `.pcq/run_stderr.log` 로 캡처한 뒤 JSON 의
`stdout_path`, `stderr_path`, `stdout_tail`, `stderr_tail` 로 요약합니다.
v3.x 동안 public JSON contract 는 additive 변경을 원칙으로 합니다. exit code 는
0 (성공) / 1 (실패·검증 fail) / 2 (argparse error) 입니다.

| Command | 무엇을 하나 | exit |
|---|---|---|
| `resolve [--cq-yaml PATH]` | cq.yaml + env 를 단일 read-only ResolvedConfig 로 해석 | 0 |
| `inspect [--load-project-atoms]` | cq.yaml + entrypoint + recipe 카탈로그 + outputs 상태를 한 번에 | 1 if path missing/import error |
| `recipe-meta` | 단일 recipe 의 atom 요약 + declared metrics | 1 if recipe unknown |
| `dry-run` | entrypoint 의 preset 으로 Trainer 조립 plan | 1 if no preset |
| `validate [--plan PLAN.json] [--strictness 0..4]` | static + recipe contract + strictness evidence validation | 1 if blocking fail |
| `summarize-run` | 완료된 output 의 best/last/provenance 요약 | 1 if status=failed |
| `run` | `cq.yaml.cmd` 실행 + `CQ_CONFIG_JSON` 자동 wiring; `--json` 은 pure JSON envelope | cmd exit code |
| `init-experiment` | `--style trainer|experiment|script` 로 cq.yaml + train.py scaffold (v1.13+) | 1 if invalid style/preset |
| `agent install` | Codex/Claude instruction + skill runtime assets 설치 | 1 if invalid |
| `agent status` | Codex/Claude instruction + skill runtime assets 상태 검사 (read-only) | 1 if invalid |
| `apply-plan` | ExperimentPlan JSON 을 cq.yaml configs 에 안전하게 적용 (v1.10+) | 1 if rejected |
| `atoms {list,show,scaffold,validate-local,smoke,validate-ref}` | atom registry 검사 / project atom scaffold (v1.8+, v1.12+) | 1 if invalid |
| `finalize [--project-root PATH] [--status completed|failed|partial]` | run_record.json + validation_report.json 작성 (v1.16+) | 1 if invalid output |
| `validate-run [--strictness 0..4]` | post-run gates (manifest evidence + metrics + summary + strictness evidence) (v1.16+) | 1 if any gate fails |
| `describe-run` | compact RunRecord summary (v1.17+) | 1 if no run_record |
| `compare-runs A B` | 두 RunRecord 의 metric / config / atom diff (v1.17+) | 1 if invalid record |
| `lineage [OUTPUT_DIR] [--max-depth N]` | parent_run_id chain 따라가며 ancestry 출력 (v1.18+) | 1 if no run_record |

## Phase D — Agent Mutation (v1.10+)

agent 는 ExperimentPlan JSON 으로 실험을 **구조적으로 수정**한다.

```bash
# 1. 새 프로젝트 scaffold
pcq init-experiment --style trainer --preset vision/fake_smoke \
                     --output ./my-experiment \
                     --name dental-baseline

# 2. ExperimentPlan 작성 (LLM 또는 사람)
cat > plan.json <<'EOF'
{
  "schema_version": 1,
  "id": "exp-001",
  "intent": "longer training",
  "base": {"preset": "vision/fake_smoke"},
  "target": {"metric": "eval_acc", "mode": "max"},
  "changes": [
    {"op": "set_config", "key": "epochs", "value": 80},
    {"op": "set_atom", "atom": "loss", "name": "cross_entropy",
     "params": {"ignore_index": -100}}
  ]
}
EOF

# 3. 검증 (apply 전)
pcq validate ./my-experiment --plan plan.json --json

# 4. 적용 (cq.yaml 만 수정, idempotent, provenance 자동 저장)
pcq apply-plan plan.json --path ./my-experiment --json

# 5. 학습
cd ./my-experiment && uv run python train.py
```

`apply-plan` 은 **cq.yaml 의 configs 만** 수정한다. `train.py` 와 `recipes/local.py`
는 사용자 영역 (수정하지 않음). provenance 는 `.pcq/plans/<plan_id>.json` 에 자동
저장 (plan + applied_at + operations).

**지원 ChangeOp**:
- `set_config` — `configs.<key> = <value>` (v1.10+)
- `set_atom` — `configs._overrides_data.<atom> = AtomRef.to_dict()`. `Trainer.from_cfg(cfg)` 가 학습 시점에 deserialize. `merge: true` 면 기존 params 와 병합 (v1.11+)
- `set_dataset_transform` — split-aware sugar for `set_atom` merge (v1.11+)

**v1.11 추가: set_atom merge + set_dataset_transform**

기존 atom 의 params 일부만 변경 (전체 ref 재명시 X):

```json
{"op": "set_atom", "atom": "dataset_train", "name": "voc_seg",
 "params": {"image_size": 384}, "merge": true}
```

dataset transform 전용 sugar:

```json
{"op": "set_dataset_transform", "split": "train", "params": {"image_size": 384}}
```

→ 내부적으로 `set_atom` (merge=true) on `dataset_train`. dataset name 은 base
recipe `SPEC.atoms` 또는 기존 override 에서 자동 상속.

**YAML Comment Preservation (v1.11+)**

기본 cq.yaml writer 는 minimal — comment 를 보존하지 않는다 (rewrite). comment
및 인용/인덴트 형태를 살리려면:

```bash
uv add 'pcq[yaml]'   # ruamel.yaml>=0.17 설치
```

설치 시 자동 감지 — read/write 가 round-trip 형태로 comment + 인용 + 인덴트 보존.

**v1.12 (예정)**: `set_smoke_override`, `custom_code_patch` (gated).

`Trainer.from_cfg(cfg)` 가 `cfg["preset"]` + `cfg["_overrides_data"]` 를 자동
인식해 trainer 를 조립한다 (init-experiment 가 만든 train.py 의 1줄).

## CI / Smoke

GitHub Actions (`.github/workflows/ci.yml`)는 매 push/PR에서 3 stage 실행:

| Stage | 명령 | 목적 |
|---|---|---|
| `lint` | `uv run ruff check src/ tests/` | 코드 스타일 |
| `test` | `uv run pytest tests/` | 단위 + 통합 + acceptance |
| `smoke` | `bash scripts/release-smoke.sh` | 실제 cq.yaml 시뮬 + 아티팩트 검증 |

수동 검증:
```bash
bash scripts/release-smoke.sh
# 6단계: lint → pytest → recipe acceptance → CQ_CONFIG_JSON subprocess →
#        artifacts/stdout/manifest/provenance 검증
```

`scripts/release-smoke.sh`는 release 태그 전 로컬에서도 실행 권장.

GitHub Pages (`.github/workflows/pages.yml`)는 `main` push 때 `site/`를
<https://playidea-lab.github.io/pcq/> 로 배포합니다.

## Status

pcq 은 **v3 single-name stable line** 입니다. 안정화된 surface:

- Contract artifacts (run 당 6 개 표준 파일 — config.json / metrics.json /
  manifest.json (v2) / run_summary.json / run_record.json / validation_report.json)
- 두 가지 패턴 — Atom + Recipe (PyTorch) / Contract Script (any framework)
- 13 개 CLI subcommand (agent operability)
- RunRecord SSOT + lineage tracking
- 814 tests passing, 4 skipped
- release smoke script passing

v3.x 는 `pcq` 단일 이름 위에서 API stability, fresh-user install, service
integration, and agent-operable evidence quality 에 집중합니다. pcq 자체는 작은
surface 를 유지합니다.

## Roadmap (v3.x)

- Fresh-user PyPI E2E and public quickstart hardening
- Contract-script examples for Hugging Face Trainer / TabPFN / PyCaret / sklearn
- Project-local atom validation and smoke checks 강화
- Generated atom provenance and review workflow
- CQ service integration contract 문서화 (post-finalize webhook 등)

## License

Apache License 2.0. See [LICENSE](LICENSE).
