# Agent Operating Guide

## Purpose

This guide describes how a coding agent should create, modify, validate, and
interpret ML experiments with `pcq`.

The agent should treat `cq.yaml` as the execution contract and
`run_record.json` as the completion record.

## First Principles

- `pcq` is an experiment boundary, not a model catalog.
- `pcq` does not compete with PyTorch, HF Trainer, Lightning, sklearn, XGBoost,
  TabPFN, PyCaret, shell scripts, or custom project code.
- The contract is the adapter.
- Production experiment logic belongs in project-local files.
- Prefer JSON/JSONL commands over scraping terminal prose.
- Treat exit code as incomplete evidence until `validate-run` has been read.
- Never infer artifact paths from cwd when `cq.yaml` defines `output_dir`.

## Non-Negotiable Contract

Every agent-authored experiment must make these five things explicit:

1. how to run: top-level `cq.yaml.cmd`
2. what changed: project-local code or `cq.yaml.configs`
3. what to measure: `cq.yaml.metrics` plus monitor/mode in config
4. where to write: `pcq.output_dir()`, never a hard-coded path
5. how to finish: `pcq.save_all(...)` or equivalent `pcq.finalize_run(...)`

Minimum `cq.yaml`:

```yaml
name: my-experiment
cmd: uv run python train.py
configs:
  output_dir: output
  seed: 42
  strictness: 2
  monitor: eval_acc
  mode: max
metrics:
  - epoch
  - eval_acc
artifacts:
  - output/
inputs: {}
```

Minimum `train.py`:

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()
pcq.seed_everything(cfg.get("seed", 42))

# Train or evaluate with any framework.
score = 0.0
history = [{"epoch": 0, "eval_acc": score}]

pcq.log(**history[-1])
pcq.save_all(history=history, status="completed")
```

If a script does not meet this contract, fix the contract before improving the
model.

## Install

```bash
uv add pcq
```

The PyPI distribution, import, and CLI are all named `pcq`:

```python
import pcq
```

```bash
pcq --help
```

## Install Agent Runtime Assets

`pcq` can install canonical agent instructions and skill files into the current
project. This is explicit; package installation never modifies project files.

```bash
pcq agent install --target codex --path .
pcq agent install --target claude --path .
pcq agent install --target both --path . --dry-run --json
pcq agent status --target both --path . --json
```

Use `pcq agent status --json` as a read-only health check before assuming an
agent runtime can see pcq instructions.

## MCP Server (v4.1.0)

To let the agent runtime call pcq directly without subprocess parsing:

```bash
uv add 'pcq[mcp]'
pcq agent install --target claude --path . --mcp   # auto-wires .mcp.json
pcq mcp serve                                       # stdio (default)
pcq mcp serve --transport sse --port 8765          # HTTP SSE
```

After `--mcp`, the project's `.mcp.json` contains a `pcq` server entry pointing
at `pcq mcp serve`. Claude Code (and any MCP-aware client) auto-attaches and
exposes 14 `mcp__pcq__*` tools matching the 14 CLI subcommands. See
[MCP Integration](MCP_INTEGRATION.md) for the full tool list and architecture.

## Initial Triage

When given a project:

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --strictness 2 --json
```

Identify:

- selected `cq.yaml`
- project root
- command
- output directory
- declared metrics
- available inputs
- existing artifacts
- prior run records
- whether evidence is already valid

Read-side commands should not train, download data, or import heavy project
code by default.

## Choosing An Implementation Style

Default to a contract script.

Use project-local helpers only when they reduce real project complexity. Do not
add a framework adapter or core `pcq` feature for one experiment.

| Situation | Use | Edit Surface |
|---|---|---|
| HF Trainer, Lightning, TabPFN, PyCaret, sklearn, XGBoost, LightGBM | contract script | `train.py`, local modules, `cq.yaml.configs` |
| simple Torch baseline | contract script | `train.py`, `cq.yaml.configs` |
| custom training lifecycle | contract script or local helper | project files only |
| command-driven external tool | contract script wrapper | `train.py`, shell command, output parser |
| repeated project pattern | local helper module | project files only |

The agent should not ask whether `pcq` has built-in support for a framework.
If the framework can be called from project-local code and its result can be
converted to metrics/artifacts, it is usable.

## Contract Script Pattern

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()

# framework code

pcq.log(epoch=0, eval_score=score)
pcq.save_all(
    history=[{"epoch": 0, "eval_score": score}],
    status="completed",
    artifacts={"model": "model.pkl"},
)
```

Agent rules:

- put framework-specific setup in project-local code
- keep hyperparameters in `cq.yaml.configs` or runtime config
- save framework artifacts under `pcq.output_dir()`
- convert framework metrics into declared `pcq.log(...)` keys
- call `pcq.save_all(...)` for completed runs
- preserve structured failure evidence for failed runs when possible

## Running A Project

`pcq run` reads `cq.yaml.cmd`, writes runtime config into `.pcq/`, sets
`CQ_CONFIG_JSON`, and executes the command.

```bash
pcq run --path .
pcq run --path . --json
pcq run --path . --jsonl
pcq run --path . --events output/events.jsonl --json
```

Use:

- `--json` for one final parseable envelope
- `--jsonl` for live events
- `--events PATH --json` for final JSON stdout plus persisted live events

Do not parse human terminal output when JSON or JSONL is available.

## Post-Run Checklist

```bash
pcq validate-run output --strictness 3 --json
pcq describe-run output --json
```

Read:

- run status
- best metric
- monitor direction
- validation status
- artifact summary
- source snapshot
- input identity
- parent run lineage
- reproducibility evidence
- `decision_facts`
- structured failure evidence

## Comparison Loop

```bash
pcq compare-runs parent_output candidate_output --json
pcq lineage candidate_output --json
```

`compare-runs` is evidence, not policy. The agent decides whether to continue,
branch, rollback, rerun, or stop.

## Iteration

Use structured config plans when a bounded config change is enough:

```bash
pcq apply-plan experiment.plan.json --json
```

Use direct project-local code edits when the next experiment needs new research
logic. Keep those edits in the user project, not in `pcq` internals.

## Metadata APIs (agent-fillable)

`run_record.json` may carry three sibling metadata objects — `attribution`,
`worker_spec`, and `fingerprint`. Each has a Python API, environment-variable
overrides, and a `cq.yaml` declared path. Agents should fill them proactively:
the more complete the record, the higher the evidence quality for downstream
comparison and matchmaking.

### When to use each API

| API | Recommended action | Call site |
|---|---|---|
| `pcq.attribution(...)` | 모든 run에 추천 — `operator`(책임자) 선언이 1순위 | `train.py` 상단, `pcq.config()` 직후 |
| `pcq.worker_spec()` | 자동 감지됨, 명시 호출 불필요. cfg override 시에만 호출 | 필요한 경우 `train.py` 상단 |
| `pcq.fingerprint(...)` | data load 직후 한 줄 추가 — `modality` 명시 | `X_train, y_train = load_data()` 바로 아래 |

All three APIs are **additive**: calling them enriches the run record without
changing the training or metric contract. They are optional at the format
level, but agents should include them when context is available.

### attribution — operator 신고

```python
import pcq

cfg = pcq.config()

# operator: 법적·평판 책임을 지는 사람/조직 (에이전트 ID 불가)
# committer: 실제로 job을 빌드·제출한 주체 (AI agent인 경우 kind="agent")
pcq.attribution(
    operator="alice-uuid",          # 필수: 사람 또는 조직 식별자
    committer_kind="agent",         # "human" | "agent"
    committer_id="claude-opus-4-7", # agent 모델명 또는 사람 UUID
)
```

`author` (의도 기원)와 `committer` (실행 주체)는 Git의 author/committer 관례를
따릅니다. 사람이 직접 `pcq run`을 실행하면 세 필드 모두 같은 사람입니다.

환경 변수로도 선언 가능 (CI / 컨테이너 환경에 적합):

```bash
export CQ_ATTRIBUTION_OPERATOR="alice-uuid"
export CQ_ATTRIBUTION_COMMITTER_KIND="agent"
export CQ_ATTRIBUTION_COMMITTER_ID="claude-opus-4-7"
```

### worker_spec — 실행 환경 (자동 감지)

```python
# 자동 감지가 기본값 — 호출 생략 가능
# cfg override가 필요할 때만 명시 호출
pcq.worker_spec()
```

`pcq`는 `psutil`과 `torch`를 사용해 CPU/메모리/GPU/OS를 자동 감지합니다.
컨테이너 환경이나 cgroup 제한이 있는 경우 자동 감지값이 부정확할 수 있으며,
이때는 `CQ_WORKER_*` 환경 변수로 override하세요:

```bash
export CQ_WORKER_MEMORY_TOTAL_GB=32
export CQ_WORKER_GPU_MODEL_0="NVIDIA RTX 5080"
export CQ_WORKER_CONTAINER_KIND="docker"
```

`source` 필드가 자동으로 `"detected"`, `"declared"`, `"merged"` 중 하나로
기록되어 어떤 값이 자동 감지인지 명시 선언인지 audit trail이 남습니다.

### fingerprint — 데이터 특성 기록

```python
import pcq

X_train, y_train = load_data()
# data load 직후 한 줄 추가
pcq.fingerprint(X_train, y_train, modality="tabular", task_kind="classification")
```

`fingerprint`는 PII나 원시 값 없이 데이터셋의 형태(shape), modality, task 종류,
domain을 기록합니다. Column 이름, raw values, value-level 분포는 **절대** 방출되지
않습니다.

**modality 자동 판단 규칙** (에이전트용):

| 데이터 타입 | modality 값 |
|---|---|
| pandas DataFrame / numpy 2-D array | `"tabular"` |
| numpy array with shape `[N, H, W, C]` or `[N, H, W]` | `"image"` |
| `list[str]` 또는 `list[list[str]]` | `"text"` |
| pandas Series / numpy 1-D with datetime index | `"time_series"` |
| numpy array with shape `[N, T]` (T = sample count) | `"audio"` |
| dict with keys `"edge_index"` or `"adjacency"` | `"graph"` |
| 위에 해당하지 않는 경우 | `"other"` + `other.hint` 추가 |

### 환경 변수 표

세 prefix 그룹으로 구성됩니다. 해결 우선순위:
`CLI flags > CQ_*_* env vars > cq.yaml 선언 > 자동 감지 > NULL`

#### CQ_ATTRIBUTION_* (8개)

| 변수 | 채우는 필드 |
|---|---|
| `CQ_ATTRIBUTION_OPERATOR` | `attribution.operator` |
| `CQ_ATTRIBUTION_AUTHOR_ID` | `attribution.author.id` |
| `CQ_ATTRIBUTION_AUTHOR_KIND` | `attribution.author.kind` |
| `CQ_ATTRIBUTION_COMMITTER_ID` | `attribution.committer.id` |
| `CQ_ATTRIBUTION_COMMITTER_KIND` | `attribution.committer.kind` |
| `CQ_ATTRIBUTION_SESSION_ID` | `attribution.session_id` |
| `CQ_ATTRIBUTION_PERSONA_AUTHOR` | `attribution.author.persona_id` |
| `CQ_ATTRIBUTION_PERSONA_COMMITTER` | `attribution.committer.persona_id` |

#### CQ_WORKER_* (13개)

| 변수 | 채우는 필드 |
|---|---|
| `CQ_WORKER_CPU_MODEL` | `worker_spec.cpu.model` |
| `CQ_WORKER_CPU_CORES_PHYSICAL` | `worker_spec.cpu.cores_physical` |
| `CQ_WORKER_CPU_CORES_LOGICAL` | `worker_spec.cpu.cores_logical` |
| `CQ_WORKER_CPU_MAX_FREQ_MHZ` | `worker_spec.cpu.max_freq_mhz` |
| `CQ_WORKER_MEMORY_TOTAL_GB` | `worker_spec.memory.total_gb` |
| `CQ_WORKER_ACCELERATOR_KIND` | `worker_spec.accelerator.kind` |
| `CQ_WORKER_GPU_MODEL_0` | `worker_spec.accelerator.gpus[0].model` |
| `CQ_WORKER_GPU_VRAM_GB_0` | `worker_spec.accelerator.gpus[0].vram_gb` |
| `CQ_WORKER_GPU_CUDA_VERSION` | `worker_spec.accelerator.gpus[0].cuda_version` |
| `CQ_WORKER_OS_SYSTEM` | `worker_spec.os.system` |
| `CQ_WORKER_OS_MACHINE` | `worker_spec.os.machine` |
| `CQ_WORKER_OS_RELEASE` | `worker_spec.os.release` |
| `CQ_WORKER_CONTAINER_KIND` | `worker_spec.container.kind` |

#### CQ_FINGERPRINT_* (5개)

| 변수 | 채우는 필드 |
|---|---|
| `CQ_FINGERPRINT_MODALITY` | `fingerprint.modality` |
| `CQ_FINGERPRINT_TASK_KIND` | `fingerprint.task_kind` |
| `CQ_FINGERPRINT_N_SAMPLES` | `fingerprint.n_samples` |
| `CQ_FINGERPRINT_DOMAIN` | `fingerprint.domain` |
| `CQ_FINGERPRINT_SAMPLE_ROWS` | stratified sampling row count (default: 100 000) |

### 도메인 게이트 (R5) — medical / financial / regulated

`domain`이 `medical`, `financial`, `regulated` 중 하나인 경우, `pcq.fingerprint()`의
**자동 추출 경로가 비활성화**됩니다 (R5). 모든 통계 필드는 null로 남으며,
`FINGERPRINT_DOMAIN_GATE_SKIP` (severity L2) 경고가 `validation_report.json`에
기록됩니다.

```yaml
# cq.yaml — 규제 데이터 선언 경로
fingerprint:
  modality: tabular
  task_kind: classification
  n_samples: 50000
  domain: medical        # 자동 추출 비활성
  tabular:
    n_columns: 25
    type_counts: { numeric: 20, categorical: 5 }
    target_balance: 0.91
  source: declared
```

**R5b — 휴리스틱 도메인 스니퍼**:
`domain`이 `"general"` 또는 미설정인 경우, `pcq`는 통계 계산 전에 column 이름을
medical/financial 키워드 사전과 비교합니다:

- **Medical 키워드**: `diagnosis`, `icd`, `mrn`, `patient`, `clinical`, `ehr`,
  `dob`, `dod`, `lab_result`, `encounter`, `prescription`
- **Financial 키워드**: `ssn`, `account_no`, `routing`, `iban`, `credit_score`,
  `tax_id`, `brokerage`, `portfolio`, `loan_amount`, `transaction_id`

키워드 일치 시 `FINGERPRINT_DOMAIN_SUSPECTED_MEDICAL` 또는
`FINGERPRINT_DOMAIN_SUSPECTED_FINANCIAL` 경고 (L2)가 발생합니다.
스니퍼는 advisory only — 자동 추출을 차단하지 않으며, 일치된 column 이름은
어떤 출력 객체에도 나타나지 않습니다.

> **에이전트 지침**: column 이름에 `patient`, `diagnosis`, `ssn` 등이 보이면
> `cq.yaml`에 `fingerprint.domain: medical` (또는 `financial`) 을 명시하세요.
> R5 게이트가 활성화되고 auto-detection 대신 declared path를 사용합니다.

### PII layered policy 요약

| 레이어 | 규칙 | 적용 API |
|---|---|---|
| L1 — 형식 금지 (R10) | auto-detection 코드는 절대 column 이름, raw values, 분포, sample rows 방출 불가 | fingerprint |
| L2 — 도메인 게이트 (R5) | domain ∈ {medical, financial, regulated} → 자동 추출 비활성, declared only | fingerprint |
| L3 — 휴리스틱 스니퍼 (R5b) | domain = "general"일 때 column 이름 키워드 검사, advisory L2 경고 | fingerprint |
| L4 — 선언 경로 PII 경고 (R14) | declared/merged 자유 문자열 필드에서 hostname/email/SSN 패턴 감지 시 L3 경고 | fingerprint, worker_spec |
| worker R10 | auto-detection 코드는 hostname, IP, MAC, 사용자 로그인 이름 방출 불가 | worker_spec |
| attribution PII 권고 | operator/id에 실제 이메일·이름 대신 UUID 또는 pseudonym 권장 | attribution |

`pcq`는 어떤 필드도 자동으로 redact하거나 hash하지 않습니다.
Redaction은 소비 시스템(CQ Hub, CI)의 책임입니다.

## Inference Metric Pattern

inference 시점 표준 키를 사용하면 run 간 비교가 용이해진다. `pcq.log()`는
자유 키를 허용하지만, 아래 권장 키를 쓰면 cross-run 분석이 편리하다.

### 기본 패턴 (latency / memory)

```python
import pcq, time, psutil

# Inference loop
start = time.perf_counter()
output = model.generate(input, max_tokens=100)
latency_ms = (time.perf_counter() - start) * 1000

pcq.log(
    latency_p50_ms=latency_ms,
    tokens_per_sec=100 / latency_ms * 1000,
    memory_peak_mb=psutil.Process().memory_info().rss / 1024**2,
    batch_size=1,
)
```

### LLM 스트리밍 변형 (time_to_first_token_ms)

```python
import pcq, time

# 스트리밍 모드: 첫 토큰 시간과 전체 throughput을 별도 기록
t0 = time.perf_counter()
first_token_received = False
total_tokens = 0

for chunk in model.stream(input):
    if not first_token_received:
        ttft_ms = (time.perf_counter() - t0) * 1000
        first_token_received = True
    total_tokens += len(chunk.tokens)

total_ms = (time.perf_counter() - t0) * 1000

pcq.log(
    time_to_first_token_ms=ttft_ms,
    tokens_per_sec=total_tokens / total_ms * 1000,
    sequence_length=len(input.tokens),
    batch_size=1,
)
```

These keys are recommended for comparison-friendliness across runs, but
`pcq.log()` accepts free keys — use whatever is meaningful for your project.
No validation gate exists for these keys.

## Failure Categories

`failure.category` in `run_summary.json` is a regex-based heuristic on
`failure.message`. Free strings are allowed when no category matches. Agents
should use the category to decide whether to retry or abort.

| Category | Meaning | Retry / Abort hint |
|---|---|---|
| `config_error` | `cq.yaml` or runtime config invalid | Abort — fix `cq.yaml` before retry |
| `missing_dependency` | Required Python package not installed | Abort — `uv add <package>` then retry |
| `dataset_missing` | Dataset file or URI not found | Abort — check input URIs / paths then retry |
| `dataset_shape` | Tensor / array dimensions mismatch | Abort — fix tensor dims in code before retry |
| `label_contract` | Label range or dtype violation | Abort — check label range / dtype before retry |
| `loss_contract` | Loss function received incompatible inputs | Abort — check loss signature before retry |
| `metric_contract` | Undeclared metric emitted at strictness ≥ 3 | Abort — declare metric in `cq.yaml` then retry |
| `oom` | CUDA or host memory exhausted | Retry with smaller `batch_size`; abort if at minimum |
| `nan_loss` | Loss became NaN or Inf | Retry with lower `lr` or gradient clipping |
| `timeout` | Run exceeded configured time budget | Retry with larger `time_budget` |
| `distributed_write_race` | Concurrent writers collided on artifact path | Retry with fewer concurrent writers |
| `accuracy_below_threshold` | Validation metric below acceptance threshold | Retune (smaller `lr`, longer training); abort after budget exhausted |
| `user_interrupted` | Explicit user or operator signal (SIGTERM / KeyboardInterrupt) | Respect the interruption — do not auto-retry |
| `disk_full` | Output directory ran out of disk space | Abort — free space then retry; auto-retry unsafe |
| `model_load_failed` | Checkpoint or weights file could not be loaded | Retry after re-download or integrity check; abort on persistent hash mismatch |
| `unknown_exception` | Unclassified exception | Manual investigation required before retry |

## Forbidden Patterns

| Pattern | Why it is bad | Fix |
|---|---|---|
| `Path("output")`, `"output/model.pt"` | ignores custom `configs.output_dir` | `pcq.output_dir()` |
| metrics emitted but not declared | strict schema cannot validate the run | update `cq.yaml.metrics` |
| process exit code treated as success | artifacts may be missing | run `validate-run` |
| parsing prose logs | fragile agent behavior | use JSON/JSONL |
| one-off framework adapter in pcq core | competes with the framework | contract script |
| artifact writes outside output dir | worker collection misses evidence | write under `pcq.output_dir()` |
| failure exits before evidence | service sees only process failure | save structured failed run when possible |
| unseeded random split | result cannot be reproduced | use `cfg.seed` and record inputs |

## Failure Handling

When a framework fails after producing partial evidence:

```python
try:
    ...
except Exception as exc:
    pcq.save_all(
        history=history,
        status="failed",
        failure={"type": type(exc).__name__, "message": str(exc)},
    )
    raise
```

The raised exception should still produce a non-zero process exit. The saved
evidence lets `describe-run` explain the failure.

## Final Agent Rule

Operate the experiment boundary.

Do not compete with the training method. Use whatever method fits, then make the
run observable, verifiable, comparable, and repeatable through `pcq`.
