# PCQ Agent Rules

This project follows the CQ experiment contract.

## Runtime Contract

- Treat `cq.yaml` as the execution contract and source of truth.
- Resolve config through pcq / the CQ runtime contract; do not hard-code `output/`.
- Relative `output_dir` values are project-root relative.
- Write results under `pcq.output_dir()`.
- End runs with `pcq.save_all(...)` or equivalent standard artifacts.

## Standard Artifacts

Completed runs should produce:

- `config.json`
- `metrics.json`
- `run_summary.json`
- `manifest.json`
- `run_record.json`
- `validation_report.json`

`run_record.json` is the canonical completion record.

## Implementation Policy

- pcq is a contract runtime + agent CLI surface. There is no model catalog,
  recipe library, or framework adapter inside pcq.
- All ML code (datasets, models, losses, optimizers, schedulers, metrics,
  training loops) lives in your project's `train.py` or your own modules.
- Any ML framework is allowed when the script honors the CQ contract
  (read `cq.config()`, write `cq.save_all()`, emit `@key=value` metrics).

## Agent Workflow

Before edits or execution:

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --strictness 2 --json
```

After a serious run:

```bash
pcq run --path . --jsonl
pcq validate-run <output_dir> --strictness 3 --json
pcq describe-run <output_dir> --json
```

When comparing iterations:

```bash
pcq compare-runs <base_output_dir> <candidate_output_dir> --json
pcq lineage <candidate_output_dir> --json
```

When applying agent-authored experiment plans:

```bash
pcq apply-plan plan.json --path . --json
pcq apply-planset planset.json --path . --output-pattern 'runs/exp{i}' --json
```

## Attribution

Every `run_record.json` (and all six standard artifacts) can carry an
`attribution` object identifying **who ran the experiment**: the human
operator, the AI committer, or both. This complements the `agent` field
in `cq.yaml` (which names the *framework*) by providing authorship
identity at the run level.

### Why attribution?

- Enables audit trails in multi-agent pipelines.
- Distinguishes human-authored runs from agent-authored runs in dashboards.
- Allows post-hoc filtering: `jq 'select(.attribution.committer.kind=="agent")'`.

### Env var table

Set these before invoking `pcq run` (or `pcq.save_all()` inside your
script). All variables are optional; omit any you do not need.

| Variable | Description | Example value |
|---|---|---|
| `CQ_ATTRIBUTION_OPERATOR` | Human or org identifier who owns the run. **Must be pseudonym or UUID — never a real email address.** | `user-a3f8c1e2` |
| `CQ_ATTRIBUTION_AUTHOR_ID` | Identity of whoever authored the change (human or agent). | `claude-sonnet-4-5` |
| `CQ_ATTRIBUTION_AUTHOR_KIND` | Kind of author: `"human"` or `"agent"`. | `agent` |
| `CQ_ATTRIBUTION_COMMITTER_ID` | Identity of whoever committed / triggered the run. | `claude-sonnet-4-5` |
| `CQ_ATTRIBUTION_COMMITTER_KIND` | Kind of committer: `"human"` or `"agent"`. | `agent` |
| `CQ_ATTRIBUTION_SESSION_ID` | Opaque session identifier (e.g. Claude Code session UUID). | `sess-9b2d4f1a` |
| `CQ_ATTRIBUTION_PERSONA_AUTHOR` | Optional persona label for the author. | `research-agent-v2` |
| `CQ_ATTRIBUTION_PERSONA_COMMITTER` | Optional persona label for the committer. | `research-agent-v2` |

### Agent launcher auto-fill guidance

When an AI agent (Claude Code, Codex, custom launcher) invokes pcq,
set the committer fields to describe the AI and pass through the
operator from the user environment:

```bash
# 예: Claude Code가 pcq를 호출하기 전에 설정
export CQ_ATTRIBUTION_COMMITTER_KIND=agent
export CQ_ATTRIBUTION_COMMITTER_ID=claude-sonnet-4-6   # 실제 모델 ID
export CQ_ATTRIBUTION_AUTHOR_KIND=agent
export CQ_ATTRIBUTION_AUTHOR_ID=claude-sonnet-4-6
# 사용자 환경에서 operator를 그대로 전달 (없으면 생략)
# export CQ_ATTRIBUTION_OPERATOR="${CQ_ATTRIBUTION_OPERATOR}"
export CQ_ATTRIBUTION_SESSION_ID="${CLAUDE_SESSION_ID:-}"   # 세션 ID (있으면)

pcq run --path . --jsonl
```

```bash
# 예: Codex 또는 임의 agent launcher
export CQ_ATTRIBUTION_COMMITTER_KIND=agent
export CQ_ATTRIBUTION_COMMITTER_ID="<your-model-id>"   # 예: gpt-4o, gemini-2.5-pro
export CQ_ATTRIBUTION_AUTHOR_KIND=agent
export CQ_ATTRIBUTION_AUTHOR_ID="<your-model-id>"
# CQ_ATTRIBUTION_OPERATOR는 사용자 환경에서 주입 — 에이전트가 덮어쓰지 않음

pcq run --path . --json
```

### PII 권고 (Privacy Warning)

> **WARNING**: `CQ_ATTRIBUTION_OPERATOR` **must not contain a real
> email address, phone number, or any directly identifying personal
> information.**
>
> Use a pseudonym, internal user handle, or UUID instead.
>
> - Bad:  `CQ_ATTRIBUTION_OPERATOR=jane.doe@company.com`
> - Good: `CQ_ATTRIBUTION_OPERATOR=user-a3f8c1e2`
> - Good: `CQ_ATTRIBUTION_OPERATOR=researcher-42`
>
> `author_id` and `committer_id` for AI agents should be a model
> identifier (e.g. `claude-sonnet-4-6`), not a personal identifier.
> For human authors, use the same pseudonym/UUID convention as operator.
>
> `run_record.json` is logged, version-controlled, and potentially
> shared. PII in attribution fields cannot be redacted retroactively.

## Worker Spec

Every `run_record.json` (and all six standard artifacts) can carry a
`worker_spec` object capturing the **hardware and runtime environment** of the
machine that executed the run. This enables reproducibility auditing,
performance regression triage, and worker-class filtering across run sets.

`worker_spec` is built automatically at run time via psutil + torch introspection,
optionally overridden or extended by 13 `CQ_WORKER_*` env vars.

### Schema summary

```
worker_spec
├── cpu?             { model?, cores_physical?, cores_logical?, max_freq_mhz? }
├── memory?          { total_gb? }
├── accelerator      { kind, gpus[], visible_devices? }   # kind: mps|cuda|cpu
├── os?              { system, machine, release? }
├── container        { kind, image?, detector_hint? }     # kind: none|docker|k8s|other
└── source           "detected" | "declared" | "merged"
```

`source` indicates how the spec was produced:
- `detected` — fully auto-detected (psutil + torch)
- `declared` — taken entirely from `CQ_WORKER_*` env vars
- `merged`   — env vars override partial auto-detection

### Env var table

Set these before invoking `pcq run` (or inside your launcher script). All
variables are optional; omit any you do not need. When set, they take
precedence over auto-detected values.

| Variable | Maps to | Example value |
|---|---|---|
| `CQ_WORKER_CPU_MODEL` | `cpu.model` | `Intel Core i9-14900K` |
| `CQ_WORKER_CORES_PHYSICAL` | `cpu.cores_physical` | `24` |
| `CQ_WORKER_CORES_LOGICAL` | `cpu.cores_logical` | `32` |
| `CQ_WORKER_MAX_FREQ_MHZ` | `cpu.max_freq_mhz` | `5800` |
| `CQ_WORKER_MEMORY_TOTAL_GB` | `memory.total_gb` | `64.0` |
| `CQ_WORKER_ACCELERATOR_KIND` | `accelerator.kind` | `cuda` |
| `CQ_WORKER_GPU_MODEL_0` | `accelerator.gpus[0].model` | `NVIDIA RTX 5080` |
| `CQ_WORKER_GPU_VRAM_GB_0` | `accelerator.gpus[0].vram_gb` | `16.0` |
| `CQ_WORKER_GPU_CUDA_VERSION` | `accelerator.gpus[0].cuda_version` | `12.4` |
| `CQ_WORKER_OS_SYSTEM` | `os.system` | `Linux` |
| `CQ_WORKER_OS_MACHINE` | `os.machine` | `x86_64` |
| `CQ_WORKER_OS_RELEASE` | `os.release` | `6.8.0-51-generic` |
| `CQ_WORKER_CONTAINER_KIND` | `container.kind` | `docker` |

### cq launcher / Claude Code auto-fill example

```bash
# 예: CQ launcher 또는 Claude Code가 pcq를 호출하기 전에 설정
export CQ_WORKER_CPU_MODEL="$(python -c 'import cpuinfo; print(cpuinfo.get_cpu_info()[\"brand_raw\"])' 2>/dev/null || echo '')"
export CQ_WORKER_ACCELERATOR_KIND=cuda
export CQ_WORKER_GPU_MODEL_0="NVIDIA RTX 5080"
export CQ_WORKER_GPU_VRAM_GB_0=16.0
export CQ_WORKER_CONTAINER_KIND=docker   # 컨테이너 안에서 실행 중인 경우

pcq run --path . --jsonl
```

When psutil is installed (`uv add psutil` or `pip install psutil`), most CPU
and memory fields are auto-detected without any env vars.

> **WARNING — cgroups host-view limitation**: `memory.total_gb` reflects
> the **host machine's total RAM**, not the memory limit imposed by the
> container runtime. Inside a Docker or k8s container, cgroups memory limits
> are not automatically reflected in `worker_spec.memory.total_gb`.
>
> If you need the container's effective memory limit (e.g. for OOM
> threshold planning), use `CQ_WORKER_MEMORY_TOTAL_GB` to declare the
> actual cgroup limit as a declared override.

### Warning codes

`pcq validate-run` and `pcq validate` emit the following worker-spec warning
codes in `validation_report.json`. All are non-blocking (they lower the report
status to `warn` but do not fail validation).

| Code | Meaning |
|---|---|
| `WORKER_PSUTIL_MISSING` | psutil not installed; CPU/memory info not collected |
| `WORKER_PSUTIL_PARTIAL` | psutil installed but some fields unavailable (permission denied etc.) |
| `WORKER_TORCH_MISSING` | torch not installed; GPU info not collected |
| `WORKER_CGROUP_DENIED` | cgroup read denied; container memory limit not reflected in `total_gb` |
| `WORKER_CONTAINER_AMBIGUOUS` | multiple container detection hints conflict; `container.kind` may be wrong |
| `WORKER_DECLARED_PII_LIKE` | declared `worker_spec` free-text field contains hostname-shaped pattern |

### R14 PII warning

> **WARNING**: Free-text fields in `worker_spec` (such as `cpu.model`,
> `os.release`, or `container.image`) populated via `CQ_WORKER_*` env vars
> or declared overrides may inadvertently contain hostname-shaped strings
> (e.g. `my-laptop.corp.example.com`). These are flagged as
> `WORKER_DECLARED_PII_LIKE` at L3+.
>
> - If your CPU model string or container image tag embeds a hostname, use
>   an anonymised alias (e.g. `worker-node-42` instead of the real hostname).
> - Auto-detected values (psutil / torch) do not contain hostnames and are
>   not subject to this warning.

## Do Not

- Do not write artifacts to a hard-coded `output/` path.
- Do not emit metrics that are missing from `cq.yaml.metrics`.
- Do not treat process exit code alone as run completion.
- Do not continue an experiment loop after blocking validation failure.
- Do not look for `pcq.Trainer`, `pcq.Experiment`, `pcq.recipes`, or
  `pcq.examples.*` — these were removed in v4.0. Write a contract script.
