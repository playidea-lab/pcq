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

## Do Not

- Do not write artifacts to a hard-coded `output/` path.
- Do not emit metrics that are missing from `cq.yaml.metrics`.
- Do not treat process exit code alone as run completion.
- Do not continue an experiment loop after blocking validation failure.
- Do not look for `pcq.Trainer`, `pcq.Experiment`, `pcq.recipes`, or
  `pcq.examples.*` — these were removed in v4.0. Write a contract script.
