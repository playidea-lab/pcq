# pcq Completion Roadmap

## Purpose

This document defines what remains before `pcq` can be treated as a complete
agent-operable ML experiment contract library.

The goal is not to add more built-in models, losses, datasets, or framework
adapters. The goal is to make arbitrary project-local ML code reliable for CQ
service agents, standalone coding agents, CI jobs, and local workflows to
author, execute, collect, validate, compare, and reproduce. `pcq` is CQ-
compatible, not CQ-only; CQ service is one managed consumer of the contract.

## Current Baseline

As of v2.7, `pcq` already has the minimum usable loop:

```text
cq.yaml
  -> CQ_CONFIG_JSON
  -> user training code
  -> pcq.log(...)
  -> pcq.output_dir()
  -> pcq.save_all() / pcq.finalize_run()
  -> metrics.json / manifest.json / run_summary.json / run_record.json
  -> pcq validate-run / pcq describe-run
```

The CLI surface also covers the core service-facing operations:

```text
pcq resolve
pcq inspect
pcq validate
pcq init-experiment
pcq atoms scaffold
pcq finalize
pcq validate-run
pcq describe-run
```

The remaining work is therefore not "make it run"; it is "make the machine
contracts stable, enforce the right evidence at the right level, and make those
guarantees durable enough for CQ service automation."

## Completion Definition

`pcq` is complete when all of the following are true:

1. An agent can inspect a project from `cq.yaml` and know how it will run.
2. An agent can generate script / Trainer / Experiment style code without
   relying on hidden conventions.
3. Pre-run validation catches missing contract requirements before execution.
4. Post-run validation can decide whether a run is complete, partial, or failed.
5. A `RunRecord` contains enough evidence to reproduce or audit the run.
6. CQ service tools can call `pcq` through stable JSON CLI/API contracts.
7. Built-in atoms remain contract examples; project-local atoms are the primary
   extension mechanism for real research code.

## Evidence Model

`pcq` should treat a run as evidence-backed only when these categories are
available or explicitly marked unavailable:

| Category | Evidence | Source |
|---|---|---|
| Runtime contract | `cq.yaml`, resolved config, command, output dir | `cq.yaml`, `CQ_CONFIG_JSON`, resolver |
| Metrics | declared metrics, emitted metric history, monitor/mode | `cq.yaml.metrics`, stdout, `metrics.json` |
| Artifacts | manifest entries, size, sha256, artifact kinds | `manifest.json` |
| Result summary | status, best/last metric, failure summary | `run_summary.json` |
| Completion record | run identity, source, env, inputs, validation | `run_record.json` |
| Source identity | git sha, dirty flag, changed files / patch hash | git workspace |
| Environment | Python, platform, package lock hash, torch/cuda/device | runtime inspection |
| Inputs | dataset uri/path, version, split, optional sha256 | `cq.yaml.inputs`, CQ worker |
| Lineage | parent run id/path, plan id, agent intent | plan/run metadata |

Missing evidence should not be silently ignored. It should become a validation
check with a status, severity, and `suggested_fix`.

## Strictness Levels

`cfg.strictness` and `pcq validate --strictness` should define how much
evidence is required.

| Level | Name | Required evidence | Intended use |
|---|---|---|---|
| 0 | Parse | `cq.yaml` parseable, `cmd` discoverable | editor feedback, very early scaffolds |
| 1 | Static | level 0 + entrypoint contract calls, declared metrics/artifacts | pre-run agent authoring |
| 2 | Standard | level 1 + recipe/atom compatibility, post-run manifest if present | default local/dev validation |
| 3 | Reproducible | level 2 + git sha/dirty, seed, lockfile hash, resolved inputs, RunRecord completeness | CI and serious experiment records |
| 4 | Service Grade | level 3 + input identity/hash or CQ URI, hardware/device evidence, lineage/plan evidence, strict metric schema | CQ managed runs and publishable comparisons |

### Level 0: Parse

Required:

- project root exists
- `cq.yaml` exists or explicit `cq_yaml_path` exists
- YAML is parseable
- `cmd` is present
- `configs.output_dir` resolves without creating directories

Typical failure fixes:

- create `cq.yaml`
- fix YAML syntax
- add top-level `cmd`

### Level 1: Static

Required in addition to level 0:

- entrypoint file exists
- script imports/calls the CQ contract helpers where relevant:
  - `pcq.config()`
  - `pcq.output_dir()`
  - `pcq.log(...)`
  - `pcq.save_all(...)` or explicit standard artifact writes
- declared metrics exist in `cq.yaml.metrics`
- declared artifact globs exist in `cq.yaml.artifacts`

Typical failure fixes:

- call `pcq.config()` instead of hard-coded config
- write artifacts under `pcq.output_dir()`
- declare every metric emitted through `pcq.log(...)`

### Level 2: Standard

Required in addition to level 1:

- recipe imports if a recipe/preset is used
- `RecipeSpec` atom refs validate against the registry
- label contracts are compatible, for example `ignore_index`
- model/dataset shape contracts are compatible when metadata exists
- required extras are available or reported
- if output artifacts already exist:
  - `manifest.json` parses
  - manifest entries point to existing files
  - manifest sha256 matches when present
  - metrics and summary are consistent

This is the current default level and should remain cheap enough for frequent
agent use.

### Level 3: Reproducible

Required in addition to level 2:

- `run_record.json` exists after execution
- `run_record.run.name` and `execution.cmd` are filled from `cq.yaml`
- source section includes git sha and dirty status
- dirty worktree has changed files and optionally a patch hash
- environment includes:
  - Python version
  - platform
  - `pcq` version
  - torch version when torch is imported
  - lockfile hash when a lockfile exists
- resolved config includes seed or explicitly records no seed
- inputs are recorded from `cq.yaml.inputs` or explicitly empty

Level 3 is the first level where a run can be called "reproducible enough for
internal experiment comparison."

### Level 4: Service Grade

Required in addition to level 3:

- every input has one of:
  - CQ URI
  - local path with sha256/manifest
  - explicit `opaque: true` reason
- metric schema includes monitor/mode for decision metrics
- hardware/device evidence is recorded:
  - CPU/GPU
  - GPU model/count when available
  - distributed world size when applicable
- lineage is present for derived runs:
  - parent run id/path
  - plan id or agent intent
- validation report itself is persisted and referenced from `RunRecord`

Level 4 is what CQ service should require before treating a run as suitable for
automated comparison, dashboard surfacing, or agent feedback loops.

## Validation Output Requirements

Every validation check should follow this shape:

```json
{
  "id": "lockfile_evidence",
  "status": "fail",
  "severity": "blocking",
  "detail": "pyproject.toml exists but no uv.lock was found",
  "evidence": {
    "strictness": 3,
    "expected": "uv.lock"
  },
  "suggested_fix": "run `uv lock` and commit uv.lock"
}
```

Rules:

- Required evidence missing at the selected strictness level should be `fail`.
- Optional evidence missing below that level should be `warn` or `skip`.
- Checks must be deterministic and JSON-stable.
- Agent-facing failures must include `suggested_fix` whenever possible.

## PR Sequence

Implementation should move in small reviewable PRs. The first four PRs define
the completion spine:

| PR | Theme | Main result |
|---|---|---|
| PR1 | JSON Contract Freeze | `run`, `describe-run`, `compare-runs`, and validation reports have machine-readable minimum JSON contracts and regression tests |
| PR2 | Strictness Evidence Matrix | `validate` / `validate-run` enforce level-specific evidence gates and persist strictness in reports |
| PR3 | Framework-Neutral Contract Examples | Torch/sklearn/other script-style examples prove pcq is a contract runtime, not a framework adapter |
| PR4 | Agent Runtime Installation Surface | pcq installs and reports Codex/Claude instructions and skills in project discovery paths with frozen JSON contracts |
| PR5 | CQ Service MCP Integration | CQ service wraps pcq JSON CLI/API with MCP tools |
| PR6 | Release Hardening | package/release gates prove the public contract before tagging |

PR1 is the first dependency because services and agents must know which JSON
fields are safe to parse before strictness, framework examples, or runtime
installation can be automated. PR2 defines what counts as pass, warn, skip, or
fail at each evidence level. PR3 proves that pcq works without framework
adapters. PR4 makes the authoring rules discoverable by real agent runtimes.
PR5 belongs mostly in the CQ service repository.

## Implementation Phases

### Phase 1: JSON Contract Freeze

Tasks:

- Define machine-readable JSON contracts for core CLI outputs.
- Keep contracts stdlib-only and dependency-free.
- Freeze required fields for:
  - `pcq run --json`
  - `pcq describe-run --json`
  - `pcq compare-runs --json`
  - `pcq validate --json`
  - `pcq validate-run --json`
- Ensure error envelopes still include `schema_version` and stable status.
- Add regression tests that validate real outputs against the contract registry.
- Document additive-only schema policy.

Acceptance:

- `pcq.agent.get_json_contracts()` returns JSON-serializable contracts.
- `pcq.agent.validate_json_contract(name, payload)` validates real outputs.
- `tests/test_json_contracts.py` covers success and error envelopes.
- `pcq run --json` stdout remains pure JSON.

### Phase 2: Strictness Semantics

Tasks:

- Make `validate_project(strictness=N)` enforce different gates per level.
- Make `validate_run(output_dir, strictness=N)` enforce the same level policy.
- Keep the level matrix in `pcq.agent.strictness.STRICTNESS_EVIDENCE_MATRIX`.
- Add a `strictness_level` check and top-level `strictness` field to every
  validation report.
- Add missing-evidence checks for levels 3 and 4.
- Persist strictness in `validation_report.json`.
- Update README, SPEC, and MCP docs when behavior changes.

Acceptance:

- `pcq validate --strictness 0` runs only parse/static-minimal checks.
- `pcq validate --strictness 2` preserves current default behavior.
- `pcq validate --strictness 3` fails when reproducibility evidence is missing.
- `pcq validate --strictness 4` fails when service-grade input/lineage evidence
  is missing.
- `strictness_level.evidence.required_evidence` exposes the cumulative matrix in
  both pre-run and post-run validation reports.

### Phase 3: RunRecord Evidence Hardening

Tasks:

- Add `cq_yaml_snapshot` or `cq_yaml_sha256` to RunRecord/source evidence.
- Add lockfile detection and `lockfile_sha256`.
- Add dirty changed files and optional patch hash.
- Add richer runtime environment evidence.
- Add explicit input evidence summary.

Acceptance:

- `describe-run` can summarize reproducibility evidence.
- `validate-run` can fail incomplete RunRecords at strictness 3+.
- Missing evidence is reported as structured checks, not free-form warnings.

v2.11 boost (system-level evidence boundary):

- `pcq.save_partial_run_record()` adds time evidence for in-progress runs
  via atomic tmp+rename. `RunInfo.partial` and `RunInfo.last_updated_at`
  surface streaming state without making pcq interpret it.
- `FailureInfo` adds `error_code` (machine-readable enum) and `evidence`
  (structured key/value) alongside the existing `category` /
  `suggested_fix` (the latter remains natural language for the agent).
- `ExperimentPlanSet` lets agents serialize multi-run intent (fork /
  grid / sweep) without baking sweep policy into pcq. `pcq
  apply-planset` expands a set into N output dirs.

Schema-only by design: pcq keeps providing the dictionary, the agent
makes the call.

### Phase 4: Golden E2E Suite

Tasks:

- Promote the manual MNIST MLP script E2E to an automated test.
- Add a script-style contract E2E that proves agent-authored code can run
  without a pcq framework adapter.
- Add an always-runnable non-Torch example using only core dependencies.
- Add Trainer fake smoke E2E.
- Add project-local atom scaffold -> load -> smoke -> train E2E.
- Add custom `output_dir` E2E.
- Add failed-run E2E with structured failure.
- Add lineage parent/child E2E.

Acceptance:

Each E2E must pass:

```text
pcq inspect
pcq validate
actual run
pcq validate-run
pcq describe-run
artifact existence checks
```

Implementation status:

- `tests/test_golden_e2e.py` covers the Phase 3 default release gate.
- The script-style gate uses synthetic MNIST-like Torch data so it remains fast
  and network-free.
- `examples/contract_numpy.py` and `tests/test_framework_neutral_examples.py`
  prove an adapter-free non-Torch contract script with core dependencies only.
- sklearn-specific coverage should live in an optional extra/dependency gate if
  sklearn is later added as a supported test dependency; it is not required for
  the core pcq contract because pcq should remain framework-adapter-free.

### Phase 5: Agent Authoring Contract

Tasks:

- Expand the agent operating guide with a decision tree:
  - script vs Trainer vs Experiment
  - direct contract script vs project-local atom
  - when to use recipes
- Add forbidden patterns:
  - hard-coded output paths
  - undeclared metrics
  - hidden network downloads
  - project atom import side effects during read-only inspect
  - modifying `pcq` internals for project-specific code
- Add copyable examples for Torch, sklearn, and arbitrary framework scripts.

Acceptance:

- An agent can create a new experiment from the guide without reading source.
- Generated experiments pass strictness 2 before execution.

Implementation status:

- `docs/AGENT_OPERATING_GUIDE.md` is the primary Phase 4 artifact.
- It defines the non-negotiable pcq authoring contract, style decision tree,
  forbidden patterns, and copyable Torch/sklearn/arbitrary-framework script
  patterns.

### Phase 5: Agent Runtime Installation Surface

Tasks:

- Package canonical pcq agent assets inside the installable library.
- Add `pcq agent install --target codex|claude|both`.
- Install Codex assets into:
  - `AGENTS.md`
  - `.agents/skills/pcq/SKILL.md`
- Install Claude Code assets into:
  - `CLAUDE.md`
  - `.claude/skills/pcq/SKILL.md`
- Add `pcq init-experiment --agent codex|claude|both`.
- Support `--dry-run --json` so services and agents can preview file writes.
- Support `--force` for explicit overwrite of managed blocks and skill files.
- Keep default behavior non-destructive:
  - append instruction marker blocks instead of replacing whole files
  - skip existing divergent skill files unless `--force`
  - never modify project files during package install itself

Acceptance:

- `pcq agent install --target codex --json` creates `AGENTS.md` and Codex
  skill files in a fresh project.
- `pcq agent install --target claude --json` creates `CLAUDE.md` and Claude
  skill files in a fresh project.
- `pcq agent install --target both --dry-run --json` reports planned writes
  without creating files.
- Existing `AGENTS.md` / `CLAUDE.md` content is preserved.
- Existing divergent skill files are skipped unless `--force`.
- `pcq init-experiment --agent ...` reuses the same install path.

Implementation status:

- `pcq agent install` is the Phase 5 CLI surface.
- Canonical packaged assets live under `src/pcq/agent_assets/`.
- Human-facing mirrors remain under `templates/` and `skills/`.

### Phase 6: Agent Runtime Status Surface

Tasks:

- Add `pcq agent status --target codex|claude|both --json`.
- Report per-asset status for Codex and Claude runtime files:
  - `installed`
  - `missing`
  - `partial`
  - `stale`
  - `unmanaged`
  - `divergent`
- Include a top-level aggregate status and repair command.
- Keep status strictly read-only.
- Cover fresh project, installed project, modified managed block, unmanaged
  instruction copy, and divergent skill file cases in tests.

Acceptance:

- Fresh project status is `missing`.
- `pcq agent install --target codex` followed by
  `pcq agent status --target codex --json` reports `installed`.
- `pcq agent install --target both` followed by
  `pcq agent status --target claude --json` reports `installed`, including the
  `@AGENTS.md` import form.
- Modified managed instruction blocks report `stale`.
- Existing non-pcq skill files report `divergent`.
- The command exits successfully for readable status states; agents inspect the
  JSON `status` field.

Implementation status:

- `pcq agent status` is the Phase 6 read-only diagnostics surface.

### Phase 7: CQ Service / MCP Integration  ✅ Done (v4.1.0, 2026-05-10)

Implemented natively in `pcq` rather than only in the managed CQ service —
agent runtimes get the same surface from the open-source library.

Delivered:

- `pcq.mcp.server.create_server()` — Anthropic MCP SDK based server,
  stdio + SSE transports.
- 14 MCP tools wrapping the 14 pcq CLI subcommands (subprocess-free):
  `resolve_project`, `inspect_project`, `validate_project`,
  `validate_run`, `describe_run`, `compare_runs`, `lineage_chain`,
  `apply_plan`, `apply_planset`, `init_experiment`, `finalize_run`,
  `agent_install`, `agent_status`, `run_experiment`.
- `pcq mcp serve [--transport stdio|sse]` CLI entry point.
- `pcq agent install --mcp` flag — auto-wires `.mcp.json` in the project
  root with the `pcq mcp serve` server entry (preserves existing
  mcpServers, idempotent without `--force`).
- `pcq[mcp]` optional extras (`mcp>=0.5`).
- Read-only tools (resolve / inspect / validate / describe / compare /
  lineage / status) have no file-system side-effects.
- Write tools mirror the same explicit semantics as their CLI peers.
- Tool I/O reuses the JSON_CONTRACTS registry — schemas stay anchored.
- Tool exceptions surface as `{status: "error", tool: name, error: ...}`
  envelopes instead of propagating.

Acceptance:

- ✅ CQ service can run the full E2E without ad hoc shell parsing.
- ✅ Tool failures are structured and recoverable by an agent.

The CQ managed service can either embed `pcq.mcp.tools.build_tools()`
directly or proxy to a hosted `pcq mcp serve` instance.

### Phase 8: Release Hardening

Tasks:

- Define release checklist.
- ~~Decide PyPI vs private package index.~~ **Decided: PyPI under the `pcq`
  distribution name** — the `cq` slot is already occupied on PyPI and is
  reserved conceptually for the managed CQ service boundary. Python `import
  pcq` and the `pcq` CLI command are the target public surfaces. The actual
  `uv build` + `uv publish` step stays a manual release task; this phase only
  prepares the metadata.
- Validate extras:
  - `pcq[vision]`
  - `pcq[dist]`
  - `pcq[nlp]`
- Require full unit suite plus golden E2E suite before tagging.
- Keep tags immutable once pushed.

Acceptance:

- A CQ service deployment can pin one tag and trust its contract behavior.

## Priority Order

1. Implement strictness level gates.
2. Harden RunRecord evidence for level 3.
3. Promote MNIST MLP E2E into the test suite.
4. Add project-local atom E2E.
5. Add agent runtime installation surface.
6. Add agent runtime status diagnostics.
7. Implement CQ service MCP wrappers.
8. Add release checklist and package publishing workflow.

## Dogfood Findings (2026-05-08)

First end-to-end dogfood ran a 9-generation MNIST ML→DL evolution with
fresh per-gen agents (sub-agent dispatch). The agents reached `eval_acc 1.0`
(360/360) by gen 8 and produced the first external evidence of pcq's value
and friction. See `.cq/runtime/ideas/pcq-mnist-dogfood.md` for the full
record.

### Verified value (real)
- Lineage auto-tracking with `best_value` populated at every depth
- `validate-run --strictness 3` blocking when reproducibility evidence is
  missing
- `compare-runs direction` (improved/regressed/tied) as an immediate
  agent-readable signal
- `pcq validate --plan` catching a manifest issue *before* training started
- Failed variants preserved in lineage, available to the next agent
- Framework freedom: sklearn → torch with the same contract surface
- `apply-planset` expanding 4 augmentation variants with one command

### 21 ranked gaps surfaced by the dogfood

**P0 — fresh user blocked on first command**
- G7-5: `pcq.config()` does not fall back to `cq.yaml`. PlanSet expand
  multiplies the manual `CQ_CONFIG_JSON` workaround by N. **The single
  highest-leverage fix in pcq today.**

**P1 — every user hits these**
- G0-2: `pcq run` command absent
- G1-2: manifest stale lock-in when reusing an output_dir
- G1-4: `compare-runs` returns `config_changes=0` despite real changes
- G7-1: `apply-planset` writes relative `output_dir`, causing artifacts
  to land at `runs/genN/runs/genN/`

**P2 — recurring friction**
- G0-3: `describe-run --json` missing `best`/`artifacts` — closed by
  decision-facts expansion after v2.12.1
- G0-4: `output/` absorbs runtime tmp files into manifest
- G1-1: ExperimentPlan has no op for editing `train.py`
- G1-3 = G7-3: `apply-plan`/`apply-planset` does not sync the
  `artifacts:` glob with `output_dir` change
- G3-1 = G5-2: `apply-plan` clobbers the `inputs:` section
- G4-1: `compare-runs` returns `-0.0` on tied float deltas
- G7-2: `validate --planset` exit code is mixed with project-level fails
- G7-4: `apply-planset` does not deploy `train.py` to expanded dirs

**P3 — informational**
- G3-2 / G5-1 / G6-1: structural ceiling hypothesis on tiny datasets
- G8-1 / G8-2: TTA timing measurement nuances

**Distribution gap (post-dogfood)**
- G9-1 [P2] **resolved**: PyPI 미발행 — fresh users had to use git URL
  gymnastics. Resolved by publishing the open-source contract library as
  `pcq`, because `cq` is already occupied on PyPI and is reserved
  conceptually for the managed CQ service boundary. See [CHANGELOG
  `[2.13.3]`](../CHANGELOG.md).

### Dogfood-driven release plan

- v2.12: P0 + P1 (the four most-felt fixes plus the new `pcq run` entry)
- v2.13: P2 (apply-plan/planset hygiene, describe-run output completeness)
- v2.13.3: G9-1 distribution-name resolution (`pcq` on PyPI)
- Public publishing: mnist-dogfood repo to GitLab, then a case-study link
  in `docs/`

## Non-Goals

- Building a model zoo.
- Adding framework-specific adapters as the primary integration path.
- Hiding CQ worker behavior behind a second runtime protocol.
- Replacing Lightning, HF Trainer, PyCaret, or custom training scripts.

## Summary

The remaining work is mostly evidence and validation work. `pcq` already has a
usable agent loop. To finish it, the project must make strictness levels real,
make RunRecord evidence stronger, turn manual E2E flows into release gates, and
connect the stable CLI/API surface to CQ service MCP tools.
