# Agent Acceptance Checklist

## Purpose

This checklist defines whether `pcq` is usable by a coding agent as an
agent-operable experiment boundary.

Passing this checklist means an agent can start from a user goal, create or
modify project-local experiment code, validate it, run it through the `cq.yaml`
contract, and use the resulting RunRecord for the next decision.

See [pcq v4 Direction](V4_DIRECTION.md).

## Acceptance Levels

### Level 0: Discoverable

The agent can understand what `pcq` is without a human explanation.

- [ ] README states that pcq is not a trainer, catalog, adapter matrix, or
      CQ-only client.
- [ ] `docs/V4_DIRECTION.md` defines the product boundary.
- [ ] `site/llms.txt` exposes a compact agent-readable guide.
- [ ] `site/llms-full.txt` exposes a fuller agent-readable guide.
- [ ] `site/agent-manifest.json` lists the primary JSON/JSONL surfaces.

### Level 1: Readable

The agent can understand the project without running training.

- [ ] `pcq resolve --json` identifies project root, `cq.yaml`, command, config,
      metrics, inputs, artifacts, and output directory.
- [ ] `pcq inspect --json` identifies entrypoint and existing output evidence.
- [ ] `pcq validate --json` reports pre-run contract status.
- [ ] Read-side commands do not train.
- [ ] Read-side commands do not download data.
- [ ] Read-side commands do not mutate output directories.
- [ ] Read-side failures are structured warnings/errors, not tracebacks only.

### Level 2: Authorable

The agent can create a valid starting experiment without pcq internals.

- [ ] `pcq init-experiment --style script` creates `cq.yaml` and `train.py`.
- [ ] Generated code reads config through `pcq.config()`.
- [ ] Generated code writes artifacts under `pcq.output_dir()`.
- [ ] Generated code declares and emits at least one metric.
- [ ] Generated code ends with `pcq.save_all(...)` or equivalent
      `pcq.finalize_run(...)`.
- [ ] Generated project can use an arbitrary framework by editing project-local
      code only.
- [ ] No framework adapter is required for HF Trainer, Lightning, sklearn,
      XGBoost, TabPFN, PyCaret, shell commands, or custom code.

### Level 3: Runnable

The agent can execute or hand off the experiment.

- [ ] `cq.yaml.cmd` is the only command the worker needs to run.
- [ ] `CQ_CONFIG_JSON` overrides `cq.yaml.configs` without losing top-level
      context.
- [ ] Relative `output_dir` resolves against project root.
- [ ] Nested cwd execution still resolves the same project.
- [ ] `pcq run --json` emits parseable final JSON only on stdout.
- [ ] `pcq run --jsonl` emits live event objects.
- [ ] `pcq run --events PATH --json` writes event evidence while preserving
      final JSON stdout.
- [ ] child stdout/stderr are captured to explicit paths or events.

### Level 4: Verifiable

The agent can determine whether the run is complete evidence.

- [ ] `pcq validate-run <output_dir> --json` verifies standard artifacts.
- [ ] validation output includes strictness level.
- [ ] validation output includes present and missing evidence.
- [ ] `manifest.json` entries include enough evidence for artifact checks.
- [ ] `metrics.json` is well-formed.
- [ ] `run_summary.json` best/last facts are consistent with metric history.
- [ ] `run_record.json` includes execution, source, environment, inputs,
      metrics, artifacts, summary, validation, and failure fields where
      available.
- [ ] failed or partial runs can be represented as structured evidence.

### Level 5: Comparable

The agent can compare two iterations without prose parsing.

- [ ] `pcq describe-run <output_dir> --json` returns status, target metric,
      best/last values, validation status, artifact summary, reproducibility
      evidence, and `decision_facts`.
- [ ] `pcq compare-runs A B --json` reports metric direction.
- [ ] `pcq compare-runs A B --json` reports config changes.
- [ ] `pcq compare-runs A B --json` reports source/artifact/validation
      differences.
- [ ] incomparable runs return structured reasons.
- [ ] `pcq lineage <output_dir> --json` exposes parent-child ancestry.

### Level 6: Iterable

The agent can start the next experiment.

- [ ] `pcq apply-plan PLAN.json --json` applies bounded config changes.
- [ ] apply output lists changed files and operations.
- [ ] apply rejects unknown operations.
- [ ] apply preserves unrelated project code.
- [ ] parent run identity can be recorded for lineage.
- [ ] agent can choose to edit project-local code directly when research logic
      changes are required.

## Dogfood Scenarios

### Framework-Neutral Script

Acceptance:

- script uses sklearn, HF Trainer, Lightning, XGBoost, TabPFN, PyCaret, shell
  command, or custom code
- no pcq adapter is required
- run produces standard artifacts
- `validate-run` passes at the selected strictness
- `describe-run` exposes decision facts

### Failed Run With Evidence

Acceptance:

- training command exits non-zero
- `pcq run --json` reports the failure
- stdout/stderr evidence is captured
- partial `run_record.json` or structured failure evidence is available when
  the script reached `pcq.save_all(status="failed", ...)`
- `describe-run` does not crash

### Sequential Improvement

Acceptance:

- run A and run B are produced from related configs
- B records parent identity when available
- `compare-runs A B --json` reports metric movement and config/source changes
- `lineage B --json` can identify ancestry
- no terminal prose scraping is required

## Release Gate

A release should not claim full agent-operable status unless:

- docs and site files state the v4 identity clearly
- read-side commands are safe
- JSON/JSONL contracts are stable
- strictness reports are explicit
- standard artifacts are validated
- failed runs remain inspectable
- comparison and lineage facts are available
- project-local training code is first-class
- no built-in training catalog is presented as the product identity
