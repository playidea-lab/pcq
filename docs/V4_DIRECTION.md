# pcq v4 Direction

## Status

This document is the v4 identity and direction note for `pcq`.

It supersedes older v3-era language that positioned `Trainer`, recipes, atoms,
or built-in model/loss/dataset catalogs as product pillars. Those ideas may
remain in old release tags or compatibility branches, but they are not the v4
center of gravity.

## Thesis

`pcq` does not compete with the means of training.

PyTorch, Hugging Face Trainer, Lightning, sklearn, XGBoost, TabPFN, PyCaret,
shell scripts, remote jobs, and project-local research code are all valid ways
to produce an ML result. `pcq` exists to standardize what surrounds that result:
execution intent, live observation, evidence capture, validation, comparison,
lineage, and iteration.

The short version:

```text
pcq = framework-neutral evidence and control layer for agent-run experiments
```

Or more operationally:

```text
pcq does not operate the model.
pcq operates the experiment boundary.
```

## Identity Stack

`agent-readable` is necessary, but it is not the final identity.

| Layer | Meaning | pcq surface |
|---|---|---|
| Agent-readable | An agent can parse the result without scraping prose. | JSON, JSONL, `run_record.json`, `decision_facts` |
| Agent-verifiable | An agent can check whether the result is trustworthy. | strictness, manifest hashes, lockfile/source/config evidence |
| Agent-operable | An agent can run, observe, validate, compare, and repeat experiments through stable commands. | `run`, `validate-run`, `describe-run`, `compare-runs`, `lineage`, `apply-plan` |
| Agent-authorable | An agent can create or modify project-local experiment code while preserving the contract. | contract templates, examples, skills, `llms.txt`, MCP-facing specs |

Therefore, v4 should not be described as a plain evidence schema. It is an
agent-operable experiment boundary. The evidence contract is the base layer that
makes operation safe.

## What pcq Owns

`pcq` owns the boundary of a run:

- how a project declares the command, config, metrics, inputs, and artifacts
- how runtime config is resolved
- where outputs are written
- how metrics are emitted and captured
- how standard artifacts are finalized
- how a run is validated after execution
- how a run is described as machine-readable facts
- how two runs are compared
- how lineage is preserved
- how a structured plan can generate the next experiment
- how agent runtimes discover the contract

The core surfaces are:

```text
cq.yaml
pcq.config()
pcq.output_dir()
pcq.log(...)
pcq.save_all(...)
pcq finalize
pcq run --json
pcq run --jsonl
pcq validate-run --json
pcq describe-run --json
pcq compare-runs --json
pcq lineage --json
pcq apply-plan --json
```

## What pcq Does Not Own

`pcq` should not own the means of training:

- no required trainer abstraction
- no model zoo
- no built-in loss/optimizer/scheduler catalog as a product promise
- no framework adapter matrix
- no hidden orchestration client in the core package
- no attempt to wrap every ML library
- no policy decision about whether a metric is good enough

The contract is the adapter. If project-local code can read config, emit
metrics, write artifacts, and finalize evidence, it is a valid `pcq`
experiment.

## Framework Neutrality

The integration pattern is intentionally boring:

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()

# Use any ML stack here.
# PyTorch, HF Trainer, Lightning, sklearn, XGBoost, TabPFN, PyCaret,
# shell command, remote job wrapper, or custom research code.

pcq.log(epoch=0, eval_score=score)
pcq.save_all(
    history=[{"epoch": 0, "eval_score": score}],
    status="completed",
    artifacts={"model": "model.pkl"},
)
```

There is no supported-framework matrix. A framework is "supported" when a
project script honors the contract.

Repeated patterns may become examples, templates, or documentation. They should
not become core adapters unless there is strong evidence that the adapter
improves the boundary without competing with the underlying training tool.

## Agent Journey

The v4 agent journey should be short and mechanical:

```text
discover pcq instructions
  -> resolve cq.yaml
  -> inspect project state
  -> validate pre-run contract
  -> write or modify project-local training code
  -> run with JSON or JSONL
  -> validate produced artifacts
  -> describe run facts
  -> compare against parent run
  -> apply next structured plan
  -> repeat or stop
```

The agent should not need to decide among `Trainer`, `Experiment`, recipe
catalog, atom registry, and direct script as equivalent product paths. The
default authoring path is a contract script. Project-local helpers are allowed
when they reduce real project complexity, but they are ordinary project code,
not `pcq` core scope.

## Control And Evidence Surfaces

### Read Side

Read-side commands must avoid training imports, heavy optional dependencies, and
output directory mutation.

Required direction:

- `pcq resolve --json` reports the selected project root, command, config,
  output directory, metrics, inputs, and artifacts.
- `pcq inspect . --json` reports project structure and existing evidence without
  importing arbitrary training code by default.
- `pcq validate . --json` validates the pre-run contract and reports warnings or
  blocking failures.
- `pcq describe-run output --json` reports facts only.
- `pcq compare-runs A B --json` reports comparison facts only.
- `pcq lineage output --json` reports ancestry facts only.

### Run Side

Run-side commands must preserve machine-readable output:

- `pcq run --json` emits one final envelope.
- `pcq run --jsonl` emits live event objects.
- `pcq run --events PATH --json` preserves final JSON stdout while writing live
  event evidence to a JSONL file.
- child stdout/stderr are captured and surfaced as paths/tails/events instead
  of requiring ad-hoc shell parsing.

### Write Side

Write-side commands must be bounded:

- `pcq init-experiment --style script` should create a smallest runnable
  contract project.
- `pcq apply-plan` should modify declared config/metadata in a predictable,
  reviewable way.
- agent install/status should be explicit; package installation must not mutate
  agent runtime files.

## CQ Service Relationship

`pcq` is not CQ-only.

```text
pcq = open-source experiment evidence/control library
cq  = managed execution, queueing, artifact collection, dashboard, and agent loop
```

CQ service is one strong consumer of the contract. It can use `pcq` to execute
agent-authored experiments, collect artifacts, compare runs, and continue the
loop. Local scripts, CI jobs, notebooks, and other orchestration systems can
also consume the same contract without CQ service.

The service owns policy and orchestration:

- user goal intake
- prompt/tool strategy
- workspace lifecycle
- queue and GPU scheduling
- credential handling
- dashboard and collaboration UI
- decision policy: accept, rollback, branch, continue, or ask a user

`pcq` owns the local evidence/control layer that makes those service decisions
grounded.

## v4 Public Identity

v4 should present these as the primary product surfaces:

1. `cq.yaml` runtime contract
2. low-level Python helpers: `config`, `output_dir`, `log`, `save_all`,
   `finalize_run`
3. run control: `pcq run --json`, `pcq run --jsonl`, `--events`
4. evidence gates: `validate`, `validate-run`, strictness levels
5. facts: `describe-run`, `compare-runs`, `lineage`
6. iteration: `apply-plan`
7. agent discovery: `llms.txt`, `llms-full.txt`, `agent-manifest.json`, skill
   and instruction templates

Anything else is secondary. If it exists, it should be clearly documented as an
example, compatibility shim, or project-local pattern rather than product
identity.

## Migration From v3 Language

v3-era wording often said:

```text
Trainer / Experiment / recipes / atoms are the way agents operate pcq.
```

v4 wording should say:

```text
The contract is the way agents operate experiments.
Project-local code may use any training tool.
pcq standardizes the run boundary and evidence loop.
```

Concrete documentation changes:

- replace "3-tier API" with "contract-first workflow"
- replace "built-in atoms" with "project-local code or examples"
- replace "framework adapters" with "contract scripts"
- replace "recipe acceptance" as a product gate with "run evidence validation"
- keep `RunRecord`, strictness, JSON/JSONL, lineage, and comparison central

## Completion Criteria

The v4 direction is working when a fresh coding agent can:

- discover pcq from the website or repository without a human explanation
- identify that pcq is not a trainer and not CQ-only
- create a runnable contract script around any ML framework
- run it with `pcq run --jsonl`
- validate the produced evidence
- describe the result using decision facts
- compare it with a parent run
- apply the next structured config plan
- repeat without scraping prose or editing pcq internals

That is the useful meaning of agent-operable.
