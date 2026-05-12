# pcq Vision

## One-Line Future

`pcq` is the framework-neutral evidence and control layer for agent-run ML
experiments.

It does not compete with the means of training. It lets any means of training
be executed, observed, validated, compared, and repeated through a standard
experiment boundary.

See [pcq v4 Direction](V4_DIRECTION.md) for the authoritative v4 identity.

## Product Position

```text
PyTorch / HF Trainer / Lightning / sklearn / XGBoost / TabPFN / PyCaret
  -> train or evaluate models

W&B / MLflow / dashboards
  -> track and visualize experiments

DVC / data platforms
  -> version data and artifacts

CQ service
  -> managed execution, queueing, dashboards, and agent loop

pcq
  -> local evidence/control contract around the run
```

`pcq` should not become a smaller clone of any tool above. Its value is the
stable boundary that lets agents and services operate experiments regardless of
the training stack.

## Core Mental Model

```text
project/
  cq.yaml
  train.py or project-local scripts
  data/ or declared inputs
  output/

cq.yaml
  -> command
  -> configs
  -> declared metrics
  -> inputs
  -> artifact globs

training code
  -> any ML framework or custom logic
  -> pcq.config()
  -> pcq.output_dir()
  -> pcq.log(...)
  -> pcq.save_all(...)

pcq CLI
  -> run
  -> validate
  -> describe
  -> compare
  -> lineage
  -> apply-plan
```

The long-term object is the `RunRecord`: a structured experiment record that can
be replayed, audited, compared, and used as the parent for the next
agent-authored experiment. See [RunRecord Standard](RUN_RECORD.md).

## The Target Loop

```text
user goal
  -> agent writes or modifies project-local experiment code
  -> pcq run --jsonl
  -> standard artifacts
  -> pcq validate-run
  -> pcq describe-run
  -> pcq compare-runs
  -> pcq apply-plan
  -> next experiment
```

The agent is free to use PyTorch, Hugging Face Trainer, Lightning, sklearn,
XGBoost, TabPFN, PyCaret, a shell command, or custom code. The required part is
not the framework. The required part is the contract around the run.

## Design Principles

1. Contract first
   - Env, stdout, filesystem, JSON, and JSONL remain the runtime interface.
   - The contract must work locally, in CI, in notebooks, and inside CQ service.

2. Do not compete with training tools
   - No required trainer abstraction.
   - No model zoo as product identity.
   - No framework adapter matrix.
   - Project-local code owns project-specific models, datasets, losses, and
     training loops.

3. Evidence before opinion
   - `pcq` reports facts.
   - Agents or services choose policy from those facts.
   - `describe-run` and `compare-runs` should not decide whether a run is
     acceptable; they should expose enough evidence for that decision.

4. Machine surfaces first
   - Every agent-facing command needs JSON or JSONL.
   - Read-side commands should be side-effect-free.
   - Output paths must resolve through the same `cq.yaml`/runtime resolver.

5. CQ-compatible, not CQ-only
   - CQ service is one managed consumer of the contract.
   - The open-source library must remain independently useful.

## What pcq Must Not Become

- a Lightning clone
- a Hugging Face Trainer clone
- a callback/plugin ecosystem
- a model zoo
- a framework adapter marketplace
- a CQ Hub/Drive client hidden inside the core package
- a policy engine that decides research direction
- a config magic layer that hides the actual command

## Success Criteria

`pcq` is succeeding when:

- an agent can discover the contract from docs/site files
- a project can use any ML framework without a pcq adapter
- every run emits standard artifacts or a structured failure
- `run_record.json` is sufficient to explain how the result was produced
- `validate-run` can state which evidence is present or missing
- `describe-run` exposes decision facts without prose scraping
- `compare-runs` exposes metric/config/source/artifact differences
- lineage lets an agent understand experiment ancestry
- `apply-plan` can start the next iteration without editing pcq internals

## Three Metadata Fields (v4.4–4.6)

pcq treats three fields as first-class evidence: they record *who made the run*
(`attribution`), *what execution environment was used* (`worker_spec`), and
*which version of code and data produced the result* (`fingerprint`). Together
they turn a bare metrics file into a traceable, reproducible record — an agent
or auditor can reconstruct the exact conditions without reading prose or hunting
through logs. PII exposure is handled through a layered policy: attribution may
be omitted or anonymised when the evidence is shared externally.

| Field | Captures |
|---|---|
| `attribution` | author / agent identity that initiated the run |
| `worker_spec` | runtime environment — hardware, OS, driver, container |
| `fingerprint` | content hash of code tree and declared data inputs |

## External Consumers

pcq's evidence artifacts may be consumed by services that orchestrate runs
(e.g., CQ) or aggregate evidence (e.g., The Commons — a separate open-source
project in development). pcq itself defines only the format and contract;
downstream consumers decide what to do with the evidence. Matching runs to
goals, routing work to workers, or presenting results in a UI are not pcq
concerns — they belong to the consumer layer.

## North Star

```text
cq.yaml + project-local training code
  -> pcq run
  -> contract artifacts
  -> RunRecord
  -> validation facts
  -> comparison facts
  -> next agent action
```

That is the durable value. Training methods can change. The experiment boundary
should stay small, explicit, and reliable.
