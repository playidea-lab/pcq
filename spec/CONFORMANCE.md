# Conformance — Golden Pair Suite

> What "this implementation is conformant with pcq schema_version=N"
> means, where the evidence lives, and how to run it.

## Purpose

The frozen registry in `src/pcq/agent/json_contracts.py` defines what
each agent-facing JSON object *should* look like. Conformance turns
that into *evidence* — for every contract in the registry, a small set
of input/expected pairs that any conformant implementation (the
reference Python `pcq`, a future Go/JS client, a CQ Go service
ingestion path) must satisfy.

Conformance is not a test of pcq the library. It is a test of *the
contract* — and the library is one of N possible passing
implementations.

## Layout

```
tests/conformance/
  <contract>/                       e.g. pcq.describe_run.record/
    <case>/                         e.g. sklearn-baseline-completed/
      input.json                    Input envelope (see below).
      expected.json                 Expected output, with placeholders.
      README.md                     One-paragraph description of the case.
  test_conformance.py               Pytest entrypoint that walks the tree.
```

Naming:

- `<contract>` matches a key from
  [`JSON_CONTRACTS`](../src/pcq/agent/json_contracts.py) (e.g.
  `pcq.describe_run.record`).
- `<case>` is a kebab-case label describing the scenario
  (`sklearn-baseline-completed`, `compare-runs-ancestor`, …).

## `input.json` shape

```json
{
  "invocation": {
    "kind": "cli",                              // "cli" | "mcp" | "python"
    "command": ["pcq", "describe-run", "FIXTURE_DIR", "--json"]
  },
  "fixtures": {
    "FIXTURE_DIR": "fixtures/sklearn-completed/"
  },
  "schema_version": 1
}
```

The runner substitutes any token of the form `FIXTURE_KEY` in
`invocation.command` with the corresponding entry in `fixtures`,
resolved against the case directory. Fixtures may include any files
the contract expects (e.g. a fully populated `output/` dir for
`describe_run`).

## `expected.json` shape

The expected response, with **placeholders** for fields that depend on
the runtime environment:

```json
{
  "schema_version": 1,
  "run_id": "...",
  "name": "sklearn-baseline",
  "status": "completed",
  "best_value": 1.0,
  "best_epoch": 0,
  "validation_status": "pass",
  "decision_facts": {
    "run_completed": true,
    "validation_passed": true,
    "has_target_metric": true,
    "has_best": true,
    "artifact_count": 4
  },
  "git_sha": "...",
  "last_updated_at": "..."
}
```

## Placeholder & matcher policy

A field whose value is the literal string `"..."` matches *any* value
of any type at that JSON path. This keeps the suite environment-
independent while preserving structural assertions.

The standard volatile-field set covered by `"..."`:

| Field family | Why it varies |
|--------------|---------------|
| `last_updated_at`, `created_at`, `finished_at`, `started_at` | Wall-clock time |
| `git_sha`, `cq_yaml_sha256`, `lockfile_sha256`, `sha256` | Source/file digests |
| Absolute paths: `output_dir`, `stdout_path`, `stderr_path`, `runtime_cfg_path`, `project_root` | Filesystem layout |
| `run_id` | Random suffix |

The runner walks `expected.json` recursively. For each leaf:

- Literal `"..."` → match any value (presence required).
- Any other literal → exact match (`==`).
- Object → match keys present in expected; **extra keys in actual are
  allowed** (forward-compat: additive-only within a `schema_version`).
- Array → element-wise match. Length must match unless expected ends
  with the literal `"..."` element (meaning "and possibly more").

Failure messages report the JSON path of the first mismatch
(`/decision_facts/has_best: expected true, got false`).

## Running the suite

Reference implementation (Python `pcq`):

```bash
uv run pytest tests/conformance/ -v
```

Other implementations (future): the same `input.json` /
`expected.json` files can drive a language-neutral runner that
invokes the implementation under test as a subprocess, captures
stdout, and compares against the matcher rules above.

## Adding a case

1. Pick the contract key (`pcq.describe_run.record`, etc.).
2. Create `tests/conformance/<contract>/<case>/`.
3. Drop in `input.json`, `expected.json`, `README.md`, and any
   `fixtures/` the input references.
4. Run `uv run pytest tests/conformance/<contract>/<case> -v` once
   locally to confirm.
5. Commit. CI will exercise the case on every PR.

## What conformance does NOT test

- Performance, memory, latency.
- Correctness of training code (that's the user's project).
- pcq library internals (those have their own unit tests under
  `tests/`).
- Any field outside the registry's `required` set — those are
  permitted to vary or be absent across implementations.

## Coverage as of this PR

| Contract | Cases |
|----------|-------|
| `pcq.run.envelope` | `completed` |
| `pcq.describe_run.record` | `sklearn-baseline` |

More to follow as conformance broadens (`compare_runs.diff`,
`validation_report`, `lineage_chain`, …).

## See also

- [`VERSIONING.md`](./VERSIONING.md) — when the contract may change
  and what counts as breaking.
- [`JSON_CONTRACTS.md`](./JSON_CONTRACTS.md) — narrative description
  of every contract.
- [`schemas/`](./schemas/) — generated JSON Schema files for every
  contract in the registry.
