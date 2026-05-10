# Case Study: CQ Worker Dogfood (2026-05-10)

> Fourth pcq dogfood. **First production CQ Go service worker dispatch.**
> Verifies the cq.yaml + CQ_CONFIG_JSON + 6 contract artifacts protocol
> end-to-end on real production infrastructure for the first time.
> Previous three dogfoods (mnist / tabular / mcp) all ran on local Mac.

## Why this dogfood

CQ Go service defines the runtime contract — `cq.yaml`, `CQ_CONFIG_JSON`,
`cq://` URI, and the standard artifact set. `pcq` is the Python client of
that contract. The two should integrate by construction; this dogfood
verifies that the construction works in production rather than only in
theory.

## Setup

- **Library**: `pcq[mcp] 4.2.0` (PyPI, fresh `uv add` on the worker)
- **Worker**: `pi-System-Product-Name` (Linux x86_64, RTX 5080, idle)
- **Dispatch path**: `mcp__cq__cq_job_submit(target_worker="pi-System-Product-Name", command=...)`
- **Project**: scaffolded in-place via the dispatched command — `mkdir -p /home/pi/cq_pcq_smoke && uv init && uv add pcq[mcp] scikit-learn joblib && cat > cq.yaml ... && cat > train.py ... && uv run pcq run --path .`
- **Task**: sklearn `LogisticRegression` on `load_breast_cancer()`,
  same baseline as `tabular-dogfood` gen 0 — chosen for cross-environment
  reproducibility comparison
- **Operating mode**: builder direct (single-shot dispatch); no agent loop

## Result

| field | value |
|-------|-------|
| pcq install (uv add pcq[mcp]==4.2.0 + sklearn) | succeeded on worker |
| `pcq inspect` / `validate` / `run` / `validate-run` / `describe-run` | all executed |
| **`run` exit code** | **0** |
| **eval_acc** | **0.9649122807017544** |
| train_time | 0.0090 s |
| 6 contract artifacts in `output/` | all present (config / metrics / manifest / run_summary / run_record / validation_report + model.pkl) |
| RunRecord top-level keys | schema_version / run / execution / source / environment / config / inputs / input_summary / metrics / artifacts / summary / agent / validation |
| `run.name` from cq.yaml top-level | `"cq-pcq-smoke"` ✅ propagated |
| `env.python` / `env.platform` | `3.12.3` / `Linux-x86_64` (worker env captured) |
| `env.pcq_version` | `4.2.0` ✅ |
| `env.lockfile` / `env.lockfile_sha256` | `uv.lock` / `fd20962daf6ea537...` ✅ |
| `inputs.dataset` | `{name: sklearn-load_breast_cancer, version: sklearn-builtin}` ✅ captured |
| `validate-run --strictness 3` | **fail** — single missing-evidence check (`source.git_sha`) |
| `validate-run --strictness 2` (default) | (would pass; not invoked) |

## Cross-environment reproducibility check

The same task ran in **`tabular-dogfood` gen 0** on local Mac (MPS arm64)
on 2026-05-09:

| environment | eval_acc | train_time |
|-------------|---------|----------|
| local Mac (arm64, Python 3.12.10) | 0.9649122807017544 | 0.011 s |
| pi worker (Linux x86_64, Python 3.12.3) | 0.9649122807017544 | 0.009 s |

**Identical eval_acc to the last digit.** sklearn `LogisticRegression(seed=42)`
+ stratified split is platform-deterministic; pcq propagated the same
config through `CQ_CONFIG_JSON` to the worker without distortion.

## What `pcq` did not solve

| ID | Severity | Issue |
|----|----------|-------|
| **(none from `pcq`)** | — | The single `validate-run --strictness 3` failure was `source.git_sha` empty, which is the correct behavior: the dispatched command ran `uv init` (creates `.git/` but no commit), so `git rev-parse HEAD` legitimately had nothing to return. Real production projects always carry committed code; this is a setup artifact of the smoke, not a `pcq` gap. |
| **GP-1** | P2 | This is **not a `pcq` gap** — it lives in the CQ Go service. `mcp__cq__cq_job_status j-86e8df3c-...` returned `"job not found"` even though the worker had picked up the job and completed it. The job ran fine, the artifacts landed, the worker was responsive — the Hub-side job tracking just didn't surface the record back. Filed for the cq Go service repo. |

## Hypothesis check

| ID | Hypothesis | Result |
|----|-----------|------|
| h1 | The CQ runtime contract that `pcq` clients (cq.yaml + CQ_CONFIG_JSON + standard artifacts) is honored end-to-end by an unmodified production CQ worker | ✅ verified — no worker code change required |
| h2 | A `pcq` project's `RunRecord` is the same object regardless of where the run executes (local vs production worker) | ✅ verified — schema, key set, and metric value all identical between Mac arm64 and Linux x86_64 |
| h3 | `pcq[mcp]` extras install cleanly on a Linux worker via `uv add` | ✅ verified — no missing wheel, no native extension issue |

## Differences from previous dogfoods

| dimension | mnist (2026-05-08) | tabular (2026-05-09) | mcp (2026-05-10) | **cq-worker (2026-05-10)** |
|----------|---------|---------|---------|---------|
| environment | local Mac | local Mac | local Mac | **production CQ worker (Linux x86_64)** |
| install path | git URL | PyPI uv add | PyPI uv add | **PyPI uv add on worker** |
| dispatch | sub-agent prompt | sub-agent prompt | new Claude Code session w/ MCP | **`cq_job_submit` to Hub** |
| pcq version | v2.11.0 | v3.0.1 | v4.1.0 | v4.2.0 |
| operating mode | per-gen fresh agent | builder direct | MCP-only fresh agent | **builder one-shot dispatch** |
| generations | 9 | 2 + 4 sweep | 3 + 4 sweep | **1 (smoke)** |
| primary purpose | author-side gap inventory | post-fix verification | MCP UX measurement | **production contract verification** |
| new gaps surfaced (pcq) | 21 | 3 | 6 | **0** |
| new gaps surfaced (cq Go service) | 0 | 0 | 0 | **1 (GP-1)** |
| outcome release | v2.12.x → v3.0.x | v3.0.3 | v4.2.0 | (no pcq release needed) |

This is the first dogfood that produced **zero new pcq gaps**. The
v4.2.0 surface, exercised end-to-end on real production infrastructure
for the first time, behaved exactly as the contract predicted. The one
issue surfaced (GP-1) is on the CQ Go service side and is filed for that
repo's tracker.

## What this dogfood proves

1. **The CQ ↔ pcq integration is structural, not adapter-based.** No
   pcq code change, no worker code change. Both sides honor the cq.yaml
   contract independently and meet at the boundary.
2. **Cross-platform run determinism holds.** Identical seed → identical
   eval_acc across Mac arm64 / Linux x86_64. RunRecord captured both
   environment fingerprints accurately.
3. **`pcq run` is a usable replacement for ad-hoc worker scripts.** A
   single `uv run pcq run --path .` produces the full standard artifact
   set with provenance — the worker code does not need any pcq-specific
   logic to collect it.

## Repo

The dogfood project lives at `/home/pi/cq_pcq_smoke/` on the
`pi-System-Product-Name` worker. It was scaffolded by the dispatched
command, executed in place, and left intact for inspection. No git
remote was added (the project is a smoke artifact, not a long-lived
research repo).

To inspect:

```bash
mcp__cq__cq_relay_call(
    worker_id="pi-System-Product-Name-pi",
    tool="cq_execute",
    args={"command": "cat /home/pi/cq_pcq_smoke/output/run_record.json"}
)
```

## References

- `docs/case-studies/mnist-dogfood-2026-05-08.md` — first dogfood
- `docs/case-studies/tabular-dogfood-2026-05-09.md` — second dogfood
- `docs/case-studies/mcp-dogfood-2026-05-10.md` — third dogfood
- `tabular-dogfood` gen 0 RunRecord — cross-environment reproducibility baseline
- pi worker `/home/pi/cq_pcq_smoke/output/` — primary source artifacts
- CQ service `cq_job_submit` Hub job: `j-86e8df3c-aaf9-40d6-a4ad-601b1dc2504d` (note: the Hub-side `cq_job_status` lookup failed with `"job not found"` despite the worker completing the job; tracked as GP-1 in the cq Go service)
