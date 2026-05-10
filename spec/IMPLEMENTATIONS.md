# pcq Implementations

> Registered implementations of the pcq contract. The contract itself
> lives at [`spec/INDEX.md`](./INDEX.md); each implementation listed
> here targets a specific `schema_version` and ships its own
> conformance evidence.

Two **labels** are used to make the kind of evidence explicit:

- **Reference** — passes the automated suite at
  [`tests/conformance/`](../tests/conformance/) (self-validation +
  live invocation with the `"..."` matcher from
  [`spec/CONFORMANCE.md`](./CONFORMANCE.md)).
- **Production** — observed conforming in production by way of a
  dated dogfood case study. No automated cross-implementation suite
  yet; that is on the roadmap.

A single implementation may carry both labels.

---

## ① Reference — Python (this repository)

| Field | Value |
|-------|-------|
| Slug | `pcq-py` |
| PyPI | [`pcq`](https://pypi.org/project/pcq/) |
| Install | `uv add pcq` (or `uv add 'pcq[mcp]'` for the MCP server) |
| Source | [`src/pcq/`](../src/pcq/) |
| License | Apache-2.0 |
| Targets `schema_version` | `1` |
| Conformance evidence | [`tests/conformance/`](../tests/conformance/) — self-validation (4 tests) + live invocation (2 cases: `pcq.run.envelope/config_only`, `pcq.describe_run.record/sklearn-baseline`); CI gates drift via [`scripts/export_schemas.py --check`](../scripts/export_schemas.py) |
| Maintainer | [playidea-lab](https://github.com/playidea-lab) |

This is the canonical implementation. It also generates the
machine-readable schemas under [`spec/schemas/`](./schemas/) from its
own frozen registry, which means **the spec, the suite, and the
reference are kept in lockstep on every PR**.

## ② Production — CQ Go Service Worker

| Field | Value |
|-------|-------|
| Slug | `cq-go-worker` |
| Source | CQ Go service worker (the production dispatcher that consumes `cq.yaml` + `CQ_CONFIG_JSON` and produces the standard 6-artifact set) |
| Targets `schema_version` | `1` |
| Conformance evidence | Production dogfood — [`docs/case-studies/cq-worker-dogfood-2026-05-10.md`](../docs/case-studies/cq-worker-dogfood-2026-05-10.md) (first end-to-end dispatch on real production infrastructure, 2026-05-10) |
| Automated conformance | Not yet — the case study verifies behavior end-to-end, but the worker does not currently exercise [`tests/conformance/`](../tests/conformance/). Adding a language-neutral runner is on the roadmap. |
| Maintainer | [playidea-lab](https://github.com/playidea-lab) |

This is the second implementation. Listing it here is what keeps the
"two reference implementations" claim in `README.md` and
[`site/llms.txt`](../site/llms.txt) honest: the dogfood case study is
the public, dated evidence the worker actually conforms.

---

## Adding a new implementation

The contract is open. To register a Go, JS, Rust, or any other
implementation:

1. **Pass the conformance suite.** Either run
   [`tests/conformance/`](../tests/conformance/) directly (Python
   reference impl plumbing today) or implement a language-neutral
   runner that consumes the same `input.json` / `expected.json` pairs
   and applies the matcher rules from
   [`spec/CONFORMANCE.md`](./CONFORMANCE.md). Cover at least the
   contracts your implementation produces.
2. **Add an entry to this file.** Include slug, source link,
   `schema_version`, and a link to the conformance evidence (test run
   log, case study, or both).
3. **Mirror the entry in [`site/agent-manifest.json`](../site/agent-manifest.json).**
   The site exposes `implementations[]` machine-readably so agent
   runtimes and indexers can discover registered implementations
   without scraping markdown.
4. **Submit a PR.** Maintainers review for matcher correctness,
   evidence link liveness, and `schema_version` alignment with the
   current registry.

There is no calendar guarantee on review turnaround — pcq's release
cadence is irregular — only an *ordering* guarantee: every merged
implementation entry has working evidence at merge time.

---

## See also

- [`INDEX.md`](./INDEX.md) — top-level spec index
- [`VERSIONING.md`](./VERSIONING.md) — `schema_version` policy and the
  "additive-only within MAJOR" rule
- [`CONFORMANCE.md`](./CONFORMANCE.md) — golden pair format and matcher rules
- [`schemas/`](./schemas/) — auto-exported JSON Schemas (one per
  registry contract, drift-checked in CI)
