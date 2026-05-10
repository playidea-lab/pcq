# Schema Versioning Policy

> Rules for evolving pcq's `schema_version` field across the JSON
> contracts (`spec/JSON_CONTRACTS.md`, `src/pcq/agent/json_contracts.py`,
> and the generated `spec/schemas/*.schema.json`).

## TL;DR

- Every pcq agent-facing JSON object carries `schema_version` (integer).
- **Within a major schema_version (e.g. `1`), changes are additive-only.**
  Callers may depend on every required field; pcq may add more.
- A breaking change → bump `schema_version` (1 → 2). Two majors
  coexist in the runtime for at least one minor pcq release.
- The library version (`pcq.__version__`, semver) is independent of
  `schema_version`. Many pcq releases share `schema_version=1`.

## Why "additive-only within MAJOR"

The frozen registry in [`src/pcq/agent/json_contracts.py`](../src/pcq/agent/json_contracts.py)
opens with this guarantee, in code:

```python
"""
The contracts here intentionally describe the minimum stable surface,
not every field a command may emit. Within ``schema_version == 1``
these required fields are additive-only: callers may depend on them,
and pcq may add more fields.
"""
```

Lifting this from a docstring to `spec/` makes it the single rule for
external implementations and conformance suites.

## Definitions

- **schema_version** — integer field present on every agent-facing JSON
  object (e.g. `pcq run --json` envelope, `pcq describe-run --json`
  record, `pcq validate-run --json` report).
- **Required field** — field listed in a contract's `required` set in
  `JSON_CONTRACTS` (or in the generated JSON Schema). Callers may rely
  on it being present, with the documented type, in every response that
  carries the matching `schema_version`.
- **Optional field** — anything else. May appear, may be absent, may be
  added in any minor pcq release within the same `schema_version`.

## Rules

### Allowed within the same `schema_version`

- Adding a new optional field to any contract.
- Adding a new contract to the registry.
- Adding a new value to an open enum (where the spec explicitly says
  "open").
- Loosening a constraint that callers were already required to handle
  (e.g. broadening `string | null` to `string | int | null` if `null`
  was always possible — agents must still defensively type-check).
- Making a previously-optional field required, **only** if every
  released pcq version that produced this `schema_version` already
  emitted it.

### Requires a `schema_version` MAJOR bump

- Removing a required field.
- Renaming a required field.
- Changing the type of a required field.
- Changing the meaning of an existing field (semantic break).
- Tightening an enum (removing a previously-valid value).

## Deprecation timeline

When a field, contract, or behavior is slated for removal in the next
MAJOR:

1. **Pre-deprecation release** (any pcq version on the current
   `schema_version`): ship the replacement alongside the old field.
   Both appear in JSON output. Both are documented.
2. **Deprecation release** (still current `schema_version`): keep both
   fields, but the old field's spec entry carries a `deprecated` note
   pointing at the replacement and the planned MAJOR release.
3. **Removal release** (MAJOR bump): old field gone, `schema_version`
   incremented, old contract still served when an agent explicitly
   requests it (see "Coexistence", below).

Minimum deprecation period: **at least one pcq minor release** with
both fields present. There is no calendar guarantee — pcq's release
cadence is irregular — only an *ordering* guarantee.

## Coexistence of two MAJORs

When `schema_version` is bumped from `N` to `N+1`:

- The new pcq runtime serves `N+1` by default.
- A caller that explicitly opts into `N` (e.g. via `--schema 1` CLI
  flag, `schema=1` in MCP tool input, or `Accept-Schema: 1` header for
  future HTTP endpoints) receives the legacy shape.
- Coexistence lasts at least one pcq minor release. After that, `N`
  may be removed; doing so is not itself a `schema_version` bump (it's
  a release-notes entry).

This protects in-flight agents and CI pipelines from same-day breakage
when a major rolls.

## Discovery

Agents can discover the runtime's supported `schema_version` set in
two equivalent ways:

- **CLI**: `pcq resolve --json` envelope includes
  `"supported_schema_versions": [1]` (single-element list today).
- **MCP**: `mcp__pcq__resolve_project` returns the same list under the
  same key.

When an agent receives a response whose `schema_version` is *not* in
its known set, the recommended behavior is:

- Log a structured warning naming the unexpected version.
- Continue processing fields it recognizes (forward-compat: extra
  fields are allowed).
- Treat absence of a known required field as the contract violation,
  not the version itself.

## Library version vs schema version

`pcq.__version__` follows semver (MAJOR.MINOR.PATCH). It changes with
every release — feature, fix, packaging, deprecation, or breaking
change. `schema_version` only changes when the JSON contract itself
breaks. As of this writing:

| pcq library | schema_version |
|-------------|----------------|
| 2.7.x — 2.12.x | `1` |
| 2.13.x — 4.x.x | `1` (frozen) |

A pcq 5.x release that, say, restructured the `validation_report`
shape would bump `schema_version` to `2` while shipping under `pcq 5`.

## Conformance link

Every contract change that needs MAJOR enforcement should have a
golden pair under
[`tests/conformance/<contract>/`](../tests/conformance/) demonstrating
the new shape — see [`CONFORMANCE.md`](./CONFORMANCE.md).
