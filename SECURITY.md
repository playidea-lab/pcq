# Security Policy

## Supported Versions

| Version | Status |
|---------|--------|
| 4.x     | ✅ Supported — security fixes land on `main` and on the next minor PyPI release. |
| 3.x     | ❌ Not supported — please upgrade to 4.x. |
| ≤ 2.x   | ❌ Not supported. |

`pcq` follows [semantic versioning](https://semver.org/) for the public Python API,
and the agent-facing JSON contract is governed by
[`spec/VERSIONING.md`](spec/VERSIONING.md) (`schema_version` MAJOR bumps are
breaking changes; within a MAJOR, the contract is additive-only).

## Reporting a Vulnerability

**Please do not open a public GitHub issue.** Use the private channel:

- **GitHub Security Advisories**:
  https://github.com/playidea-lab/pcq/security/advisories/new
- Or by email to the maintainer (`playideaab@gmail.com`) with the subject
  `pcq security`.

Include, where applicable:

- Affected version(s) — `pcq --version` output.
- The contract or surface involved (CLI / MCP tool / artifact format / spec).
- Minimal reproducer.
- Suggested fix or mitigation, if any.

## Response Expectations

`pcq` is a single-maintainer project. Best-effort response targets:

- **Acknowledgement**: within 5 working days.
- **Triage / severity rating**: within 2 weeks.
- **Fix or mitigation**: timeline communicated after triage.

There are no calendar guarantees beyond *ordering*: a confirmed advisory will
not be silently dropped, and credit will be given to the reporter (unless
they request otherwise) once a fix ships.

## What Is In Scope

- The `pcq` Python package on PyPI and any code under `src/pcq/`.
- The MCP server surface exposed by `pcq mcp serve` and the 14 MCP tools.
- Build artifacts on PyPI (wheel / sdist), including their sigstore
  signatures and SLSA provenance attestations produced by
  `.github/workflows/release.yml`.
- The contract specification under [`spec/`](spec/INDEX.md) — schema
  ambiguities that could enable spoofing of run evidence count as
  in-scope.
- The Dockerfile and the published container image.

## What Is Not In Scope

- The user's own training code (`train.py`, etc.) — pcq deliberately does
  not own the training loop; security of that code is the user's
  responsibility.
- The CQ Go service worker and the CQ managed-service backend (those
  have their own security boundaries; report issues there to the
  appropriate maintainer).
- Third-party dependencies of pcq's `mcp` extras (e.g. the upstream
  `mcp` SDK, `starlette`, `uvicorn`). Please report those upstream.
- Issues in tools that *use* the pcq contract but are not part of this
  repository — for example, third-party MCP clients that mishandle a
  valid pcq response.

## Disclosure Policy

Default: coordinated disclosure after a fix is available on PyPI. The
reporter may request a different schedule (e.g. immediate disclosure if
the issue is already public, or extended embargo for downstream
coordination) and we will discuss it on the advisory thread.

## Past Advisories

None at this time. Once advisories are issued they will be linked from
this section.
