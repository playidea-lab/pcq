# pcq.run.envelope · config_only

`pcq run --path . --config-only --json` envelope. The `--config-only`
flag tells pcq to materialize `runtime_cfg.json` and emit the envelope
without executing `cq.yaml.cmd`, which makes the case fast and
framework-independent (no PyTorch/sklearn/etc. needed in the test
environment).

Fixture: a minimal `fixtures/project/` containing a valid `cq.yaml`
plus a `train.py` stub that `--config-only` never invokes.

Structural assertions enforced by the matcher in
[`spec/CONFORMANCE.md`](../../../../spec/CONFORMANCE.md):

- `schema_version == 1`
- `status == "config_only"` (one of the registry's allowed enum values)
- `cmd == "uv run python train.py"` (mirrors the fixture's `cq.yaml`)
- `runtime_cfg_path` and `project_root` present (placeholdered — they
  are absolute paths that depend on the test environment).
- All required fields from `pcq.agent.json_contracts.JSON_CONTRACTS["pcq.run.envelope"]["required"]` present.

The live test invokes pcq via subprocess and compares its stdout JSON
against `expected.json` with the `"..."` matcher.
