# pcq.run.envelope · completed

Real `pcq run --path . --json` envelope from a fresh sklearn dogfood
run (`examples/contract_sklearn`, 2026-05-10). Volatile fields
(absolute paths) are placeholdered as `"..."`. Structural assertions:

- `schema_version == 1`
- `status == "completed"` (one of the registry's allowed enum values)
- `exit_code == 0`
- All required fields from `pcq.agent.json_contracts.JSON_CONTRACTS["pcq.run.envelope"]["required"]` present.

For the matcher and placeholder rules, see [`spec/CONFORMANCE.md`](../../../../spec/CONFORMANCE.md).
