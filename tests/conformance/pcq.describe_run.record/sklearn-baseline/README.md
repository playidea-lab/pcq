# pcq.describe_run.record · sklearn-baseline

Real `pcq describe-run output --json` record from a fresh sklearn
dogfood run (`examples/contract_sklearn`, 2026-05-10, Iris,
RandomForest, eval_acc reaches 1.0 in epoch 0). Volatile fields
(timestamps, git_sha, sha256, absolute paths, run_id random suffix)
are placeholdered as `"..."`.

Structural assertions:

- All required fields from `pcq.agent.json_contracts.JSON_CONTRACTS["pcq.describe_run.record"]["required"]` present.
- `decision_facts` includes the canonical 9 booleans/integers an agent uses for go/no-go decisions.
- `best.metrics` shape is `{<declared metric>: <value>}` not a flat scalar.

For the matcher and placeholder rules, see [`spec/CONFORMANCE.md`](../../../../spec/CONFORMANCE.md).
