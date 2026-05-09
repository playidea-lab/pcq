#!/usr/bin/env bash
# scripts/release-smoke.sh — clean-room end-to-end smoke for pcq releases (v4.0).
#
# Verifies the library works as advertised:
#   - ruff lint clean
#   - full pytest suite passes
#   - real CQ_CONFIG_JSON subprocess produces declared artifacts and metrics
#   - init-experiment template generates a valid contract project
#
# Usage: bash scripts/release-smoke.sh
# CI:    runs in stage `smoke` of .gitlab-ci.yml
# Local: invoke before tagging a release

set -euo pipefail

# Move to repo root
cd "$(git rev-parse --show-toplevel)"

SMOKE_DIR=".smoke_e2e"
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

step() { echo -e "${BLUE}=== $1 ===${NC}"; }
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }

cleanup() {
    if [ -d "$SMOKE_DIR" ]; then
        rm -rf "$SMOKE_DIR"
    fi
}
trap cleanup EXIT

# ── 1. Lint ─────────────────────────────────────────────────────────────
step "1. ruff lint"
uv run ruff check src/ tests/ scripts/ 2>&1 | tail -3
ok "lint clean"

# ── 2. Pytest ───────────────────────────────────────────────────────────
step "2. pytest suite"
uv run pytest tests/ -q --tb=short 2>&1 | tail -5
ok "all tests passed"

# ── 3. Real CQ_CONFIG_JSON subprocess smoke ────────────────────────────
step "3. end-to-end CQ contract simulation"
mkdir -p "$SMOKE_DIR"
cat > "$SMOKE_DIR/cfg.json" <<EOF
{
  "output_dir": "$SMOKE_DIR/output",
  "epochs": 2,
  "batch_size": 16,
  "lr": 0.001,
  "seed": 42,
  "_metrics_declared": ["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc"]
}
EOF

CQ_CONFIG_JSON="$SMOKE_DIR/cfg.json" uv run python examples/train.py \
    > "$SMOKE_DIR/stdout.log" 2> "$SMOKE_DIR/stderr.log" \
    || fail "examples/train.py exited non-zero"

# 3a. contract artifacts present (v4.0: 6 standard contract files via save_all)
for f in config.json metrics.json manifest.json run_summary.json run_record.json validation_report.json; do
    [ -f "$SMOKE_DIR/output/$f" ] || fail "missing artifact: $f"
done
ok "6 contract artifacts present"

# 3b. stdout @key=value lines
grep -q "@epoch=" "$SMOKE_DIR/stdout.log" || fail "stdout missing @epoch="
grep -q "@train_loss=" "$SMOKE_DIR/stdout.log" || fail "stdout missing @train_loss="
grep -q "@eval_loss=" "$SMOKE_DIR/stdout.log" || fail "stdout missing @eval_loss="
ok "stdout has declared @key=value metrics"

# 3c. manifest schema_version (v1.14: schema v2 with sha256 + size_bytes)
SMOKE_DIR="$SMOKE_DIR" uv run python <<'PY'
import hashlib
import json
import os

smoke = os.environ["SMOKE_DIR"]
out_dir = f"{smoke}/output"
m = json.load(open(f"{out_dir}/manifest.json"))
assert m["schema_version"] in (1, 2), f"unexpected schema_version: {m['schema_version']}"
assert len(m["files"]) >= 3, f"manifest has only {len(m['files'])} files"
if m["schema_version"] == 2:
    for f in m["files"]:
        assert "sha256" in f, f"v2 manifest entry missing sha256: {f['path']}"
        assert "size_bytes" in f, f"v2 manifest entry missing size_bytes: {f['path']}"
        assert "created_at" in f, f"v2 manifest entry missing created_at: {f['path']}"
        # sha256 round-trip
        with open(f"{out_dir}/{f['path']}", "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()
        assert actual == f["sha256"], f"sha256 mismatch for {f['path']}"
PY
ok "manifest.json schema valid (v2 sha256 round-trip)"

# 3d. provenance fields
SMOKE_DIR="$SMOKE_DIR" uv run python <<'PY'
import json, os
smoke = os.environ["SMOKE_DIR"]
c = json.load(open(f"{smoke}/output/config.json"))
for key in ("_pcq_version", "_git_sha"):
    assert key in c, f"config.json missing {key}"
PY
ok "provenance fields present"

# 3e. metrics.json history
SMOKE_DIR="$SMOKE_DIR" uv run python <<'PY'
import json, os
smoke = os.environ["SMOKE_DIR"]
m = json.load(open(f"{smoke}/output/metrics.json"))
assert "history" in m
assert len(m["history"]) == 2, f"expected 2 epochs, got {len(m['history'])}"
PY
ok "metrics history correct"

# 3f. v1.16: run_record.json + validation_report.json schema
SMOKE_DIR="$SMOKE_DIR" uv run python <<'PY'
import json, os
out_dir = os.environ["SMOKE_DIR"] + "/output"
rr = json.load(open(out_dir + "/run_record.json"))
assert rr["schema_version"] == 1
for k in ("run", "execution", "source", "environment", "metrics", "artifacts"):
    assert k in rr, f"missing: {k}"
assert rr["run"]["status"] == "completed"
assert rr["environment"]["python"]
print(f"run_record OK: schema v{rr['schema_version']}, status={rr['run']['status']}")

vr = json.load(open(out_dir + "/validation_report.json"))
assert vr["schema_version"] == 1
print(f"validation_report OK: status={vr['status']}, {len(vr['checks'])} checks")
PY
ok "run_record + validation_report"

# ── 4. Fresh-user pyproject template (v4.0) ────────────────────────────
# Generates a contract-script project, parses pyproject.toml,
# asserts non-package mode + pcq dependency.
step "4. fresh-user pyproject template"
FRESH_DIR="$SMOKE_DIR/fresh"
mkdir -p "$FRESH_DIR"
uv run pcq init-experiment \
    --output "$FRESH_DIR" --force --with-pyproject --json > /dev/null \
    || fail "init-experiment --with-pyproject command failed"
[ -f "$FRESH_DIR/pyproject.toml" ] || fail "pyproject.toml not generated"

uv run python <<PY
import tomllib
with open("$FRESH_DIR/pyproject.toml", "rb") as f:
    data = tomllib.load(f)
assert data["project"]["name"], "project.name missing"
assert "pcq" in str(data["project"]["dependencies"]), "pcq dep missing"
assert "tool" not in data or "sources" not in data.get("tool", {}).get("uv", {}), \\
    "[tool.uv.sources] should be absent (pcq is on PyPI)"
assert data.get("tool", {}).get("uv", {}).get("package") is False, \\
    "tool.uv.package must be false (non-package experiment project)"
assert "build-system" not in data, \\
    "[build-system] block back — hatchling wheel build will fail"
PY
ok "fresh-user pyproject template (pcq distribution)"

# ── 5. init-experiment generates working contract script ───────────────
step "5. init-experiment generates a runnable contract script"
INIT_DIR="$SMOKE_DIR/init"
mkdir -p "$INIT_DIR"
uv run pcq init-experiment --output "$INIT_DIR" --force --json > /dev/null \
    || fail "init-experiment failed"
[ -f "$INIT_DIR/cq.yaml" ] || fail "cq.yaml not generated"
[ -f "$INIT_DIR/train.py" ] || fail "train.py not generated"

# inspect generated project
uv run pcq inspect "$INIT_DIR" --json > "$SMOKE_DIR/inspect.json" \
    || fail "inspect failed on init-experiment output"
ok "init-experiment template produces inspectable project"

step "✅ All smoke checks passed"
