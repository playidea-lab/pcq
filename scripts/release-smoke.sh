#!/usr/bin/env bash
# scripts/release-smoke.sh — clean-room end-to-end smoke for pcq releases.
#
# Verifies the library works as advertised:
#   - ruff lint clean
#   - full pytest suite passes
#   - all registered recipes pass acceptance framework
#   - real CQ_CONFIG_JSON subprocess produces declared artifacts and metrics
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

# ── 3. Recipe acceptance framework ─────────────────────────────────────
step "3. recipe acceptance — all registered presets"
uv run python <<'PY'
import sys
from pcq.testing import list_failures
fails = list_failures()
if fails:
    for f in fails:
        print(f)
        print()
    sys.exit(1)
print("acceptance: 0 failures")
PY
ok "all recipes pass acceptance"

# ── 4. Real CQ_CONFIG_JSON subprocess smoke ────────────────────────────
step "4. end-to-end CQ contract simulation"
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

# 4a. artifacts present (v1.16: + run_record.json + validation_report.json)
for f in model.pt config.json metrics.json last.ckpt best.ckpt manifest.json run_summary.json run_record.json validation_report.json; do
    [ -f "$SMOKE_DIR/output/$f" ] || fail "missing artifact: $f"
done
ok "9 artifacts present"

# 4b. stdout @key=value lines
grep -q "@epoch=" "$SMOKE_DIR/stdout.log" || fail "stdout missing @epoch="
grep -q "@train_loss=" "$SMOKE_DIR/stdout.log" || fail "stdout missing @train_loss="
grep -q "@eval_loss=" "$SMOKE_DIR/stdout.log" || fail "stdout missing @eval_loss="
ok "stdout has declared @key=value metrics"

# 4c. manifest schema_version (v1.14: schema v2 with sha256 + size_bytes)
SMOKE_DIR="$SMOKE_DIR" uv run python <<'PY'
import hashlib
import json
import os
import sys

smoke = os.environ["SMOKE_DIR"]
out_dir = f"{smoke}/output"
m = json.load(open(f"{out_dir}/manifest.json"))
assert m["schema_version"] in (1, 2), f"unexpected schema_version: {m['schema_version']}"
assert len(m["files"]) >= 5, f"manifest has only {len(m['files'])} files"
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

# 4d. provenance fields
SMOKE_DIR="$SMOKE_DIR" uv run python <<'PY'
import json, os, sys
smoke = os.environ["SMOKE_DIR"]
c = json.load(open(f"{smoke}/output/config.json"))
for key in ("_pcq_version", "_git_sha"):
    assert key in c, f"config.json missing {key}"
PY
ok "provenance fields present"

# 4e. metrics.json history
SMOKE_DIR="$SMOKE_DIR" uv run python <<'PY'
import json, os
smoke = os.environ["SMOKE_DIR"]
m = json.load(open(f"{smoke}/output/metrics.json"))
assert "history" in m
assert len(m["history"]) == 2, f"expected 2 epochs, got {len(m['history'])}"
PY
ok "metrics history correct"

# 4f. v1.16: run_record.json + validation_report.json schema
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

# ── 5. Fresh-user pyproject template (v2.1.1+) ─────────────────────────
# Catches breakage in `pcq init-experiment --with-pyproject` template
# before users hit it. Generates a project, parses the resulting
# pyproject.toml, and asserts non-package mode + git source.
step "5. fresh-user pyproject template"
FRESH_DIR="$SMOKE_DIR/fresh"
mkdir -p "$FRESH_DIR"
uv run pcq init-experiment --style trainer --preset vision/fake_smoke \
    --output "$FRESH_DIR" --force --with-pyproject --json > /dev/null \
    || fail "init-experiment --with-pyproject command failed"
[ -f "$FRESH_DIR/pyproject.toml" ] || fail "pyproject.toml not generated"

uv run python <<PY
import tomllib
with open("$FRESH_DIR/pyproject.toml", "rb") as f:
    data = tomllib.load(f)
assert data["project"]["name"], "project.name missing"
# v3.0.0: PyPI distribution, Python import, and CLI command are all "pcq".
assert "pcq" in str(data["project"]["dependencies"]), "pcq dep missing"
# v3.0.1: pcq is on PyPI — generated template no longer pins via git source.
assert "tool" not in data or "sources" not in data.get("tool", {}).get("uv", {}), \\
    "v3.0.1 regression: [tool.uv.sources] should be absent now that pcq is on PyPI"
# v2.1.1: non-package mode (no [build-system], no hatchling wheel)
assert data.get("tool", {}).get("uv", {}).get("package") is False, \\
    "tool.uv.package must be false (non-package experiment project)"
assert "build-system" not in data, \\
    "P0 regression: [build-system] block back — hatchling wheel build will fail"
PY
ok "fresh-user pyproject template (pcq distribution)"

step "✅ All smoke checks passed"
