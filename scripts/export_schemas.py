#!/usr/bin/env python3
"""Export pcq.agent.json_contracts.JSON_CONTRACTS registry to spec/schemas/.

For every entry in the frozen registry this writes (or verifies) a
JSON Schema draft-07 file at ``spec/schemas/<contract>.schema.json``.

Usage:
    uv run python scripts/export_schemas.py            # write/update files
    uv run python scripts/export_schemas.py --check    # CI drift check (exit 1 on mismatch)

The schema files are intentionally derived (the registry is the SoT),
so CI must guard against the on-disk copies drifting from the source.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 레지스트리 SoT 임포트 — 여기 외 다른 출처 사용 금지
from pcq.agent.json_contracts import JSON_CONTRACTS, JSON_CONTRACT_VERSION

# 레지스트리의 단순 타입 표기 → JSON Schema 타입
TYPE_MAP: dict[str, str] = {
    "int": "integer",
    "integer": "integer",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "object": "object",
    "float": "number",
    "number": "number",
    "list": "array",
    "array": "array",
    "null": "null",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = REPO_ROOT / "spec" / "schemas"
BASE_ID = "https://playidea-lab.github.io/pcq/spec/schemas"


def _prop_schema(field_type: str) -> dict[str, Any]:
    """레지스트리의 타입 문자열을 JSON Schema property 노드로 변환."""
    json_type = TYPE_MAP.get(field_type, field_type)
    return {"type": json_type}


def contract_to_schema(name: str, contract: dict[str, Any]) -> dict[str, Any]:
    """레지스트리 엔트리 한 개 → JSON Schema draft-07 dict."""
    required_fields = contract.get("required", {})
    optional_fields = contract.get("optional", {})
    enums = contract.get("enums", {})

    properties: dict[str, dict[str, Any]] = {}
    for fname, ftype in {**required_fields, **optional_fields}.items():
        properties[fname] = _prop_schema(ftype)
    # enum 필드는 properties 노드에 enum 키 추가
    for fname, values in enums.items():
        if fname in properties:
            properties[fname]["enum"] = list(values)

    return {
        "$schema": "https://json-schema.org/draft-07/schema#",
        "$id": f"{BASE_ID}/{name}.schema.json",
        "title": name,
        "description": contract.get("description", ""),
        "type": "object",
        "additionalProperties": True,  # additive-only 정책: 추가 필드 허용
        "required": sorted(required_fields.keys()),
        "properties": dict(sorted(properties.items())),
        "x-pcq-schema-version": contract.get("schema_version", JSON_CONTRACT_VERSION),
        "x-pcq-additive-only": contract.get("additive_only", True),
    }


def render_all() -> dict[str, dict[str, Any]]:
    """전체 레지스트리를 schema dict 매핑으로 렌더."""
    return {name: contract_to_schema(name, c) for name, c in JSON_CONTRACTS.items()}


def write_all(target_dir: Path) -> list[Path]:
    """schema 파일 N개를 디스크에 기록. 쓰여진 경로 목록 반환."""
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, schema in render_all().items():
        path = target_dir / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=False) + "\n")
        written.append(path)
    return written


def check_drift(target_dir: Path) -> list[str]:
    """디스크 파일과 레지스트리에서 렌더한 결과를 비교. 불일치 사유 목록 반환."""
    rendered = render_all()
    issues: list[str] = []

    expected_files = {f"{name}.schema.json" for name in rendered}
    on_disk = {p.name for p in target_dir.glob("*.schema.json")} if target_dir.exists() else set()

    missing = expected_files - on_disk
    extra = on_disk - expected_files
    for m in sorted(missing):
        issues.append(f"missing: spec/schemas/{m}")
    for e in sorted(extra):
        issues.append(f"extra (registry no longer has it): spec/schemas/{e}")

    for name, schema in rendered.items():
        path = target_dir / f"{name}.schema.json"
        if not path.exists():
            continue
        actual = json.loads(path.read_text())
        if actual != schema:
            issues.append(f"drift: spec/schemas/{name}.schema.json differs from registry")

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify on-disk schemas match the registry (exit 1 on drift). Used in CI.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=SCHEMAS_DIR,
        help=f"Output directory (default: {SCHEMAS_DIR.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args(argv)

    if args.check:
        issues = check_drift(args.target)
        if issues:
            print("Schema drift detected:", file=sys.stderr)
            for i in issues:
                print(f"  - {i}", file=sys.stderr)
            print(
                "\nRun: uv run python scripts/export_schemas.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK no drift; {len(JSON_CONTRACTS)} contracts in sync")
        return 0

    written = write_all(args.target)
    print(f"Exported {len(written)} schema(s) to {args.target.relative_to(REPO_ROOT)}/")
    for p in written:
        print(f"  - {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
