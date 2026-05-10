"""Conformance suite — golden pair self-validation.

For every ``tests/conformance/<contract>/<case>/expected.json`` file:

1. The contract name (directory) must exist in
   ``pcq.agent.json_contracts.JSON_CONTRACTS``.
2. Every field listed in the contract's ``required`` set must appear
   in expected.json (value may be a literal placeholder ``"..."``).
3. Every field whose name appears in the contract's ``enums`` and whose
   value is *not* a placeholder must be a member of the declared enum.
4. Numeric/boolean/string values that are not placeholders must use a
   JSON type compatible with the registry's declared type.

This is the *minimum* conformance gate — the suite asserts that golden
pairs faithfully reflect the contract registry, before the next PR
extends it to live ``invocation`` runs that compare actual stdout
against expected.json with the placeholder matcher.

The suite runs under ``uv run pytest tests/conformance/ -q``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pcq.agent.json_contracts import JSON_CONTRACTS

CONFORMANCE_ROOT = Path(__file__).parent
PLACEHOLDER = "..."

# 레지스트리의 단순 타입 표기 → 허용되는 Python isinstance 튜플
TYPE_CHECK = {
    "int": int,
    "integer": int,
    "string": str,
    "boolean": bool,
    "bool": bool,
    "object": dict,
    "list": list,
    "array": list,
    "float": (int, float),
    "number": (int, float),
}


def _discover_cases() -> list[tuple[str, str, Path]]:
    """tests/conformance/<contract>/<case>/expected.json 모두 수집."""
    cases: list[tuple[str, str, Path]] = []
    for contract_dir in sorted(CONFORMANCE_ROOT.iterdir()):
        if not contract_dir.is_dir() or contract_dir.name.startswith("_"):
            continue
        for case_dir in sorted(contract_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            expected = case_dir / "expected.json"
            if expected.exists():
                cases.append((contract_dir.name, case_dir.name, expected))
    return cases


CASES = _discover_cases()


@pytest.mark.parametrize(
    ("contract_name", "case_name", "expected_path"),
    CASES,
    ids=[f"{c}/{case}" for c, case, _ in CASES],
)
def test_golden_pair_conforms_to_registry(
    contract_name: str, case_name: str, expected_path: Path
) -> None:
    """골든 expected.json이 frozen registry의 contract 정의를 만족하는지 검증."""
    assert contract_name in JSON_CONTRACTS, (
        f"contract '{contract_name}' is not in JSON_CONTRACTS registry; "
        f"either rename the directory or add the contract to the registry"
    )
    contract = JSON_CONTRACTS[contract_name]
    expected = json.loads(expected_path.read_text())

    # required 필드 모두 존재
    required = contract.get("required", {})
    missing = [f for f in required if f not in expected]
    assert not missing, (
        f"{contract_name}/{case_name}: expected.json missing required fields: {missing}"
    )

    # required 필드의 타입이 placeholder가 아니라면 검증
    for fname, ftype in required.items():
        value = expected[fname]
        if value == PLACEHOLDER:
            continue
        py_type = TYPE_CHECK.get(ftype)
        if py_type is None:
            continue  # 알 수 없는 타입은 통과 (registry에 새 타입이 추가된 경우)
        assert isinstance(value, py_type), (
            f"{contract_name}/{case_name}.{fname}: expected {ftype}, "
            f"got {type(value).__name__} ({value!r})"
        )

    # enum 필드 값 검증 (placeholder 아닌 경우만)
    enums = contract.get("enums", {})
    for fname, allowed in enums.items():
        if fname not in expected:
            continue
        value = expected[fname]
        if value == PLACEHOLDER:
            continue
        assert value in allowed, (
            f"{contract_name}/{case_name}.{fname}: value {value!r} "
            f"not in declared enum {sorted(allowed)}"
        )

    # schema_version은 contract의 schema_version과 정확히 일치해야 함
    if "schema_version" in expected and expected["schema_version"] != PLACEHOLDER:
        contract_version = contract.get("schema_version", 1)
        assert expected["schema_version"] == contract_version, (
            f"{contract_name}/{case_name}: schema_version mismatch — "
            f"expected.json={expected['schema_version']}, registry={contract_version}"
        )


def test_at_least_one_case_per_advertised_contract() -> None:
    """spec/CONFORMANCE.md 'Coverage' 표가 약속한 두 contract에 케이스가 있는지 게이트."""
    advertised = {"pcq.run.envelope", "pcq.describe_run.record"}
    covered = {c for c, _, _ in CASES}
    missing = advertised - covered
    assert not missing, (
        f"spec/CONFORMANCE.md advertises coverage for {sorted(advertised)} "
        f"but no case found for {sorted(missing)}"
    )


def test_registry_types_recognized() -> None:
    """레지스트리의 모든 타입 토큰이 TYPE_CHECK에 매핑돼 있어야 향후 drift 감지."""
    seen_types: set[str] = set()
    for contract in JSON_CONTRACTS.values():
        for ftype in contract.get("required", {}).values():
            seen_types.add(ftype)
        for ftype in contract.get("optional", {}).values():
            seen_types.add(ftype)
    unknown = seen_types - set(TYPE_CHECK.keys())
    # 알 수 없는 타입이 등장하면 명시적으로 알려야 함
    assert not unknown, (
        f"registry uses types not handled by conformance suite: {sorted(unknown)}; "
        f"add them to TYPE_CHECK in tests/conformance/test_conformance.py"
    )
