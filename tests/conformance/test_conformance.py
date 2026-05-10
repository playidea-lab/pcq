"""Conformance suite — golden pair validation.

Two layers per case:

1. **Self-validation** (``test_golden_pair_conforms_to_registry``):
   asserts that ``expected.json`` faithfully reflects the contract
   registry — required fields present, enum values valid, schema_version
   matches.

2. **Live invocation** (``test_live_invocation_matches_expected``):
   runs the case's ``input.json`` invocation as a subprocess (CLI today,
   MCP/python in future), parses stdout JSON, and compares it against
   ``expected.json`` using the ``"..."`` placeholder matcher defined in
   ``spec/CONFORMANCE.md``. Skipped when ``input.json`` is absent or the
   invocation kind is not yet supported by this runner.

Run: ``uv run pytest tests/conformance/ -q``.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from pcq.agent.json_contracts import JSON_CONTRACTS

CONFORMANCE_ROOT = Path(__file__).parent
PLACEHOLDER = "..."
LIVE_TIMEOUT_SEC = 30

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


# ---------------------------------------------------------------------------
# Live invocation layer — placeholder matcher + subprocess runner
# ---------------------------------------------------------------------------


def placeholder_match(actual: Any, expected: Any, path: str = "$") -> str | None:
    """spec/CONFORMANCE.md의 매처 규칙 구현. 불일치 시 JSON path를 담은 사유 문자열 반환, 일치 시 None.

    규칙:
    - expected가 "..."이면 actual의 *어떤* 값이든 OK (단순 존재만 요구).
    - dict면 expected의 모든 키가 actual에 존재하고 재귀 비교. additive-only
      정책에 따라 actual에 추가 키가 있어도 통과.
    - list면 expected가 마지막에 "..."로 끝나면 prefix match + open-ended,
      그 외엔 길이 동일 + element-wise.
    - leaf primitive면 == 비교.
    """
    if expected == PLACEHOLDER:
        return None  # 임의 값 OK
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path}: expected object, got {type(actual).__name__}"
        for key, exp_val in expected.items():
            sub_path = f"{path}.{key}"
            if key not in actual:
                return f"{sub_path}: missing in actual"
            err = placeholder_match(actual[key], exp_val, sub_path)
            if err is not None:
                return err
        return None  # actual의 추가 키는 허용 (additionalProperties=true)
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return f"{path}: expected array, got {type(actual).__name__}"
        open_ended = bool(expected) and expected[-1] == PLACEHOLDER
        if open_ended:
            head = expected[:-1]
            if len(actual) < len(head):
                return f"{path}: expected at least {len(head)} elements, got {len(actual)}"
            for i, exp_v in enumerate(head):
                err = placeholder_match(actual[i], exp_v, f"{path}[{i}]")
                if err is not None:
                    return err
            return None
        if len(actual) != len(expected):
            return f"{path}: expected length {len(expected)}, got {len(actual)}"
        for i, exp_v in enumerate(expected):
            err = placeholder_match(actual[i], exp_v, f"{path}[{i}]")
            if err is not None:
                return err
        return None
    # leaf
    if actual != expected:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def _live_invocation_cases() -> list[tuple[str, str, Path]]:
    """input.json이 있고 invocation.kind=='cli'인 케이스만 수집."""
    out: list[tuple[str, str, Path]] = []
    for contract, case, expected_path in CASES:
        case_dir = expected_path.parent
        input_path = case_dir / "input.json"
        if not input_path.exists():
            continue
        try:
            spec = json.loads(input_path.read_text())
        except json.JSONDecodeError:
            continue
        if spec.get("invocation", {}).get("kind") == "cli":
            out.append((contract, case, case_dir))
    return out


LIVE_CASES = _live_invocation_cases()


@pytest.mark.parametrize(
    ("contract_name", "case_name", "case_dir"),
    LIVE_CASES,
    ids=[f"{c}/{case}" for c, case, _ in LIVE_CASES],
)
def test_live_invocation_matches_expected(
    contract_name: str, case_name: str, case_dir: Path
) -> None:
    """input.json invocation을 실제로 호출하고 stdout JSON을 expected.json과 매처로 비교."""
    spec = json.loads((case_dir / "input.json").read_text())
    expected = json.loads((case_dir / "expected.json").read_text())

    invocation = spec.get("invocation", {})
    assert invocation.get("kind") == "cli", (
        f"{contract_name}/{case_name}: only invocation.kind='cli' supported by this runner"
    )

    raw_command: list[str] = list(invocation["command"])
    # FIXTURE_KEY 토큰을 케이스 디렉토리 기준 절대 경로로 치환
    fixtures = spec.get("fixtures", {}) or {}
    resolved: list[str] = []
    for tok in raw_command:
        if tok in fixtures:
            resolved.append(str((case_dir / fixtures[tok]).resolve()))
        else:
            resolved.append(tok)

    proc = subprocess.run(
        resolved,
        capture_output=True,
        text=True,
        timeout=LIVE_TIMEOUT_SEC,
        check=False,
        cwd=str(case_dir),
    )

    # JSON 응답 파싱 — pcq run/describe-run/...은 모두 한 JSON 오브젝트를 stdout에 emit
    try:
        actual = json.loads(proc.stdout)
    except json.JSONDecodeError as err:
        pytest.fail(
            f"{contract_name}/{case_name}: stdout is not valid JSON\n"
            f"  exit_code={proc.returncode}\n"
            f"  stdout (first 500 chars)={proc.stdout[:500]!r}\n"
            f"  stderr (first 500 chars)={proc.stderr[:500]!r}\n"
            f"  parse_error={err}"
        )

    err = placeholder_match(actual, expected)
    if err is not None:
        pytest.fail(
            f"{contract_name}/{case_name}: live response does not match expected.json\n"
            f"  mismatch: {err}\n"
            f"  command: {' '.join(resolved)}\n"
            f"  exit_code: {proc.returncode}"
        )


def test_live_invocation_coverage() -> None:
    """advertised contract마다 라이브 케이스 ≥ 1을 게이트로 강제."""
    advertised = {"pcq.run.envelope", "pcq.describe_run.record"}
    live_covered = {c for c, _, _ in LIVE_CASES}
    missing = advertised - live_covered
    assert not missing, (
        f"advertised contracts without a live invocation case: {sorted(missing)}; "
        f"add input.json with invocation.kind='cli' under those case dirs"
    )
