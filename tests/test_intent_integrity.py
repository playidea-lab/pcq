"""build_intent_object + build_integrity_object 단위 테스트 — T-PCQ2X-6.

각 테스트는 pcq-2x-canonical.md 요구사항(R1,R2,R5,R6)과
spec/SPEC.md ## pcq 2.x Contract 에 대응한다.

AAA 구조(Arrange / Act / Assert), 한국어 docstring, 영어 식별자.
테스트 간 공유 상태 없음 — tmp_path / monkeypatch 픽스처로 격리.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pcq
from pcq.contract import build_intent_object, build_integrity_object


# ─────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _setup_cfg(tmp_path: Path, **extra) -> Path:
    """테스트용 CQ_CONFIG_JSON 파일 생성."""
    cfg: dict = {"output_dir": str(tmp_path), "seed": 42}
    cfg.update(extra)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def _run_full_pipeline(tmp_path: Path, monkeypatch, **save_all_kwargs) -> dict:
    """save_all + run_record.json 까지 실행한 뒤 run_record dict 반환."""
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    history = [{"epoch": 0, "eval_loss": 0.5}]
    pcq.save_all(history=history, **save_all_kwargs)
    return json.loads((tmp_path / "run_record.json").read_text())


def _canonical_hash(subset: dict) -> str:
    """R15 canonical form — build_integrity_object 와 동일한 byte 순서.

    json.dumps(indent=2, sort_keys=True, default=str) → sha256 → 'sha256:<hex>'.
    """
    canonical = json.dumps(subset, indent=2, sort_keys=True, default=str)
    hex_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{hex_digest}"


# ─────────────────────────────────────────────────────────────────────────────
# build_intent_object — 유효한 goal 열거형 5종
# ─────────────────────────────────────────────────────────────────────────────

VALID_GOALS = [
    "baseline_reproduction",
    "sota_challenge",
    "ablation",
    "hyperparam_sweep",
    "exploration",
]


@pytest.mark.parametrize("goal", VALID_GOALS)
def test_build_intent_valid_goal_returns_dict(goal: str):
    """유효한 goal 5종 각각에 대해 intent dict 가 반환되고 경고가 없어야 한다.

    Arrange: 허용된 goal 열거형 값
    Act: build_intent_object(goal=goal)
    Assert: dict 반환, warnings 빈 목록, intent["goal"] == goal
    """
    # Act
    result, warnings = build_intent_object(goal=goal)

    # Assert
    assert result is not None, f"goal={goal!r} → None 반환됨 (dict 기대)"
    assert isinstance(result, dict)
    assert result["goal"] == goal
    assert warnings == [], f"유효한 goal 에서 경고 발생: {warnings}"


# ─────────────────────────────────────────────────────────────────────────────
# build_intent_object — 잘못된 goal → INTENT_GOAL_INVALID 경고, 예외 없음
# ─────────────────────────────────────────────────────────────────────────────

def test_build_intent_invalid_goal_emits_warning_no_exception():
    """알 수 없는 goal 값은 INTENT_GOAL_INVALID 경고를 발생시키고 예외를 던지지 않는다.

    Arrange: 허용 열거형에 없는 goal 문자열
    Act: build_intent_object(goal="unknown_goal")
    Assert: warnings 에 INTENT_GOAL_INVALID 코드 포함, goal 필드 None, 예외 없음
    """
    # Arrange
    bad_goal = "unknown_goal"

    # Act — 예외 없이 반환되어야 함
    result, warnings = build_intent_object(goal=bad_goal)

    # Assert — 경고 코드 확인
    codes = [w.get("code") for w in warnings]
    assert "INTENT_GOAL_INVALID" in codes, f"INTENT_GOAL_INVALID 경고 없음: {warnings}"

    # goal 은 None 으로 처리
    if result is not None:
        assert result["goal"] is None, (
            f"잘못된 goal 은 None 처리되어야 하지만: {result['goal']!r}"
        )


def test_build_intent_invalid_goal_does_not_raise():
    """잘못된 goal 이 ValueError 등 예외를 발생시키지 않아야 한다.

    Arrange: 허용 열거형에 없는 goal
    Act + Assert: 예외 발생 시 테스트 실패
    """
    # Act + Assert: 예외가 발생하면 테스트 실패
    try:
        build_intent_object(goal="definitely_not_a_valid_goal")
    except Exception as exc:
        pytest.fail(f"잘못된 goal 에서 예외 발생 (기대 안 함): {exc!r}")


# ─────────────────────────────────────────────────────────────────────────────
# build_intent_object — malformed expected_baseline / tolerance
# ─────────────────────────────────────────────────────────────────────────────

def test_build_intent_malformed_expected_baseline_emits_warning():
    """올바르지 않은 expected_baseline 구조는 INTENT_TOLERANCE_MALFORMED 경고를 발생시킨다.

    Arrange: metric 키는 있지만 value 가 문자열 (숫자 아님)
    Act: build_intent_object(expected_baseline={"metric": "loss", "value": "bad"})
    Assert: INTENT_TOLERANCE_MALFORMED 경고 포함
    """
    # Arrange
    bad_baseline = {"metric": "loss", "value": "not_a_number"}

    # Act
    result, warnings = build_intent_object(
        goal="ablation",
        expected_baseline=bad_baseline,
    )

    # Assert
    codes = [w.get("code") for w in warnings]
    assert "INTENT_TOLERANCE_MALFORMED" in codes, (
        f"malformed expected_baseline 에서 INTENT_TOLERANCE_MALFORMED 경고 없음: {warnings}"
    )


def test_build_intent_malformed_tolerance_emits_warning():
    """올바르지 않은 tolerance 구조는 INTENT_TOLERANCE_MALFORMED 경고를 발생시킨다.

    Arrange: direction 없이 margin 만 있는 tolerance
    Act: build_intent_object(goal="sota_challenge", tolerance={"margin": 0.05})
    Assert: INTENT_TOLERANCE_MALFORMED 경고 포함
    """
    # Arrange
    bad_tolerance = {"margin": 0.05}  # direction 키 누락

    # Act
    result, warnings = build_intent_object(
        goal="sota_challenge",
        tolerance=bad_tolerance,
    )

    # Assert
    codes = [w.get("code") for w in warnings]
    assert "INTENT_TOLERANCE_MALFORMED" in codes, (
        f"malformed tolerance 에서 INTENT_TOLERANCE_MALFORMED 경고 없음: {warnings}"
    )


def test_build_intent_non_dict_expected_baseline_emits_warning():
    """dict 가 아닌 expected_baseline 은 INTENT_TOLERANCE_MALFORMED 경고를 발생시킨다.

    Arrange: expected_baseline 이 문자열
    Act: build_intent_object(goal="exploration", expected_baseline="not_a_dict")
    Assert: INTENT_TOLERANCE_MALFORMED 경고 포함, 예외 없음
    """
    # Act
    _, warnings = build_intent_object(
        goal="exploration",
        expected_baseline="not_a_dict",  # type: ignore[arg-type]
    )

    # Assert
    codes = [w.get("code") for w in warnings]
    assert "INTENT_TOLERANCE_MALFORMED" in codes


# ─────────────────────────────────────────────────────────────────────────────
# build_intent_object — 전부 None 입력 → None 반환 (omit-not-null 규칙)
# ─────────────────────────────────────────────────────────────────────────────

def test_build_intent_all_null_returns_none():
    """goal/expected_baseline/tolerance 모두 None 이면 None 을 반환한다 (omit-not-null).

    Arrange: 세 인자 모두 기본값(None)
    Act: build_intent_object()
    Assert: result is None, warnings 빈 목록
    """
    # Act
    result, warnings = build_intent_object()

    # Assert
    assert result is None, (
        f"전부 None 입력 시 None 반환 기대, 실제: {result!r}"
    )
    assert warnings == []


# ─────────────────────────────────────────────────────────────────────────────
# build_intent_object — valid expected_baseline + tolerance 는 dict 에 포함
# ─────────────────────────────────────────────────────────────────────────────

def test_build_intent_valid_baseline_and_tolerance_present_in_dict():
    """올바른 expected_baseline 과 tolerance 는 결과 dict 에 포함되어야 한다.

    Arrange: goal + 유효한 expected_baseline + 유효한 tolerance
    Act: build_intent_object(...)
    Assert: result["expected_baseline"] == 입력값, result["tolerance"] == 입력값
    """
    # Arrange
    baseline = {"metric": "eval_loss", "value": 0.35}
    tolerance = {"direction": "lower_is_better", "margin": 0.05}

    # Act
    result, warnings = build_intent_object(
        goal="hyperparam_sweep",
        expected_baseline=baseline,
        tolerance=tolerance,
    )

    # Assert
    assert result is not None
    assert warnings == [], f"유효한 입력에서 경고 발생: {warnings}"
    assert result["expected_baseline"] == baseline
    assert result["tolerance"] == tolerance
    assert result["goal"] == "hyperparam_sweep"


# ─────────────────────────────────────────────────────────────────────────────
# build_integrity_object — content_hash 형식 검증
# ─────────────────────────────────────────────────────────────────────────────

def test_build_integrity_content_hash_format():
    """content_hash 가 'sha256:<64자 hex>' 형식이어야 한다.

    Arrange: 임의 payload dict
    Act: build_integrity_object(payload)
    Assert: content_hash 가 'sha256:' prefix + 64자 hex
    """
    # Arrange
    payload: dict = {"config": {"lr": 0.001}, "metrics": {"eval_loss": 0.42}}

    # Act
    result, warnings = build_integrity_object(payload)

    # Assert
    assert result is not None
    assert warnings == []
    content_hash = result["content_hash"]
    assert isinstance(content_hash, str)
    assert content_hash.startswith("sha256:"), (
        f"content_hash 가 'sha256:' prefix 로 시작해야 함: {content_hash!r}"
    )
    hex_part = content_hash[len("sha256:"):]
    assert len(hex_part) == 64, (
        f"sha256 hex digest 는 64자여야 하지만 {len(hex_part)}자: {hex_part!r}"
    )
    assert all(c in "0123456789abcdef" for c in hex_part), (
        f"sha256 hex digest 에 비 hex 문자 포함: {hex_part!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# build_integrity_object — 결정론적 검증 (동일 payload → 동일 해시)
# ─────────────────────────────────────────────────────────────────────────────

def test_build_integrity_deterministic_same_payload():
    """동일 payload 로 두 번 호출하면 content_hash 가 동일해야 한다.

    Arrange: 동일 payload dict
    Act: build_integrity_object 두 번 호출
    Assert: 두 content_hash 동일
    """
    # Arrange
    payload: dict = {
        "intent": {"goal": "ablation", "expected_baseline": None, "tolerance": None},
        "config": {"lr": 0.01, "batch_size": 32},
        "metrics": {"eval_loss": 0.3},
    }

    # Act
    result1, _ = build_integrity_object(payload)
    result2, _ = build_integrity_object(payload)

    # Assert
    assert result1 is not None
    assert result2 is not None
    assert result1["content_hash"] == result2["content_hash"], (
        "동일 payload 에서 두 번 호출했을 때 content_hash 가 달라야 하지 않음"
    )


# ─────────────────────────────────────────────────────────────────────────────
# build_integrity_object — R15 byte-identical canonical form 검증
# ─────────────────────────────────────────────────────────────────────────────

def test_build_integrity_r15_canonical_form_byte_identical():
    """R15: content_hash 가 canonical json.dumps(indent=2, sort_keys=True, default=str) 형식과 byte 동일.

    build_integrity_object 가 계산한 hash 를 직접 hashlib 로 재현하여
    동일성을 확인한다 — _atomic_write_json 과의 불변식 보장.

    Arrange: payload + 명시적 hashed_fields
    Act: build_integrity_object 호출 후 독립적으로 동일 canonical hash 계산
    Assert: 두 hash 완전 일치
    """
    # Arrange
    payload: dict = {
        "intent": {"goal": "baseline_reproduction", "expected_baseline": None, "tolerance": None},
        "config": {"lr": 0.001, "seed": 42},
        "metrics": None,
        "worker_spec": None,
        "contract_version": "2.0",
    }
    hashed_fields = ["intent", "config", "metrics", "contract_version"]

    # Act — build_integrity_object 호출
    result, warnings = build_integrity_object(payload, hashed_fields=hashed_fields)

    # 독립적으로 canonical hash 계산 (테스트 직접 구현)
    subset = {path: payload.get(path) for path in hashed_fields}
    expected_hash = _canonical_hash(subset)

    # Assert
    assert result is not None
    assert warnings == []
    assert result["content_hash"] == expected_hash, (
        f"canonical hash 불일치:\n"
        f"  build_integrity_object: {result['content_hash']!r}\n"
        f"  직접 계산:              {expected_hash!r}"
    )


def test_build_integrity_r15_dotted_path_canonical_form():
    """R15 dotted path: attribution.author 등 중첩 경로도 canonical hash 에 올바르게 포함.

    Arrange: attribution 포함 payload + 점 구분 hashed_fields
    Act: build_integrity_object 호출
    Assert: 직접 계산한 canonical hash 와 동일
    """
    # Arrange
    payload: dict = {
        "attribution": {
            "author": {"kind": "human", "id": "alice"},
            "committer": {"kind": "agent", "id": "claude"},
            "operator": "alice",
            "signature": "sig-should-be-excluded",
        },
        "config": {"batch_size": 64},
    }
    hashed_fields = ["attribution.author", "attribution.committer", "config"]

    # Act
    result, warnings = build_integrity_object(payload, hashed_fields=hashed_fields)

    # 독립 canonical 계산 — dotted path resolve 포함
    def resolve(p: dict, path: str):
        parts = path.split(".")
        cur = p
        for part in parts:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    subset = {path: resolve(payload, path) for path in hashed_fields}
    expected_hash = _canonical_hash(subset)

    # Assert
    assert result is not None
    assert warnings == []
    assert result["content_hash"] == expected_hash


# ─────────────────────────────────────────────────────────────────────────────
# build_integrity_object — anti-recursion: "integrity" + "attribution.signature" 제외
# ─────────────────────────────────────────────────────────────────────────────

def test_build_integrity_anti_recursion_excludes_integrity_and_signature():
    """anti-recursion: 'integrity' 와 'attribution.signature' 는 hashed_fields 에서 제거된다.

    Arrange: hashed_fields 에 "integrity", "attribution.signature", "intent" 포함
    Act: build_integrity_object(payload, hashed_fields=[...])
    Assert: 결과 hashed_fields 에 "integrity"/"attribution.signature" 없고 "intent" 있음
    """
    # Arrange
    payload: dict = {
        "intent": {"goal": "exploration"},
        "integrity": {"content_hash": "sha256:abc", "hashed_fields": []},
        "attribution": {
            "author": {"kind": "human", "id": "bob"},
            "signature": "old-signature",
        },
    }
    caller_hashed_fields = ["integrity", "attribution.signature", "intent"]

    # Act
    result, warnings = build_integrity_object(
        payload, hashed_fields=caller_hashed_fields
    )

    # Assert
    assert result is not None
    actual_fields = result["hashed_fields"]
    assert "integrity" not in actual_fields, (
        f"'integrity' 가 hashed_fields 에 남아 있으면 안 됨: {actual_fields}"
    )
    assert "attribution.signature" not in actual_fields, (
        f"'attribution.signature' 가 hashed_fields 에 남아 있으면 안 됨: {actual_fields}"
    )
    assert "intent" in actual_fields, (
        f"'intent' 는 hashed_fields 에 있어야 함: {actual_fields}"
    )


def test_build_integrity_anti_recursion_changing_integrity_does_not_change_hash():
    """anti-recursion 불변식: payload["integrity"] 값을 바꿔도 content_hash 가 달라지지 않아야 한다.

    Arrange: base payload + 명시적 hashed_fields (intent 포함)
    Act: build_integrity_object 두 번 — 두 번째는 payload["integrity"] 변경
    Assert: 두 content_hash 동일
    """
    # Arrange
    base_payload: dict = {
        "intent": {"goal": "ablation"},
        "integrity": {"content_hash": "sha256:old_value", "hashed_fields": ["x"]},
        "config": {"lr": 0.001},
    }
    # 'integrity' 는 제외되므로 변경해도 hash 불변.
    hashed_fields = ["integrity", "intent", "config"]

    # Act — 두 번 호출, 두 번째는 integrity 내용 변경
    result1, _ = build_integrity_object(base_payload, hashed_fields=hashed_fields)

    modified_payload = dict(base_payload)
    modified_payload["integrity"] = {"content_hash": "sha256:TOTALLY_DIFFERENT", "hashed_fields": ["y", "z"]}
    result2, _ = build_integrity_object(modified_payload, hashed_fields=hashed_fields)

    # Assert
    assert result1 is not None
    assert result2 is not None
    assert result1["content_hash"] == result2["content_hash"], (
        f"'integrity' 값 변경이 content_hash 에 영향을 줬음 (anti-recursion 위반):\n"
        f"  hash1: {result1['content_hash']!r}\n"
        f"  hash2: {result2['content_hash']!r}"
    )


def test_build_integrity_anti_recursion_changing_attribution_signature_does_not_change_hash():
    """anti-recursion 불변식: payload["attribution"]["signature"] 변경이 content_hash 에 영향을 주지 않아야 한다.

    Arrange: attribution.signature 포함 payload
    Act: attribution.signature 변경 후 두 번 빌드
    Assert: 두 content_hash 동일
    """
    # Arrange
    def make_payload(sig: str) -> dict:
        return {
            "intent": {"goal": "exploration"},
            "attribution": {
                "author": {"kind": "human", "id": "carol"},
                "signature": sig,
            },
            "config": {"seed": 7},
        }

    hashed_fields = ["attribution.signature", "intent", "config"]

    # Act
    result1, _ = build_integrity_object(make_payload("sig-v1"), hashed_fields=hashed_fields)
    result2, _ = build_integrity_object(make_payload("sig-v2-completely-different"), hashed_fields=hashed_fields)

    # Assert
    assert result1 is not None
    assert result2 is not None
    assert result1["content_hash"] == result2["content_hash"], (
        f"'attribution.signature' 변경이 content_hash 에 영향을 줬음 (anti-recursion 위반):\n"
        f"  hash1: {result1['content_hash']!r}\n"
        f"  hash2: {result2['content_hash']!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# build_integrity_object — hashed_fields 는 dotted leaf-path 목록
# ─────────────────────────────────────────────────────────────────────────────

def test_build_integrity_hashed_fields_are_dotted_leaf_paths():
    """hashed_fields 결과 값은 dotted leaf-path 목록이고, None 이면 기본값을 사용한다.

    Arrange: hashed_fields=None (기본값 사용)
    Act: build_integrity_object(payload)
    Assert: hashed_fields 가 비어 있지 않고, 각 항목이 문자열이며 "integrity" 미포함
    """
    # Arrange
    payload: dict = {"config": {"lr": 0.01}, "metrics": {"loss": 0.5}}

    # Act
    result, warnings = build_integrity_object(payload)

    # Assert
    assert result is not None
    fields = result["hashed_fields"]
    assert isinstance(fields, list), f"hashed_fields 가 list 여야 함: {type(fields)}"
    assert len(fields) > 0, "hashed_fields 가 빈 목록"
    for f in fields:
        assert isinstance(f, str), f"hashed_fields 항목이 문자열 아님: {f!r}"
    assert "integrity" not in fields, "'integrity' 는 기본 hashed_fields 에 없어야 함"
    assert "attribution.signature" not in fields, (
        "'attribution.signature' 는 기본 hashed_fields 에 없어야 함"
    )


# ─────────────────────────────────────────────────────────────────────────────
# build_integrity_object — 계산 실패 시 INTEGRITY_HASH_UNCOMPUTABLE 경고
# ─────────────────────────────────────────────────────────────────────────────

def test_build_integrity_uncomputable_emits_warning_no_exception(monkeypatch):
    """json 직렬화 실패 시 INTEGRITY_HASH_UNCOMPUTABLE 경고를 발생시키고 None 반환, 예외 없음.

    monkeypatch 로 json.dumps 를 강제로 예외 발생시켜 실패 경로를 테스트한다.

    Arrange: json.dumps 를 monkeypatch 로 RuntimeError 발생하도록 교체
    Act: build_integrity_object(payload)
    Assert: result is None, INTEGRITY_HASH_UNCOMPUTABLE 경고 포함
    """
    import pcq.contract as _contract_module

    # Arrange — json.dumps 를 에러 발생으로 교체
    def _failing_dumps(*args, **kwargs):
        raise RuntimeError("강제 직렬화 실패")

    monkeypatch.setattr(_contract_module, "json", type("FakeJson", (), {
        "dumps": staticmethod(_failing_dumps),
        "loads": json.loads,
    })())

    payload: dict = {"config": {"lr": 0.001}}

    # Act
    result, warnings = build_integrity_object(payload)

    # Assert
    assert result is None, f"계산 실패 시 None 반환 기대, 실제: {result!r}"
    codes = [w.get("code") for w in warnings]
    assert "INTEGRITY_HASH_UNCOMPUTABLE" in codes, (
        f"INTEGRITY_HASH_UNCOMPUTABLE 경고 없음: {warnings}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# R6 backward-compat: 1.x 경로 — intent/integrity 미전달 → contract_version 없음
# ─────────────────────────────────────────────────────────────────────────────

def test_R6_1x_path_no_intent_no_integrity_no_contract_version(
    tmp_path: Path, monkeypatch
):
    """R6: intent/integrity 없이 save_all → run_record 에 contract_version/intent/integrity 없음.

    1.x 호환 레코드 조건: contract_version, intent, integrity 모두 부재.

    Arrange: CQ_CONFIG_JSON 설정, intent 인자 없이 save_all 호출
    Act: run_record.json 읽기
    Assert: 'contract_version', 'intent', 'integrity' 키 없음
    """
    # Arrange
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    history = [{"epoch": 0, "eval_loss": 0.6}]

    # Act
    pcq.save_all(history=history)
    rr = json.loads((tmp_path / "run_record.json").read_text())

    # Assert — 1.x 필드들 부재 확인
    assert "contract_version" not in rr, (
        f"1.x 경로에서 contract_version 이 나타남: {rr.get('contract_version')!r}"
    )
    assert "intent" not in rr, (
        f"1.x 경로에서 intent 가 나타남: {rr.get('intent')!r}"
    )
    assert "integrity" not in rr, (
        f"1.x 경로에서 integrity 가 나타남: {rr.get('integrity')!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# R5 + R1 + R2: 2.x 경로 — intent 전달 → contract_version=="2.0", intent, integrity 존재
# ─────────────────────────────────────────────────────────────────────────────

def test_R5_2x_path_with_intent_has_contract_version_intent_integrity(
    tmp_path: Path, monkeypatch
):
    """R5/R1/R2: intent 전달 시 run_record 에 contract_version='2.0', intent, integrity 포함.

    Arrange: valid goal + expected_baseline + tolerance 포함 intent 빌드 후 save_all 전달
    Act: run_record.json 읽기
    Assert: contract_version=="2.0", intent 존재, integrity 존재 (content_hash 형식 확인)
    """
    # Arrange
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    history = [{"epoch": 0, "eval_loss": 0.33}]

    # Act — save_all 은 intent_goal/expected_baseline/tolerance 인자로 받아 내부 빌드
    pcq.save_all(
        history=history,
        intent_goal="baseline_reproduction",
        intent_expected_baseline={"metric": "eval_loss", "value": 0.35},
        intent_tolerance={"direction": "lower_is_better", "margin": 0.05},
    )
    rr = json.loads((tmp_path / "run_record.json").read_text())

    # Assert — 2.x 필드 모두 존재
    assert rr.get("contract_version") == "2.0", (
        f"contract_version 이 '2.0' 이어야 함: {rr.get('contract_version')!r}"
    )
    assert "intent" in rr, "intent 키가 run_record 에 없음"
    assert rr["intent"]["goal"] == "baseline_reproduction"

    assert "integrity" in rr, "integrity 키가 run_record 에 없음"
    integrity = rr["integrity"]
    assert "content_hash" in integrity
    assert integrity["content_hash"].startswith("sha256:"), (
        f"integrity.content_hash 형식 오류: {integrity['content_hash']!r}"
    )
    assert "hashed_fields" in integrity
    assert isinstance(integrity["hashed_fields"], list)


def test_R5_2x_path_intent_field_reflected_in_run_record(
    tmp_path: Path, monkeypatch
):
    """2.x 경로: save_all 로 전달된 intent 가 run_record 의 intent 필드에 정확히 반영된다.

    Arrange: goal="sota_challenge" intent 빌드
    Act: save_all → run_record 읽기
    Assert: run_record["intent"]["goal"] == "sota_challenge"
    """
    # Arrange
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))

    # Act — save_all 은 intent_goal 인자로 직접 전달
    pcq.save_all(
        history=[{"epoch": 0, "eval_loss": 0.4}],
        intent_goal="sota_challenge",
    )
    rr = json.loads((tmp_path / "run_record.json").read_text())

    # Assert
    assert rr["intent"]["goal"] == "sota_challenge"
    assert rr["contract_version"] == "2.0"
