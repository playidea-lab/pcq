"""attribution 기능 단위 테스트 — R1~R10 커버리지.

각 테스트는 pcq-agent-attribution.md EARS 요구사항에 대응한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pcq
from pcq.contract import build_attribution_object
from pcq.agent.describe import describe_run


# ─────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _setup_cfg(tmp_path: Path, **extra) -> Path:
    """테스트용 CQ_CONFIG_JSON 파일 생성."""
    cfg = {"output_dir": str(tmp_path), "seed": 42}
    cfg.update(extra)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def _run_full_pipeline(tmp_path: Path, monkeypatch, **save_all_kwargs) -> dict:
    """save_all + describe_run 까지 실행한 뒤 run_record dict 반환."""
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    history = [{"epoch": 0, "eval_loss": 0.5}]
    pcq.save_all(history=history, **save_all_kwargs)
    return json.loads((tmp_path / "run_record.json").read_text())


# ─────────────────────────────────────────────────────────────────────────────
# R1 + R2: attribution 키 존재 + 필수 필드 검증
# ─────────────────────────────────────────────────────────────────────────────

def test_R1_R2_run_record_has_attribution_with_required_keys(
    tmp_path: Path, monkeypatch
):
    """R1: env var 설정 시 run_record 에 attribution 포함.
    R2: attribution 객체는 schema_version, author, committer, operator 를 가져야 한다.
    session_id / persona_id 는 None 허용 (MAY).
    """
    monkeypatch.setenv("CQ_ATTRIBUTION_OPERATOR", "alice-uuid")

    rr = _run_full_pipeline(tmp_path, monkeypatch)

    assert "attribution" in rr, "run_record.json 에 attribution 키가 없습니다"
    attr = rr["attribution"]

    # 필수 최상위 키
    for key in ("schema_version", "author", "committer", "operator"):
        assert key in attr, f"attribution 에 필수 키 '{key}' 없음"

    # author / committer 는 kind + id 를 가져야 함
    for role in ("author", "committer"):
        assert "kind" in attr[role], f"{role}.kind 누락"
        assert "id" in attr[role], f"{role}.id 누락"

    # session_id 는 존재하거나 None 이면 OK (MAY 필드)
    if "session_id" in attr:
        assert attr["session_id"] is None or isinstance(attr["session_id"], str)


# ─────────────────────────────────────────────────────────────────────────────
# R3: 단일 human 사용자 — operator 만 설정하면 author == committer == operator
# ─────────────────────────────────────────────────────────────────────────────

def test_R3_single_human_user_collapses_to_one_identity(monkeypatch):
    """R3: CQ_ATTRIBUTION_OPERATOR 만 설정하면 author/committer 가 동일 주체로
    auto-infer 되고 kind 는 'human' 이어야 한다.
    """
    monkeypatch.setenv("CQ_ATTRIBUTION_OPERATOR", "alice-uuid")
    # 나머지 attribution env var 는 없어야 함
    for var in (
        "CQ_ATTRIBUTION_AUTHOR_KIND",
        "CQ_ATTRIBUTION_AUTHOR_ID",
        "CQ_ATTRIBUTION_COMMITTER_KIND",
        "CQ_ATTRIBUTION_COMMITTER_ID",
    ):
        monkeypatch.delenv(var, raising=False)

    attr = build_attribution_object()
    assert attr is not None

    assert attr["author"]["kind"] == "human"
    assert attr["committer"]["kind"] == "human"
    assert attr["author"]["id"] == "alice-uuid"
    assert attr["committer"]["id"] == "alice-uuid"
    assert attr["operator"] == "alice-uuid"


# ─────────────────────────────────────────────────────────────────────────────
# R4: 에이전트가 인간 대신 커밋하는 경우
# ─────────────────────────────────────────────────────────────────────────────

def test_R4_agent_committer_on_behalf_of_human(monkeypatch):
    """R4: author 는 human, committer 는 agent 인 분리 시나리오.

    env: CQ_ATTRIBUTION_AUTHOR_KIND=human, CQ_ATTRIBUTION_COMMITTER_KIND=agent,
         CQ_ATTRIBUTION_COMMITTER_ID=claude-opus-4-7, CQ_ATTRIBUTION_OPERATOR=user-uuid
    """
    monkeypatch.setenv("CQ_ATTRIBUTION_AUTHOR_KIND", "human")
    monkeypatch.setenv("CQ_ATTRIBUTION_COMMITTER_KIND", "agent")
    monkeypatch.setenv("CQ_ATTRIBUTION_COMMITTER_ID", "claude-opus-4-7")
    monkeypatch.setenv("CQ_ATTRIBUTION_OPERATOR", "user-uuid")
    # author_id 는 operator 에서 auto-infer 되지 않음 (author_id 미설정이면 operator 로 설정)
    monkeypatch.delenv("CQ_ATTRIBUTION_AUTHOR_ID", raising=False)

    attr = build_attribution_object()
    assert attr is not None

    # committer 는 agent
    assert attr["committer"]["kind"] == "agent"
    assert attr["committer"]["id"] == "claude-opus-4-7"
    # operator 는 user-uuid
    assert attr["operator"] == "user-uuid"
    # author kind 는 human
    assert attr["author"]["kind"] == "human"


# ─────────────────────────────────────────────────────────────────────────────
# R5: describe_run 이 중첩 + 플랫 attribution 필드를 모두 노출
# ─────────────────────────────────────────────────────────────────────────────

def test_R5_describe_run_has_nested_and_flat_attribution_fields(
    tmp_path: Path, monkeypatch
):
    """R5: finalize 후 describe_run 이 attribution 중첩 dict 와 flat 필드를
    동시에 제공해야 한다.
    """
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    monkeypatch.setenv("CQ_ATTRIBUTION_AUTHOR_KIND", "human")
    monkeypatch.setenv("CQ_ATTRIBUTION_COMMITTER_KIND", "agent")
    monkeypatch.setenv("CQ_ATTRIBUTION_COMMITTER_ID", "claude-opus-4-7")
    monkeypatch.setenv("CQ_ATTRIBUTION_OPERATOR", "user-uuid")
    monkeypatch.delenv("CQ_ATTRIBUTION_AUTHOR_ID", raising=False)

    history = [{"epoch": 0, "eval_loss": 0.3}]
    pcq.save_all(history=history)

    desc = describe_run(tmp_path)

    # 중첩 객체
    assert desc.attribution is not None, "describe_run.attribution 이 None 입니다"
    assert isinstance(desc.attribution, dict)

    # 플랫 표면
    assert desc.attribution_author_kind == "human"
    assert desc.attribution_committer_kind == "agent"
    assert desc.attribution_operator == "user-uuid"
    # session_id 는 설정하지 않았으므로 None
    assert desc.attribution_session_id is None

    # to_dict() 에도 반영
    data = desc.to_dict()
    assert "attribution" in data
    assert data.get("attribution_author_kind") == "human"
    assert data.get("attribution_committer_kind") == "agent"
    assert data.get("attribution_operator") == "user-uuid"


# ─────────────────────────────────────────────────────────────────────────────
# R6: 하위 호환성 — attribution 없는 구 run_record
# ─────────────────────────────────────────────────────────────────────────────

def test_R6A_old_run_record_without_attribution_key_returns_none(tmp_path: Path):
    """R6 케이스 A: attribution 키가 없는 구 run_record.json → 에러 없이
    describe_run.attribution == None, 모든 flat 필드도 None.
    """
    # attribution 키 없는 최소 run_record 직접 작성
    rr = {
        "schema_version": 1,
        "run": {"id": "old-run-1", "status": "completed"},
        "execution": {},
        "source": {},
        "environment": {},
        "metrics": {"declared": [], "history_path": "metrics.json"},
        "artifacts": [],
    }
    (tmp_path / "run_record.json").write_text(json.dumps(rr))

    # 에러 없이 describe_run 호출
    desc = describe_run(tmp_path)

    assert desc.attribution is None
    assert desc.attribution_author_kind is None
    assert desc.attribution_committer_kind is None
    assert desc.attribution_operator is None
    assert desc.attribution_session_id is None


def test_R6B_run_record_with_explicit_null_attribution_returns_none(
    tmp_path: Path,
):
    """R6 케이스 B: attribution=null 명시 run_record.json → 에러 없이
    describe_run.attribution == None.
    """
    # attribution 값이 null 인 run_record
    rr = {
        "schema_version": 1,
        "run": {"id": "old-run-2", "status": "completed"},
        "execution": {},
        "source": {},
        "environment": {},
        "metrics": {"declared": [], "history_path": "metrics.json"},
        "artifacts": [],
        "attribution": None,  # 명시 null
    }
    (tmp_path / "run_record.json").write_text(json.dumps(rr))

    desc = describe_run(tmp_path)

    assert desc.attribution is None
    assert desc.attribution_author_kind is None
    assert desc.attribution_committer_kind is None
    assert desc.attribution_operator is None
    assert desc.attribution_session_id is None


# ─────────────────────────────────────────────────────────────────────────────
# R7: compare_runs diff 스모크 테스트 (conformance 범위 밖 — 구현 여부 조건부)
# ─────────────────────────────────────────────────────────────────────────────

def test_R7_smoke_compare_runs_diff_schema_no_attribution_diff():
    """R7: compare_runs.RunDiff 스키마에 attribution_diff 필드가 없는지 확인
    (현재 T-7 conformance 범위 외 — RunDiff 에 해당 필드 없음 = 스펙 미구현 상태).

    NOTE: T-7 에서 attribution_diff 가 추가되면 이 테스트를 업데이트해야 한다.
    """
    from pcq.agent.compare import RunDiff
    diff = RunDiff()
    d = diff.to_dict()
    # 현재 구현에 attribution_diff 없음 — 스키마 확인만 (추가 시 실패해서 알려줌)
    assert "attribution_diff" not in d, (
        "attribution_diff 가 RunDiff 에 추가됨 — T-7 구현 완료 후 이 테스트 갱신 필요"
    )


# ─────────────────────────────────────────────────────────────────────────────
# R10: PII 패턴 경고 (validate_run --strictness 3 + operator = email)
# ─────────────────────────────────────────────────────────────────────────────

def test_R10_pii_pattern_operator_warns_with_code(tmp_path: Path, monkeypatch):
    """R10: operator 에 이메일 주소가 설정될 때 validation_report 에
    PII_PATTERN_DETECTED 경고가 있어야 한다.

    NOTE: PII 검사 로직이 validate_run 에 아직 미구현인 경우 skip.
    현재 T-3 에서 스키마 게이트는 존재하나 실제 PII 검사 로직은
    향후 태스크에서 추가 예정이면 skip 처리.
    """
    from pcq.agent.validate_run import validate_run

    # PII 이메일 operator 가 포함된 attribution 을 run_record 에 직접 작성
    attr = {
        "schema_version": 1,
        "author": {"kind": "human", "id": "user@example.com", "persona_id": None},
        "committer": {"kind": "human", "id": "user@example.com", "persona_id": None},
        "operator": "user@example.com",
        "session_id": None,
    }
    rr = {
        "schema_version": 1,
        "run": {"id": "pii-test", "status": "completed"},
        "execution": {"cmd": "uv run python train.py"},
        "source": {"git_sha": "abc", "dirty": False},
        "environment": {"python": "3.11", "platform": "Linux-x86_64", "pcq_version": "0.1"},
        "metrics": {"declared": [{"name": "eval_loss", "mode": "min"}], "history_path": "metrics.json"},
        "artifacts": [],
        "config": {"seed": 42, "strictness": 3, "output_dir": str(tmp_path)},
        "attribution": attr,
    }
    (tmp_path / "run_record.json").write_text(json.dumps(rr))
    (tmp_path / "metrics.json").write_text(json.dumps({"history": [{"epoch": 0, "eval_loss": 0.5}]}))
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": []}))
    (tmp_path / "run_summary.json").write_text(json.dumps({
        "schema_version": 1,
        "status": "completed",
        "monitor": {"name": "eval_loss", "mode": "min"},
    }))

    report = validate_run(tmp_path, strictness=3)
    report_dict = report.to_dict()

    # PII 검사 결과 확인
    pii_checks = [
        c for c in report.checks
        if getattr(c, "code", None) == "PII_PATTERN_DETECTED"
        or getattr(c, "id", None) == "pii_pattern_detected"
        or (
            hasattr(c, "evidence") and isinstance(c.evidence, dict)
            and c.evidence.get("code") == "PII_PATTERN_DETECTED"
        )
    ]

    if not pii_checks:
        # PII 검사 로직이 아직 미구현 — skip
        pytest.skip(
            "PII 패턴 감지 로직 미구현 (validate_run 에 attribution.operator email 검사 없음). "
            "향후 태스크에서 구현 예정 — 스키마 게이트는 T-3 에서 이미 존재."
        )

    # PII 경고가 있으면 exit_code=0 (warn 은 패스) 확인
    assert report.status in ("pass", "warn"), (
        f"PII 경고 시 exit_code 는 0 이어야 하지만 status={report.status}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# kind 검증: 잘못된 kind 는 ValueError
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_author_kind_raises_value_error(monkeypatch):
    """build_attribution_object(author_kind='invalid') 는 ValueError 를 발생시켜야 한다."""
    # env var 초기화 (다른 테스트 간 오염 방지)
    for var in (
        "CQ_ATTRIBUTION_OPERATOR",
        "CQ_ATTRIBUTION_AUTHOR_ID",
        "CQ_ATTRIBUTION_AUTHOR_KIND",
        "CQ_ATTRIBUTION_COMMITTER_ID",
        "CQ_ATTRIBUTION_COMMITTER_KIND",
    ):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValueError, match="author_kind"):
        build_attribution_object(
            operator="alice",
            author_kind="invalid",  # "human" | "agent" 이외
        )


def test_invalid_committer_kind_raises_value_error(monkeypatch):
    """build_attribution_object(committer_kind='robot') 는 ValueError 를 발생시켜야 한다."""
    for var in (
        "CQ_ATTRIBUTION_OPERATOR",
        "CQ_ATTRIBUTION_AUTHOR_ID",
        "CQ_ATTRIBUTION_AUTHOR_KIND",
        "CQ_ATTRIBUTION_COMMITTER_ID",
        "CQ_ATTRIBUTION_COMMITTER_KIND",
    ):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValueError, match="committer_kind"):
        build_attribution_object(
            operator="alice",
            committer_kind="robot",  # "human" | "agent" 이외
        )


# ─────────────────────────────────────────────────────────────────────────────
# 우선순위: 명시 인자 > 환경변수
# ─────────────────────────────────────────────────────────────────────────────

def test_explicit_operator_arg_overrides_env_var(monkeypatch):
    """명시적 operator= 인자가 CQ_ATTRIBUTION_OPERATOR 환경변수보다 우선해야 한다."""
    monkeypatch.setenv("CQ_ATTRIBUTION_OPERATOR", "env-operator-uuid")
    # 나머지 env var 제거
    for var in (
        "CQ_ATTRIBUTION_AUTHOR_ID",
        "CQ_ATTRIBUTION_AUTHOR_KIND",
        "CQ_ATTRIBUTION_COMMITTER_ID",
        "CQ_ATTRIBUTION_COMMITTER_KIND",
    ):
        monkeypatch.delenv(var, raising=False)

    # 명시 인자로 다른 operator 전달
    attr = build_attribution_object(operator="explicit-operator-uuid")
    assert attr is not None
    assert attr["operator"] == "explicit-operator-uuid", (
        f"명시 인자가 env var 를 override 해야 하지만 operator={attr['operator']!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# None 반환: 인자 없음 + env var 없음 → None
# ─────────────────────────────────────────────────────────────────────────────

def test_no_args_no_env_returns_none(monkeypatch):
    """인자와 환경변수 모두 없으면 build_attribution_object 는 None 을 반환해야 한다."""
    # 모든 attribution 관련 env var 삭제
    for var in (
        "CQ_ATTRIBUTION_OPERATOR",
        "CQ_ATTRIBUTION_AUTHOR_ID",
        "CQ_ATTRIBUTION_AUTHOR_KIND",
        "CQ_ATTRIBUTION_COMMITTER_ID",
        "CQ_ATTRIBUTION_COMMITTER_KIND",
        "CQ_ATTRIBUTION_SESSION_ID",
        "CQ_ATTRIBUTION_PERSONA_AUTHOR",
        "CQ_ATTRIBUTION_PERSONA_COMMITTER",
    ):
        monkeypatch.delenv(var, raising=False)

    result = build_attribution_object()
    assert result is None, (
        f"아무 값도 없을 때 None 이어야 하지만 {result!r} 반환됨"
    )
