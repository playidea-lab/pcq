"""Recipe acceptance framework tests + 모든 등록 recipe 일괄 검증.

SPEC §"Recipe Acceptance Criteria" 7항목을 framework 가 자동 검증.
"""
from __future__ import annotations

import pytest

import pcq
from pcq.testing import AcceptanceReport, list_failures, recipe_smoke


@pytest.mark.parametrize("preset", pcq.Trainer.list_presets())
def test_acceptance_smoke_for_each_preset(preset, tmp_path):
    """모든 등록된 recipe 가 7항목 acceptance 통과 (또는 명시적 skip)."""
    report = recipe_smoke(preset, tmp_path=tmp_path)
    if report.skipped_reason:
        pytest.skip(report.skipped_reason)
    assert report.passed, f"{preset} 실패:\n{report}"


def test_acceptance_returns_full_report(tmp_path):
    """fake_smoke 는 7개 check 모두 통과해야 한다."""
    report = recipe_smoke("vision/fake_smoke", tmp_path=tmp_path)
    assert isinstance(report, AcceptanceReport)
    assert report.passed
    expected_checks = {
        "import",
        "inspect",
        "smoke_path",
        "fit_smoke",
        "declared_metrics",
        "artifacts",
        "resume",
    }
    assert expected_checks.issubset(report.checks.keys())


def test_acceptance_report_str_renders_all_checks(tmp_path):
    """__str__ 에 PASS/FAIL 마커와 모든 check 가 보여야 한다."""
    report = recipe_smoke("vision/fake_smoke", tmp_path=tmp_path)
    text = str(report)
    assert "vision/fake_smoke" in text
    assert "PASS" in text
    for check in ("import", "fit_smoke", "resume"):
        assert check in text


def test_acceptance_unknown_recipe_fails():
    """존재하지 않는 recipe → import check 실패."""
    report = recipe_smoke("vision/does_not_exist")
    assert not report.passed
    assert not report.checks["import"]["pass"]


def test_acceptance_detects_smoke_overrides_metadata():
    """smoke_safe=False 인 recipe 는 smoke_overrides 가 명시되어 있어야 한다."""
    meta = pcq.recipe_meta("vision/cifar10_smallcnn_baseline")
    if "import_error" in meta:
        pytest.skip(f"recipe import failed: {meta['import_error']}")
    assert meta["smoke_safe"] is False
    assert meta["has_smoke_overrides"] is True


def test_list_failures_returns_no_failures():
    """모든 recipe 통과 (또는 skip) 시 list_failures() 가 빈 리스트."""
    failures = list_failures()
    assert failures == [], (
        f"실패한 recipe: {[f.recipe for f in failures]}\n"
        + "\n".join(str(f) for f in failures)
    )


def test_acceptance_skip_when_optional_dep_missing(tmp_path):
    """torchvision 미설치 환경에서 cifar10_resnet18 은 skip 되어야 한다."""
    pytest.importorskip("torch")  # torch 는 필수
    try:
        import torchvision  # noqa: F401
    except ImportError:
        # torchvision 없는 환경 — skipped_reason 설정 + passed=True 기대
        report = recipe_smoke("vision/cifar10_resnet18", tmp_path=tmp_path)
        assert report.skipped_reason is not None
        assert "torchvision" in report.skipped_reason
    else:
        pytest.skip("torchvision installed — this assertion only meaningful without it")
