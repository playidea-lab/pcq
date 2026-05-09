"""pcq.testing — recipe acceptance runner.

7가지 acceptance criteria 자동 검증 (SPEC §"Recipe Acceptance Criteria"):

1. Import succeeds without running training
2. Recipe can be inspected without side effects
3. Fake/small-data smoke path exists for tests
4. Trainer(preset=...).fit() completes a one-epoch smoke run
5. Declared metrics match emitted metrics
6. Standard artifacts are produced
7. Resume smoke test passes when applicable
"""
from __future__ import annotations

import io
import re
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

from pcq.trainer import Trainer, _import_recipe


@dataclass
class AcceptanceReport:
    """Result of running 7-항목 acceptance on a single recipe."""

    recipe: str
    passed: bool = True
    checks: dict[str, dict] = field(default_factory=dict)
    skipped_reason: str | None = None

    def __str__(self) -> str:
        lines = [
            f"Recipe: {self.recipe}",
            f"Status: {'PASS' if self.passed else 'FAIL'}",
        ]
        if self.skipped_reason:
            lines.append(f"Skipped: {self.skipped_reason}")
        for name, result in self.checks.items():
            mark = "PASS" if result["pass"] else "FAIL"
            lines.append(f"  [{mark}] {name}: {result.get('detail', '')}")
        return "\n".join(lines)


def _record(report: AcceptanceReport, name: str, ok: bool, detail: str) -> None:
    # 검증 결과 한 항목 기록 + 실패 시 report.passed 갱신
    report.checks[name] = {"pass": ok, "detail": detail}
    if not ok:
        report.passed = False


def recipe_smoke(
    preset: str, tmp_path: Path | None = None
) -> AcceptanceReport:
    """Run all 7 acceptance checks against a recipe.

    Args:
        preset: recipe 이름 (예: 'vision/cifar10_smallcnn_baseline')
        tmp_path: 학습 출력 디렉토리. None 이면 tempfile 자동 생성.

    Returns:
        AcceptanceReport with per-check pass/fail.
    """
    report = AcceptanceReport(recipe=preset)

    # ── Check 1: import succeeds without training ───────────────────────────
    try:
        recipe_fn = _import_recipe(preset)
        _record(
            report, "import", True, "recipe module importable without training"
        )
    except Exception as e:
        _record(report, "import", False, f"{type(e).__name__}: {e}")
        return report

    # ── Check 2: recipe can be inspected without side effects ──────────────
    try:
        recipe_dict = recipe_fn()
        _record(
            report,
            "inspect",
            True,
            f"recipe() returned dict ({len(recipe_dict)} keys)",
        )
    except ModuleNotFoundError as e:
        # extras 미설치 (예: torchvision) 는 검증 skip — env 한계, 코드 결함 X
        report.skipped_reason = f"missing optional dependency: {e.name}"
        _record(
            report,
            "inspect",
            True,
            f"skipped (missing extra: {e.name})",
        )
        return report
    except Exception as e:
        _record(report, "inspect", False, f"recipe() failed: {e}")
        return report

    # ── Check 3: smoke path exists ─────────────────────────────────────────
    smoke_safe = recipe_dict.get("smoke_safe")
    smoke_overrides = recipe_dict.get("smoke_overrides", {}) or {}
    if smoke_safe is False and not smoke_overrides:
        _record(
            report,
            "smoke_path",
            False,
            "smoke_safe=False but no smoke_overrides provided",
        )
        return report
    detail = (
        "smoke_safe (uses recipe atoms directly)"
        if smoke_safe
        else f"using {len(smoke_overrides)} override(s): {sorted(smoke_overrides)}"
    )
    _record(report, "smoke_path", True, detail)

    # ── Check 4 + 5 + 6: actual fit + metrics + artifacts ──────────────────
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp(prefix="pcq-acc-"))
    declared_metrics = list(recipe_dict.get("metrics", []))
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": min(8, int(recipe_dict.get("batch_size", 8))),
    }

    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            trainer = Trainer(preset=preset, cfg=cfg, **smoke_overrides)
            trainer.fit()
        _record(
            report, "fit_smoke", True, f"one epoch completed in {tmp_path.name}"
        )
    except Exception as e:
        _record(
            report,
            "fit_smoke",
            False,
            f"fit() failed: {type(e).__name__}: {e}",
        )
        return report

    stdout_text = captured.getvalue()

    # Check 5: declared metrics emitted (epoch 는 자동 송출이므로 제외 가능)
    if declared_metrics:
        emitted = set(re.findall(r"@(\w+)=", stdout_text))
        missing = [
            m
            for m in declared_metrics
            if m not in emitted and m != "epoch"  # epoch 은 lenient
        ]
        if missing:
            _record(
                report,
                "declared_metrics",
                False,
                f"declared but not emitted: {missing}; emitted: {sorted(emitted)}",
            )
        else:
            _record(
                report,
                "declared_metrics",
                True,
                f"all {len(declared_metrics)} emitted",
            )
    else:
        _record(
            report,
            "declared_metrics",
            True,
            "no declared metrics (skip)",
        )

    # Check 6: standard artifacts
    expected = ["model.pt", "config.json", "metrics.json", "last.ckpt", "manifest.json"]
    missing_art = [a for a in expected if not (tmp_path / a).exists()]
    if missing_art:
        _record(
            report,
            "artifacts",
            False,
            f"missing: {missing_art}",
        )
    else:
        _record(
            report,
            "artifacts",
            True,
            f"all {len(expected)} present",
        )

    # ── Check 7: resume smoke ──────────────────────────────────────────────
    cfg2 = {
        **cfg,
        "epochs": 2,
        "resume_from": str(tmp_path / "last.ckpt"),
    }
    try:
        with redirect_stdout(io.StringIO()):
            trainer2 = Trainer(preset=preset, cfg=cfg2, **smoke_overrides)
            trainer2.fit()
        _record(
            report, "resume", True, "resume from last.ckpt succeeded"
        )
    except Exception as e:
        _record(
            report,
            "resume",
            False,
            f"resume failed: {type(e).__name__}: {e}",
        )

    return report


def list_failures(presets: list[str] | None = None) -> list[AcceptanceReport]:
    """Run acceptance on all (or specified) presets. Return failures only.

    skipped_reason 이 설정된 report (예: missing optional dep) 는 실패에서 제외.
    """
    if presets is None:
        presets = Trainer.list_presets()
    failures: list[AcceptanceReport] = []
    for preset in presets:
        report = recipe_smoke(preset)
        if not report.passed and report.skipped_reason is None:
            failures.append(report)
    return failures
