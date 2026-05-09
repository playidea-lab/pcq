"""smoke_atom contract verification (v1.12)."""
from __future__ import annotations

import pytest

from pcq.agent.smoke import smoke_atom


@pytest.mark.parametrize("kind,name", [
    ("model", "mlp"),
    ("model", "small_cnn"),
    ("model", "unet"),
    ("model", "text_classifier"),
    ("loss", "cross_entropy"),
    ("loss", "dice"),
    ("loss", "focal"),
    ("dataset", "fake"),
    ("dataset", "fake_seg"),
    ("dataset", "fake_text"),
    ("metric", "accuracy"),
    ("metric", "top_5_accuracy"),
    ("metric", "iou"),
    ("metric", "dice_score"),
    ("metric", "pixel_accuracy"),
    ("metric", "mse"),
    ("metric", "mae"),
    ("optim", "adamw"),
    ("sched", "cosine"),
])
def test_smoke_passes_for_builtin(kind, name):
    report = smoke_atom(kind, name)
    assert report.passed, (
        f"{kind}/{name}: {report.error or report.detail}"
    )


def test_smoke_unknown_kind():
    report = smoke_atom("bogus", "anything")
    assert not report.passed
    assert "unknown kind" in (report.error or "")


def test_smoke_unknown_atom_name():
    report = smoke_atom("model", "does_not_exist_v12")
    assert not report.passed
    assert "unknown" in (report.error or "").lower()


def test_smoke_report_to_dict():
    report = smoke_atom("model", "mlp")
    d = report.to_dict()
    assert d["schema_version"] == 1
    assert d["kind"] == "model"
    assert d["name"] == "mlp"
    assert d["passed"] is True
    assert "detail" in d
