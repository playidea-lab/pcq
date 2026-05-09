"""tests/test_validate_run_rescan.py — Fix 3 (G1-2).

manifest stale lock-in. output_dir 을 reuse 했더니 manifest 안에 옛 file (지금은
삭제된) 잠겨 있어 validate strictness>=2 fail. --rescan-manifest 로 누락 파일
무시하도록 변형 가능. 미사용 시에는 기존 fail 동작 + suggested_fix 포함.
"""
from __future__ import annotations

import json
from pathlib import Path

from pcq.agent.validate_run import validate_run


def _write_manifest(out: Path, files: list[dict]) -> None:
    """schema v2 manifest.json 작성."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "files": files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def test_validate_run_default_fails_on_missing_with_clear_fix(tmp_path: Path):
    """기본 validate-run: manifest 의 파일이 disk 에 없으면 fail.

    suggested_fix 는 --rescan-manifest 옵션을 명시해야 한다.
    """
    out = tmp_path / "out"
    real = out / "real.txt"
    out.mkdir()
    real.write_text("hello", encoding="utf-8")

    _write_manifest(
        out,
        [
            {"path": "real.txt", "sha256": _sha256(real)},
            {"path": "ghost.bin", "sha256": "0" * 64},
        ],
    )

    report = validate_run(out, strictness=0)
    # manifest_evidence fail 이어야 한다 (ghost.bin 누락).
    me = next((c for c in report.checks if c.id == "manifest_evidence"), None)
    assert me is not None, "manifest_evidence check not found"
    assert me.status == "fail"
    fix = (me.suggested_fix or "")
    assert "rescan-manifest" in fix or "rescan_manifest" in fix


def test_validate_run_rescan_manifest_ignores_missing(tmp_path: Path):
    """rescan_manifest=True 시 manifest 의 누락 파일은 무시 (output_dir 재스캔)."""
    out = tmp_path / "out"
    real = out / "real.txt"
    out.mkdir()
    real.write_text("hello", encoding="utf-8")

    _write_manifest(
        out,
        [
            {"path": "real.txt", "sha256": _sha256(real)},
            {"path": "ghost.bin", "sha256": "0" * 64},
        ],
    )

    report = validate_run(out, strictness=0, rescan_manifest=True)
    me = next((c for c in report.checks if c.id == "manifest_evidence"), None)
    assert me is not None
    assert me.status == "pass", (
        f"expected pass with rescan, got {me.status}: {me.detail}"
    )


def test_validate_run_rescan_manifest_still_catches_sha_mismatch(tmp_path: Path):
    """rescan 모드여도 disk 에 있는 file 의 sha256 mismatch 는 잡아야 한다."""
    out = tmp_path / "out"
    real = out / "real.txt"
    out.mkdir()
    real.write_text("hello", encoding="utf-8")

    _write_manifest(
        out,
        [
            {"path": "real.txt", "sha256": "f" * 64},  # wrong sha
        ],
    )

    report = validate_run(out, strictness=0, rescan_manifest=True)
    me = next((c for c in report.checks if c.id == "manifest_evidence"), None)
    assert me is not None
    assert me.status == "fail"
    assert "sha256" in (me.detail or "").lower()


def test_validate_run_cli_accepts_rescan_manifest_flag(
    tmp_path: Path, capsys
):
    """CLI: pcq validate-run --rescan-manifest 옵션 노출."""
    from pcq.cli import main as cli_main

    out = tmp_path / "out"
    real = out / "real.txt"
    out.mkdir()
    real.write_text("hello", encoding="utf-8")
    _write_manifest(
        out,
        [
            {"path": "real.txt", "sha256": _sha256(real)},
            {"path": "ghost.bin", "sha256": "0" * 64},
        ],
    )
    # metrics.json 도 만들어야 다른 gate 가 fail 하지 않음.
    (out / "metrics.json").write_text(
        json.dumps({"history": []}), encoding="utf-8"
    )
    (out / "run_summary.json").write_text(
        json.dumps({"best": {}, "last": {}}), encoding="utf-8"
    )

    # 기본 — fail expected
    rc_default = cli_main([
        "validate-run", str(out), "--strictness", "0", "--json"
    ])
    capsys.readouterr()  # drain

    # --rescan-manifest — manifest_evidence pass
    rc_rescan = cli_main([
        "validate-run", str(out), "--strictness", "0",
        "--rescan-manifest", "--json",
    ])
    out_rescan = capsys.readouterr().out
    payload = json.loads(out_rescan)
    me = next(
        (c for c in payload.get("checks", []) if c.get("id") == "manifest_evidence"),
        None,
    )
    assert me is not None
    assert me.get("status") == "pass", (
        f"expected pass with --rescan-manifest, got: {me}"
    )
    # 적어도 동작이 다르거나 (rescan rc <= default rc), 둘 다 0 이어도 OK.
    assert rc_default >= 0  # sanity
    assert rc_rescan == 0
