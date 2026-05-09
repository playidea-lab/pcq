"""manifest schema v2 — sha256 + size_bytes + created_at (v1.14)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pcq


def _setup_cfg(tmp_path: Path, **extra) -> Path:
    cfg = {"output_dir": str(tmp_path), "seed": 42}
    cfg.update(extra)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def test_save_manifest_default_enrich(tmp_path, monkeypatch):
    """default cfg → schema v2 + sha256/size_bytes/created_at 모두 포함."""
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    (tmp_path / "model.pkl").write_bytes(b"fake-model-bytes")
    pcq.save_manifest()
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["schema_version"] == 2
    entry = next(f for f in m["files"] if f["path"] == "model.pkl")
    assert entry["sha256"] == hashlib.sha256(b"fake-model-bytes").hexdigest()
    assert entry["size_bytes"] == len(b"fake-model-bytes")
    assert "created_at" in entry


def test_save_manifest_enrich_false_falls_back_to_v1(tmp_path, monkeypatch):
    """명시적 enrich=False → schema v1 (path/kind only)."""
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    (tmp_path / "model.pkl").write_bytes(b"x")
    pcq.save_manifest(enrich=False)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["schema_version"] == 1
    entry = next(f for f in m["files"] if f["path"] == "model.pkl")
    assert "sha256" not in entry
    assert "size_bytes" not in entry
    assert "created_at" not in entry


def test_save_manifest_cfg_opt_out(tmp_path, monkeypatch):
    """cfg['manifest_checksums']=False → schema v1 fallback."""
    monkeypatch.setenv(
        "CQ_CONFIG_JSON",
        str(_setup_cfg(tmp_path, manifest_checksums=False)),
    )
    (tmp_path / "model.pkl").write_bytes(b"x")
    pcq.save_manifest()
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["schema_version"] == 1


def test_save_manifest_cfg_opt_in_explicit_true(tmp_path, monkeypatch):
    """cfg['manifest_checksums']=True 명시 → schema v2."""
    monkeypatch.setenv(
        "CQ_CONFIG_JSON",
        str(_setup_cfg(tmp_path, manifest_checksums=True)),
    )
    (tmp_path / "model.pkl").write_bytes(b"hi")
    pcq.save_manifest()
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["schema_version"] == 2


def test_save_manifest_explicit_files_with_enrich(tmp_path, monkeypatch):
    """explicit files=[...] + enrich=True → 각 entry 가 sha256/size_bytes 포함."""
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    payload = b"abc" * 100
    (tmp_path / "weights.bin").write_bytes(payload)
    pcq.save_manifest(files=[("weights.bin", "model")], enrich=True)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["schema_version"] == 2
    e = m["files"][0]
    assert e["path"] == "weights.bin"
    assert e["kind"] == "model"
    assert e["size_bytes"] == len(payload)
    assert e["sha256"] == hashlib.sha256(payload).hexdigest()


def test_save_manifest_missing_file_skipped_in_enrich(tmp_path, monkeypatch):
    """enrich=True 인데 file 없는 explicit entry → entry 는 보존, sha256 없음."""
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    pcq.save_manifest(files=[("nonexistent.bin", "model")], enrich=True)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["schema_version"] == 2
    e = m["files"][0]
    assert e["path"] == "nonexistent.bin"
    assert "sha256" not in e
    assert "size_bytes" not in e


def test_save_manifest_no_cqcfg_env_defaults_to_enrich(tmp_path, monkeypatch):
    """CQ_CONFIG_JSON 없음 + 로컬 사용 → enrich default True (schema v2)."""
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    # output_dir 를 monkeypatch 로 임시 디렉토리로
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model.pkl").write_bytes(b"abc")
    pcq.save_manifest(files=[("model.pkl", "model")])
    m = json.loads((tmp_path / "output" / "manifest.json").read_text())
    assert m["schema_version"] == 2


def test_save_all_emits_v2_manifest(tmp_path, monkeypatch):
    """save_all → manifest 도 schema v2 default."""
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    (tmp_path / "model.pkl").write_bytes(b"fake")
    history = [{"epoch": 0, "eval_acc": 0.5}]
    paths = pcq.save_all(history=history, artifacts={"model": "model.pkl"})
    m = json.loads(paths["manifest"].read_text())
    assert m["schema_version"] == 2
    # config.json/metrics.json/run_summary.json/model.pkl 모두 sha256 포함
    by_path = {f["path"]: f for f in m["files"]}
    for p in ("model.pkl", "config.json", "metrics.json", "run_summary.json"):
        assert p in by_path, f"missing {p}"
        assert "sha256" in by_path[p]
