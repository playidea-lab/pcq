"""pcq.save_* contract helpers — standalone (script-style 사용)."""
import json
from pathlib import Path

import pcq


def _setup_cfg(tmp_path: Path, **extra) -> Path:
    cfg = {"output_dir": str(tmp_path), "seed": 42}
    cfg.update(extra)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def test_save_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    history = [{"epoch": 0, "loss": 0.5}, {"epoch": 1, "loss": 0.3}]
    p = pcq.save_metrics(history)
    assert p.name == "metrics.json"
    data = json.loads(p.read_text())
    assert data == {"history": history}


def test_save_config_snapshot_includes_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path, lr=0.001)))
    p = pcq.save_config_snapshot()
    data = json.loads(p.read_text())
    assert data["lr"] == 0.001
    assert "_pcq_version" in data
    assert "_git_sha" in data


def test_save_config_snapshot_explicit_cfg_uses_cfg_output_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    p = pcq.save_config_snapshot(cfg={"output_dir": str(tmp_path), "lr": 0.01})
    assert p == tmp_path / "config.json"
    data = json.loads(p.read_text())
    assert data["lr"] == 0.01


def test_save_manifest_autodetect(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    (tmp_path / "model.pkl").write_bytes(b"fake")
    (tmp_path / "metrics.json").write_text("{}")
    pcq.save_manifest()
    m = json.loads((tmp_path / "manifest.json").read_text())
    paths = {f["path"]: f["kind"] for f in m["files"]}
    assert paths["model.pkl"] == "model"
    assert paths["metrics.json"] == "metrics"
    assert "manifest.json" not in paths


def test_save_manifest_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    pcq.save_manifest(files=[("foo.bin", "model"), ("bar.txt", "other")])
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert {"path": "foo.bin", "kind": "model"} in m["files"]


def test_save_run_summary_min_mode(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CQ_CONFIG_JSON",
        str(_setup_cfg(tmp_path, monitor="loss", mode="min")),
    )
    history = [
        {"epoch": 0, "loss": 0.5},
        {"epoch": 1, "loss": 0.3},
        {"epoch": 2, "loss": 0.4},
    ]
    pcq.save_run_summary(history=history, artifacts={"model": "model.pkl"})
    s = json.loads((tmp_path / "run_summary.json").read_text())
    assert s["best"]["epoch"] == 1
    assert s["last"]["epoch"] == 2


def test_save_run_summary_max_mode(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CQ_CONFIG_JSON",
        str(_setup_cfg(tmp_path, monitor="acc", mode="max")),
    )
    history = [{"epoch": 0, "acc": 0.7}, {"epoch": 1, "acc": 0.9}]
    pcq.save_run_summary(history=history)
    s = json.loads((tmp_path / "run_summary.json").read_text())
    assert s["best"]["epoch"] == 1


def test_save_all(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    history = [{"epoch": 0, "eval_acc": 0.5}]
    paths = pcq.save_all(history=history, artifacts={"model": "model.pkl"})
    for key in ("config", "metrics", "manifest", "run_summary"):
        assert paths[key].exists()


def test_save_run_summary_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    pcq.save_run_summary(
        history=[],
        status="failed",
        failure={"category": "oom", "message": "OOM"},
    )
    s = json.loads((tmp_path / "run_summary.json").read_text())
    assert s["status"] == "failed"
    assert s["failure"]["category"] == "oom"
