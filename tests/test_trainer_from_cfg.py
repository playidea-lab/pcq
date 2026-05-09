"""Trainer.from_cfg(cfg) — preset/overrides_data 자동 인식 — v1.10."""
from __future__ import annotations

import pcq


def test_from_cfg_with_preset(tmp_path):
    cfg = {
        "preset": "vision/fake_smoke",
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
    }
    trainer = pcq.Trainer.from_cfg(cfg)
    trainer.fit()
    assert (tmp_path / "model.pt").exists()


def test_from_cfg_with_atomref_loss_override(tmp_path):
    """_overrides_data.loss = AtomRef.to_dict() — fit() 까지 통과."""
    cfg = {
        "preset": "vision/fake_smoke",
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "_overrides_data": {
            "loss": {
                "kind": "loss",
                "name": "cross_entropy",
                "params": {"ignore_index": -100},
            },
        },
    }
    trainer = pcq.Trainer.from_cfg(cfg)
    trainer.fit()
    assert (tmp_path / "model.pt").exists()


def test_from_cfg_with_atomref_optim_override(tmp_path):
    """_overrides_data.optim = AtomRef.to_dict() → optim_factory 로 resolve."""
    cfg = {
        "preset": "vision/fake_smoke",
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "_overrides_data": {
            "optim": {
                "kind": "optim",
                "name": "adamw",
                "params": {"lr": 1e-4},
            },
        },
    }
    trainer = pcq.Trainer.from_cfg(cfg)
    trainer.fit()
    assert (tmp_path / "model.pt").exists()


def test_from_cfg_strips_internal_keys_from_clean_cfg():
    """preset/_overrides_data 는 trainer cfg 에 들어가면 안 됨."""
    cfg = {
        "preset": "vision/fake_smoke",
        "_overrides_data": {},
        "epochs": 2,
    }
    trainer = pcq.Trainer.from_cfg(cfg)
    assert "preset" not in trainer.cfg
    assert "_overrides_data" not in trainer.cfg
    assert trainer.cfg["epochs"] == 2


def test_from_cfg_no_preset_falls_back_to_atom_only(tmp_path):
    """preset 없는 cfg → atom-only 모드 trainer (단, 기본 atoms 부재 → fit 불가).

    여기서는 instantiation 만 확인.
    """
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 8}
    # preset 없으면 atom-only — 사용자가 추가 atom 안주면 fit 불가하므로
    # 여기선 trainer 생성만 검증.
    trainer = pcq.Trainer.from_cfg(cfg)
    assert trainer.cfg["output_dir"] == str(tmp_path)
