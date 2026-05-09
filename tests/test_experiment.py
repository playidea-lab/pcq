"""tests for pcq.Experiment (T-CQPY-004)."""
from __future__ import annotations

import json

import pytest

import pcq
from pcq import Experiment
from pcq import core as _core


@pytest.fixture(autouse=True)
def _reset_core_state():
    """test_experiment 도 declared cache + undeclared 누적 리셋."""
    _core._reset_declared_cache()
    _core._undeclared_warned.clear()
    _core._undeclared_count.clear()
    yield
    _core._reset_declared_cache()
    _core._undeclared_warned.clear()
    _core._undeclared_count.clear()


class SmokeExp(Experiment):
    """단순한 MLP + fake dataset 으로 fit 루프를 검증하는 smoke subclass."""

    def build_dataset(self, split):
        # 결정적 fake dataset (seed=42 고정). split 별 분리는 smoke 테스트라 동일.
        return pcq.datasets.fake(
            num_samples=32, num_classes=3, image_size=8, channels=3
        )

    def build_model(self):
        return pcq.models.mlp(3 * 8 * 8, [16], 3)

    def build_loss(self):
        return pcq.loss.cross_entropy()

    def build_optimizer(self, params):
        return pcq.optim.adamw(params, lr=1e-2)

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        return loss, {"acc": pcq.metric.accuracy(logits, y).item()}

    def eval_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        return {"loss": loss.item(), "acc": pcq.metric.accuracy(logits, y).item()}


def test_experiment_smoke_cpu(tmp_path):
    cfg = {"output_dir": str(tmp_path), "epochs": 2, "batch_size": 16, "seed": 42}
    exp = SmokeExp(cfg=cfg)
    exp.fit()
    assert len(exp.history) == 2


def test_experiment_saves_5_artifacts(tmp_path):
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16, "seed": 42}
    SmokeExp(cfg=cfg).fit()
    for name in ["model.pt", "config.json", "metrics.json", "last.ckpt", "best.ckpt"]:
        assert (tmp_path / name).exists(), f"missing {name}"


def test_experiment_history_recorded(tmp_path):
    cfg = {"output_dir": str(tmp_path), "epochs": 2, "batch_size": 16, "seed": 42}
    exp = SmokeExp(cfg=cfg)
    exp.fit()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(metrics["history"]) == 2
    assert "train_loss" in metrics["history"][0]
    assert "eval_loss" in metrics["history"][0]


def test_experiment_resume_loads_checkpoint(tmp_path):
    # 1차 학습: 2 epochs
    cfg1 = {"output_dir": str(tmp_path), "epochs": 2, "batch_size": 16, "seed": 42}
    SmokeExp(cfg=cfg1).fit()
    # Resume: 총 4 epochs 목표 (epoch 2 부터 재개)
    cfg2 = {
        "output_dir": str(tmp_path),
        "epochs": 4,
        "batch_size": 16,
        "seed": 42,
        "resume_from": str(tmp_path / "last.ckpt"),
    }
    exp2 = SmokeExp(cfg=cfg2)
    exp2.fit()
    # 재개된 인스턴스의 history 는 epoch 2,3 만 포함
    assert len(exp2.history) == 2
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["history"][0]["epoch"] == 2


def test_experiment_resume_missing_path_raises(tmp_path):
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "resume_from": str(tmp_path / "nonexistent.ckpt"),
    }
    with pytest.raises(FileNotFoundError, match="resume_from"):
        SmokeExp(cfg=cfg).fit()


def test_experiment_auto_resume_picks_up_last_ckpt(tmp_path):
    """cfg['resume']=True + output_dir/last.ckpt exists → 자동 resume."""
    # 1차: 2 epochs 학습 → output_dir/last.ckpt 생성
    cfg1 = {"output_dir": str(tmp_path), "epochs": 2, "batch_size": 16, "seed": 42}
    SmokeExp(cfg=cfg1).fit()
    # 2차: resume=True (resume_from 명시 X) → 자동 발견
    cfg2 = {
        "output_dir": str(tmp_path),
        "epochs": 4,
        "batch_size": 16,
        "seed": 42,
        "resume": True,
    }
    exp2 = SmokeExp(cfg=cfg2)
    exp2.fit()
    # 새 fit 에서 epochs 2~3 만 실행되어야 한다
    assert len(exp2.history) == 2
    assert exp2.history[0]["epoch"] == 2


def test_experiment_auto_resume_silent_when_no_ckpt(tmp_path):
    """cfg['resume']=True + output_dir/last.ckpt 없음 → silent fresh start."""
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "resume": True,
    }
    exp = SmokeExp(cfg=cfg)
    # last.ckpt 없는 fresh 디렉토리 → 에러 없이 fresh start
    exp.fit()
    assert len(exp.history) == 1


def test_experiment_explicit_resume_from_overrides_auto(tmp_path):
    """resume_from 명시 우선. resume_from missing → 여전히 raise (auto fallback X)."""
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "resume": True,
        "resume_from": str(tmp_path / "nonexistent.ckpt"),
    }
    with pytest.raises(FileNotFoundError, match="resume_from"):
        SmokeExp(cfg=cfg).fit()


def test_experiment_accelerate_optional(tmp_path):
    # accelerate 미설치 환경에서도 fit() 이 정상 동작해야 한다
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    SmokeExp(cfg=cfg).fit()


def test_experiment_log_called_per_epoch(tmp_path, capsys):
    cfg = {"output_dir": str(tmp_path), "epochs": 2, "batch_size": 16, "seed": 42}
    SmokeExp(cfg=cfg).fit()
    captured = capsys.readouterr()
    # 두 epoch → @epoch=0, @epoch=1 가 stdout 에 기록되어야 한다
    assert "@epoch=0" in captured.out
    assert "@epoch=1" in captured.out
    assert "@train_loss=" in captured.out
    assert "@eval_loss=" in captured.out


def test_training_step_must_return_tuple(tmp_path):
    """training_step 이 dict 만 반환하면 명확한 TypeError."""

    class BadExp(SmokeExp):
        def training_step(self, batch):
            x, y = batch
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            return {"loss": loss}

    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    with pytest.raises(TypeError, match="training_step must return"):
        BadExp(cfg=cfg).fit()


def test_training_step_auto_includes_loss_metric(tmp_path, capsys):
    """training_step 이 metrics 에 'loss' 안 넣어도 자동 추가되어 logging."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    SmokeExp(cfg=cfg).fit()
    out = capsys.readouterr().out
    assert "@train_loss=" in out


def test_eval_step_must_not_return_tuple(tmp_path):
    """eval_step 이 tuple 반환하면 명확한 TypeError (training_step 과 혼동 방지)."""

    class BadExp(SmokeExp):
        def eval_step(self, batch):
            x, y = batch
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            return loss, {}

    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    with pytest.raises(TypeError, match="eval_step must return"):
        BadExp(cfg=cfg).fit()


# ── v1.2.1 / v1.14: Manifest ────────────────────────────────────────────────
def test_manifest_written_with_schema_v2_default(tmp_path):
    """v1.14 default — schema v2 + sha256/size_bytes/created_at per entry."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    SmokeExp(cfg=cfg).fit()
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    paths = {f["path"] for f in manifest["files"]}
    assert {
        "model.pt",
        "config.json",
        "metrics.json",
        "last.ckpt",
        "best.ckpt",
    }.issubset(paths)
    # manifest.json 자기참조 X
    assert "manifest.json" not in paths
    # kind 분류 정확 (v1.13 호환)
    kinds = {f["path"]: f["kind"] for f in manifest["files"]}
    assert kinds["model.pt"] == "weights"
    assert kinds["config.json"] == "config"
    assert kinds["metrics.json"] == "metrics"
    assert kinds["last.ckpt"] == "checkpoint"
    assert kinds["best.ckpt"] == "checkpoint"
    # v2 evidence — 모든 entry 가 sha256/size_bytes/created_at 보유
    import hashlib

    for entry in manifest["files"]:
        assert "sha256" in entry, f"missing sha256 in {entry['path']}"
        assert "size_bytes" in entry
        assert "created_at" in entry
        actual = hashlib.sha256(
            (tmp_path / entry["path"]).read_bytes()
        ).hexdigest()
        assert actual == entry["sha256"], f"sha256 mismatch: {entry['path']}"


def test_manifest_v1_when_opt_out(tmp_path):
    """cfg['manifest_checksums']=False → schema v1 fallback (path/kind only)."""
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "manifest_checksums": False,
    }
    SmokeExp(cfg=cfg).fit()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    for entry in manifest["files"]:
        assert "sha256" not in entry
        assert "size_bytes" not in entry


# ── v1.2.1: Best monitor + mode ─────────────────────────────────────────────
def test_best_monitor_default_eval_loss_min(tmp_path):
    """기본 동작 — eval_loss 최소를 best 로 (기존 동작과 동일)."""
    cfg = {"output_dir": str(tmp_path), "epochs": 2, "batch_size": 16, "seed": 42}
    SmokeExp(cfg=cfg).fit()
    assert (tmp_path / "best.ckpt").exists()


def test_best_monitor_custom_mode_max(tmp_path):
    """eval_acc max 로 monitor — best.ckpt 가 acc 가장 높은 epoch 저장."""
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 2,
        "batch_size": 16,
        "seed": 42,
        "monitor": "eval_acc",
        "mode": "max",
    }
    SmokeExp(cfg=cfg).fit()
    assert (tmp_path / "best.ckpt").exists()


def test_best_monitor_warns_when_key_missing(tmp_path, capsys):
    """monitor key 가 metrics 에 없으면 stderr 1회 경고 + best.ckpt skip."""
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 2,
        "batch_size": 16,
        "monitor": "nonexistent_metric",
        "mode": "min",
    }
    SmokeExp(cfg=cfg).fit()
    err = capsys.readouterr().err
    assert "nonexistent_metric" in err
    assert err.count("not in metrics") == 1  # 1회만 경고
    # best.ckpt 는 monitor 못해서 안 만들어짐
    assert not (tmp_path / "best.ckpt").exists()
    # last.ckpt 는 매 epoch 저장되니 존재
    assert (tmp_path / "last.ckpt").exists()


def test_best_monitor_invalid_mode_raises(tmp_path):
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "mode": "bogus",
    }
    with pytest.raises(ValueError, match="mode"):
        SmokeExp(cfg=cfg)


# ── v1.2.1: Device resolve ──────────────────────────────────────────────────
def test_device_explicit_cfg_wins(tmp_path):
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "device": "cpu",
    }
    exp = SmokeExp(cfg=cfg)
    assert exp.device == "cpu"


def test_device_invalid_raises():
    with pytest.raises(ValueError, match="device"):
        SmokeExp(cfg={"device": ""})  # empty 명시 → invalid


def test_device_auto_detects(tmp_path):
    """cfg 에 device 없으면 cuda/mps/cpu 자동 감지."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    exp = SmokeExp(cfg=cfg)
    assert exp.device in ("cuda", "mps", "cpu")


def test_device_falls_back_to_cpu_when_no_accelerator(monkeypatch, tmp_path):
    """cuda/mps 둘 다 없는 환경에선 cpu."""
    import torch.backends

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    exp = SmokeExp(cfg=cfg)
    assert exp.device == "cpu"


def test_explicit_device_skips_accelerate(tmp_path):
    """cfg['device'] 명시 시 accelerate 우회 — model.to(device) 직접."""
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "device": "cpu",
    }
    exp = SmokeExp(cfg=cfg)
    exp.fit()
    assert exp._accelerator is None  # accelerate 미사용
    assert exp.device == "cpu"


# ── v1.3: AMP / Gradient Accumulation ───────────────────────────────────────
def test_amp_disabled_by_default(tmp_path):
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    exp = SmokeExp(cfg=cfg)
    assert exp._amp_enabled is False


def test_amp_invalid_dtype_raises(tmp_path):
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "amp_dtype": "fp32",
    }
    with pytest.raises(ValueError, match="amp_dtype"):
        SmokeExp(cfg=cfg)


def test_amp_bf16_cpu_smoke(tmp_path):
    """bf16 + cpu — GradScaler 불필요, autocast 만 동작."""
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "device": "cpu",
        "amp": True,
        "amp_dtype": "bf16",
    }
    exp = SmokeExp(cfg=cfg)
    exp.fit()
    assert (tmp_path / "model.pt").exists()
    # GradScaler 는 fp16+cuda 에서만 — cpu/bf16 는 None
    assert exp._scaler is None


def test_grad_accum_smoke(tmp_path):
    """grad_accum=4 + batch_size=4 — 학습 정상 진행."""
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 4,
        "grad_accum": 4,
        "device": "cpu",
    }
    SmokeExp(cfg=cfg).fit()
    assert (tmp_path / "model.pt").exists()


def test_grad_accum_invalid_clamps_to_one(tmp_path):
    """grad_accum=0 또는 음수 → 1 로 clamp."""
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "grad_accum": 0,
    }
    exp = SmokeExp(cfg=cfg)
    assert exp._grad_accum == 1


# ── v1.3: Early Stopping ────────────────────────────────────────────────────
def test_early_stop_disabled_by_default(tmp_path):
    cfg = {"output_dir": str(tmp_path), "epochs": 3, "batch_size": 16, "seed": 42}
    exp = SmokeExp(cfg=cfg)
    exp.fit()
    # patience=0 (default) → 절대 중단 안 됨
    assert len(exp.history) == 3
    assert exp._early_stopped_at is None


def test_early_stop_triggers(tmp_path, capsys):
    """min_delta 를 매우 크게 → 사실상 절대 개선 안 됨 → 첫 무개선부터 카운트."""
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 10,
        "batch_size": 16,
        "seed": 42,
        "early_stop_patience": 1,
        "early_stop_min_delta": 100.0,  # 절대 개선 못함
    }
    exp = SmokeExp(cfg=cfg)
    exp.fit()
    err = capsys.readouterr().err
    assert "early stop" in err
    assert exp._early_stopped_at is not None
    assert len(exp.history) < 10
    # metrics.json 에 early_stopped_at_epoch 기록
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert "early_stopped_at_epoch" in metrics


def test_early_stop_count_resumed(tmp_path):
    """no_improve_count 가 ckpt 에 저장 + resume 시 복원."""
    # 1차: 무개선 epoch 만 — patience=5 라 중단 X
    cfg1 = {
        "output_dir": str(tmp_path),
        "epochs": 2,
        "batch_size": 16,
        "seed": 42,
        "early_stop_patience": 5,
        "early_stop_min_delta": 100.0,
    }
    exp1 = SmokeExp(cfg=cfg1)
    exp1.fit()
    assert exp1._no_improve_count >= 1
    saved_count = exp1._no_improve_count
    # 2차: resume → no_improve_count 복원되어야 함
    cfg2 = {
        "output_dir": str(tmp_path),
        "epochs": 10,
        "batch_size": 16,
        "seed": 42,
        "early_stop_patience": 5,
        "early_stop_min_delta": 100.0,
        "resume_from": str(tmp_path / "last.ckpt"),
    }
    exp2 = SmokeExp(cfg=cfg2)
    exp2.fit()
    # 누적 카운트 → 더 일찍 중단 (resume 후)
    assert exp2._no_improve_count >= saved_count


# ── v1.3: Metrics aggregation ───────────────────────────────────────────────
def test_metrics_aggregation_default_is_mean(tmp_path):
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    exp = SmokeExp(cfg=cfg)
    assert exp._agg_mode == "mean"


def test_metrics_aggregation_weighted_mean(tmp_path):
    """가변 batch size 환경에서 weighted_mean 도 정상 학습."""
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "metrics_aggregation": "weighted_mean",
    }
    exp = SmokeExp(cfg=cfg)
    exp.fit()
    assert exp._agg_mode == "weighted_mean"
    assert (tmp_path / "model.pt").exists()


def test_metrics_aggregation_invalid_raises(tmp_path):
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "metrics_aggregation": "median",
    }
    with pytest.raises(ValueError, match="metrics_aggregation"):
        SmokeExp(cfg=cfg)


# ── v1.3: Resume restores scheduler / monitor / early-stop state ────────────
def test_resume_restores_scheduler_state(tmp_path):
    """scheduler step 카운트가 ckpt 에 저장 + 복원 (구체적으로 sched.last_epoch)."""

    class SchedExp(SmokeExp):
        def build_scheduler(self, optimizer):
            return pcq.sched.cosine(optimizer, T_max=10, warmup=2)

    cfg1 = {"output_dir": str(tmp_path), "epochs": 3, "batch_size": 16, "seed": 42}
    SchedExp(cfg=cfg1).fit()
    cfg2 = {
        "output_dir": str(tmp_path),
        "epochs": 5,
        "batch_size": 16,
        "seed": 42,
        "resume_from": str(tmp_path / "last.ckpt"),
    }
    exp2 = SchedExp(cfg=cfg2)
    exp2.fit()
    # 이어서 epoch 3,4 실행
    assert exp2._start_epoch == 3
    assert len(exp2.history) == 2


def test_resume_restores_monitor_state(tmp_path):
    """_best_monitored 가 정확히 복원 (v1.2.1 + v1.3 강화)."""
    cfg1 = {"output_dir": str(tmp_path), "epochs": 2, "batch_size": 16, "seed": 42}
    SmokeExp(cfg=cfg1).fit()
    cfg2 = {
        "output_dir": str(tmp_path),
        "epochs": 4,
        "batch_size": 16,
        "seed": 42,
        "resume_from": str(tmp_path / "last.ckpt"),
    }
    exp2 = SmokeExp(cfg=cfg2)
    exp2.fit()
    assert exp2._best_monitored is not None


# ── v1.6 Gap 4: accelerate multi-process guard ──────────────────────────────
def test_main_process_guard_default_true(tmp_path):
    """accelerate 미사용 시 _is_main_process 는 항상 True."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    exp = SmokeExp(cfg=cfg)
    assert exp._is_main_process() is True


def test_artifacts_written_in_single_process(tmp_path):
    """단일 프로세스 — main process guard 적용 후에도 모든 artifact 정상 저장.

    Multi-process 실제 검증은 v2 (gap 4 회귀 방지 단일 프로세스 회귀 테스트).
    """
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    SmokeExp(cfg=cfg).fit()
    for name in [
        "model.pt",
        "config.json",
        "metrics.json",
        "last.ckpt",
        "best.ckpt",
        "manifest.json",
    ]:
        assert (tmp_path / name).exists(), f"missing {name}"


# ── v1.6 Gap 5: monitor pre-validation against declared metrics ─────────────
def test_monitor_warning_when_not_in_declared(
    tmp_path, monkeypatch, capsys
):
    """declared metrics 가 있고 monitor 가 그 안에 없으면 fit() 시작 시 stderr 경고."""
    monkeypatch.setenv("CQ_DECLARED_METRICS", "epoch,train_loss")
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "monitor": "eval_iou",
        "mode": "max",
    }
    SmokeExp(cfg=cfg).fit()
    err = capsys.readouterr().err
    assert "monitor" in err
    assert "eval_iou" in err
    assert "not in declared" in err


def test_no_monitor_warning_when_declared(tmp_path, monkeypatch, capsys):
    """monitor 가 declared metrics 안에 있으면 fit() 시작 경고 없음."""
    monkeypatch.setenv(
        "CQ_DECLARED_METRICS", "epoch,train_loss,eval_loss,eval_acc,train_acc"
    )
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "monitor": "eval_loss",
        "mode": "min",
    }
    SmokeExp(cfg=cfg).fit()
    err = capsys.readouterr().err
    assert "not in declared" not in err


def test_no_monitor_warning_when_no_declared_metrics(
    tmp_path, monkeypatch, capsys
):
    """declared metrics 자체가 없으면 monitor 검증 skip — 경고 없음."""
    monkeypatch.delenv("CQ_DECLARED_METRICS", raising=False)
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "monitor": "eval_iou",
        "mode": "max",
    }
    SmokeExp(cfg=cfg).fit()
    err = capsys.readouterr().err
    assert "not in declared" not in err


# ── v1.16: Experiment.fit() 자동 finalize → run_record.json ─────────────────
def test_experiment_fit_writes_run_record(tmp_path):
    """Experiment.fit() 자동 finalize → run_record.json + validation_report.json."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    SmokeExp(cfg=cfg).fit()
    assert (tmp_path / "run_record.json").exists()
    assert (tmp_path / "validation_report.json").exists()
    rr = json.loads((tmp_path / "run_record.json").read_text())
    assert rr["schema_version"] == 1
    assert rr["run"]["status"] == "completed"


def test_experiment_fit_run_record_has_environment(tmp_path):
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    SmokeExp(cfg=cfg).fit()
    rr = json.loads((tmp_path / "run_record.json").read_text())
    assert rr["environment"]["python"]
    assert "platform" in rr["environment"]


def test_experiment_fit_run_record_artifacts_match_manifest(tmp_path):
    """run_record.artifacts 가 manifest.files 와 동일한 path 들을 포함."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    SmokeExp(cfg=cfg).fit()
    rr = json.loads((tmp_path / "run_record.json").read_text())
    m = json.loads((tmp_path / "manifest.json").read_text())
    rr_paths = {a.get("path") for a in rr["artifacts"]}
    manifest_paths = {f.get("path") for f in m["files"]}
    assert manifest_paths.issubset(rr_paths)
