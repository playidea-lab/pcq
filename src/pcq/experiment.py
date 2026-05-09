"""pcq.experiment — Lightning-style mid-level training API (T-CQPY-004).

사용자가 build_* / *_step 메서드를 override 하면 fit() 이 train/eval epoch 를
자동으로 돌리고 stdout @key=value 메트릭, 체크포인트, history 를 산출한다.
accelerate 는 선택 의존성으로, 설치되어 있을 때만 multi-device 가속을 사용한다.
"""
from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from pcq import core


def _git_sha() -> str:
    # 현재 HEAD 의 git sha 반환. git 없거나 timeout 이면 빈 문자열.
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _pcq_version() -> str:
    # cq 패키지 __version__ 반환. import 실패 시 'unknown'.
    try:
        from pcq import __version__

        return __version__
    except ImportError:
        return "unknown"


def _try_import_accelerate():
    # accelerate 가 설치되어 있으면 Accelerator 클래스 반환, 아니면 None.
    try:
        from accelerate import Accelerator

        return Accelerator
    except ImportError:
        return None


def _resolve_device(cfg: dict) -> str:
    """Device resolution priority: cfg['device'] > cuda > mps > cpu.

    명시값은 비어있지 않은 문자열만 허용. 자동 감지는 cuda → mps → cpu 순서.
    """
    explicit = cfg.get("device")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError(
                f"cfg['device'] must be a non-empty string, got {explicit!r}"
            )
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    # MPS (Apple Silicon)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


class Experiment:
    """Lightning-style 중레벨 학습 API. subclass 후 build_* / *_step override."""

    def __init__(self, cfg: Optional[dict] = None) -> None:
        if cfg is None:
            try:
                cfg = core.config()
            except RuntimeError:
                cfg = {}
        self.cfg: dict = cfg
        self.output_dir: Path = Path(cfg.get("output_dir", "output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []
        # Device 우선순위: cfg['device'] > cuda > mps > cpu
        self.device: str = _resolve_device(self.cfg)
        # Best monitor — cfg['monitor'] (default 'eval_loss') + cfg['mode']
        self._monitor_key: str = self.cfg.get("monitor", "eval_loss")
        self._monitor_mode: str = self.cfg.get("mode", "min")
        if self._monitor_mode not in ("min", "max"):
            raise ValueError(
                f"cfg['mode'] must be 'min' or 'max', got {self._monitor_mode!r}"
            )
        self._monitor_warned: bool = False
        self._best_monitored: Optional[float] = None
        self._start_epoch: int = 0
        # accelerate 사용 여부는 fit() 안에서 모델/옵티마이저 빌드 후 결정
        self._accelerator = None
        # ── v1.3 ────────────────────────────────────────────────────────────
        # AMP — autocast + GradScaler. accelerate 활성화 시 우회.
        self._amp_enabled: bool = bool(self.cfg.get("amp", False))
        self._amp_dtype: str = self.cfg.get("amp_dtype", "fp16")
        if self._amp_dtype not in ("fp16", "bf16"):
            raise ValueError(
                f"cfg['amp_dtype'] must be 'fp16' or 'bf16', got {self._amp_dtype!r}"
            )
        self._scaler: Optional[torch.amp.GradScaler] = None
        # Gradient accumulation — effective batch_size = batch_size * grad_accum
        self._grad_accum: int = max(1, int(self.cfg.get("grad_accum", 1)))
        # Early stopping — patience>0 활성화. min_delta 만큼 개선 못하면 카운트.
        self._patience: int = int(self.cfg.get("early_stop_patience", 0))
        self._min_delta: float = float(self.cfg.get("early_stop_min_delta", 0.0))
        self._no_improve_count: int = 0
        self._early_stopped_at: Optional[int] = None
        # Metric aggregation — 'mean' (default) | 'weighted_mean' (batch_size 가중)
        self._agg_mode: str = self.cfg.get("metrics_aggregation", "mean")
        if self._agg_mode not in ("mean", "weighted_mean"):
            raise ValueError(
                f"cfg['metrics_aggregation'] must be 'mean' or 'weighted_mean', "
                f"got {self._agg_mode!r}"
            )

    # ── Override these ───────────────────────────────────────────────────────
    def build_dataset(self, split: str) -> Dataset:
        raise NotImplementedError("Override build_dataset(split: 'train'|'eval')")

    def build_model(self) -> nn.Module:
        raise NotImplementedError("Override build_model()")

    def build_loss(self) -> nn.Module:
        raise NotImplementedError("Override build_loss()")

    def build_optimizer(self, params) -> Optimizer:
        raise NotImplementedError("Override build_optimizer(params)")

    def build_scheduler(self, optimizer: Optimizer):
        # 선택. 기본은 scheduler 미사용.
        return None

    def training_step(self, batch) -> tuple:
        raise NotImplementedError(
            "Override training_step(batch) → (loss_tensor, metrics_dict). "
            "loss_tensor 는 backward 용 grad 살아있는 tensor, "
            "metrics_dict 는 logging 용 스칼라 dict."
        )

    def eval_step(self, batch) -> dict:
        raise NotImplementedError(
            "Override eval_step(batch) → dict[str, float|tensor]"
        )

    def _is_main_process(self) -> bool:
        """multi-process accelerate 환경에서 main 인지 판정.

        accelerate 미사용 시 항상 True. multi-GPU 학습에서 stdout/체크포인트/
        artifact 쓰기 race 를 막는 가드 헬퍼.
        """
        if self._accelerator is not None:
            return self._accelerator.is_main_process
        return True

    # ── Default fit loop ─────────────────────────────────────────────────────
    def fit(self) -> None:
        # 시드 고정 (재현성)
        core.seed_everything(int(self.cfg.get("seed", 42)))

        # Monitor 사전 검증 — declared metrics 가 있고 monitor 가 그 안에 없으면 경고.
        # 학습 시작 전에 best.ckpt 못 만들 수 있다는 사실을 agent/사용자에게 알림.
        declared = core._read_declared_metrics()
        if declared is not None and self._monitor_key not in declared:
            print(
                f"[cq] warning: monitor={self._monitor_key!r} not in declared "
                f"metrics {sorted(declared)}; best.ckpt may not be created.",
                file=sys.stderr,
            )

        # 데이터셋 / 로더
        train_ds = self.build_dataset("train")
        eval_ds = self.build_dataset("eval")
        batch_size = int(self.cfg.get("batch_size", 32))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

        # 모델 / 손실 / 옵티마이저 / 스케줄러
        self.model = self.build_model()
        self.loss_fn = self.build_loss()
        self.optimizer = self.build_optimizer(self.model.parameters())
        self.scheduler = self.build_scheduler(self.optimizer)

        # accelerate 가용 시 wrap, 아니면 device 로 직접 이동
        # 단, cfg['device'] 가 명시되면 사용자 의지 우선 — accelerate 우회
        explicit_device = self.cfg.get("device")
        Accelerator = (
            _try_import_accelerate() if explicit_device is None else None
        )
        if Accelerator is not None:
            self._accelerator = Accelerator()
            (
                self.model,
                self.optimizer,
                train_loader,
                eval_loader,
            ) = self._accelerator.prepare(
                self.model, self.optimizer, train_loader, eval_loader
            )
            self.device = str(self._accelerator.device)
        else:
            self.model.to(self.device)

        # AMP GradScaler — accelerate 미사용 + fp16 + cuda 일 때만 필요
        # bf16 / cpu / mps 는 GradScaler 불필요 (autocast 만 사용)
        if (
            self._amp_enabled
            and self._accelerator is None
            and self._amp_dtype == "fp16"
            and self.device.startswith("cuda")
        ):
            self._scaler = torch.amp.GradScaler()

        # Resume from checkpoint
        # 우선순위: 명시적 resume_from > 자동 resume (output_dir/last.ckpt)
        resume_from = self.cfg.get("resume_from")
        if not resume_from and self.cfg.get("resume"):
            # 자동 resume: output_dir/last.ckpt 가 있으면 resume, 없으면 silent fresh start
            candidate = self.output_dir / "last.ckpt"
            if candidate.exists():
                resume_from = str(candidate)
        if resume_from:
            ckpt_path = Path(resume_from)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"resume_from path not found: {resume_from}"
                )
            ckpt = torch.load(
                ckpt_path, map_location=self.device, weights_only=False
            )
            self._unwrap(self.model).load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self._start_epoch = int(ckpt.get("epoch", -1)) + 1
            # 하위호환: 구 ckpt 는 'best_eval_loss' 만 있을 수 있음
            self._best_monitored = ckpt.get("best_monitored")
            if self._best_monitored is None:
                self._best_monitored = ckpt.get("best_eval_loss")
            if self.scheduler is not None and "scheduler" in ckpt:
                with contextlib.suppress(Exception):
                    self.scheduler.load_state_dict(ckpt["scheduler"])
            # AMP scaler 복원
            if self._scaler is not None and ckpt.get("scaler") is not None:
                with contextlib.suppress(Exception):
                    self._scaler.load_state_dict(ckpt["scaler"])
            # Early stop 누적 카운트 복원
            if "no_improve_count" in ckpt:
                self._no_improve_count = int(ckpt["no_improve_count"])

        epochs = int(self.cfg.get("epochs", 1))
        for epoch in range(self._start_epoch, epochs):
            train_metrics = self._run_epoch(train_loader, train=True)
            eval_metrics = self._run_epoch(eval_loader, train=False)

            # stdout 로 @key=value 메트릭 송출 (cq Hub MetricWriter 가 파싱)
            # multi-process accelerate: main process 만 stdout 출력 (race 방지)
            log_kwargs: dict[str, Any] = {"epoch": epoch}
            for k, v in train_metrics.items():
                log_kwargs[f"train_{k}"] = v
            for k, v in eval_metrics.items():
                log_kwargs[f"eval_{k}"] = v
            if self._is_main_process():
                core.log(**log_kwargs)

            # In-memory history (metrics.json 으로도 저장)
            entry: dict[str, Any] = {"epoch": epoch}
            entry.update({f"train_{k}": v for k, v in train_metrics.items()})
            entry.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            self.history.append(entry)

            # 체크포인트 — last 는 매 epoch, best 는 monitor 기준 갱신 시.
            # _save_checkpoint 내부에서 main process guard.
            self._save_checkpoint(epoch, "last.ckpt")
            monitored_value = self._extract_monitor_value(entry)
            if monitored_value is not None:
                if self._best_monitored is None:
                    self._best_monitored = monitored_value
                    self._save_checkpoint(epoch, "best.ckpt")
                    self._no_improve_count = 0
                else:
                    # min_delta 만큼 개선되어야 'improved' 로 카운트
                    improved = (
                        monitored_value < self._best_monitored - self._min_delta
                        if self._monitor_mode == "min"
                        else monitored_value
                        > self._best_monitored + self._min_delta
                    )
                    if improved:
                        self._best_monitored = monitored_value
                        self._save_checkpoint(epoch, "best.ckpt")
                        self._no_improve_count = 0
                    else:
                        self._no_improve_count += 1

            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping — patience 초과 시 중단
            if self._patience > 0 and self._no_improve_count >= self._patience:
                if self._is_main_process():
                    print(
                        f"[cq] early stop at epoch {epoch} "
                        f"(no improvement for {self._patience} epochs)",
                        file=sys.stderr,
                    )
                self._early_stopped_at = epoch
                break

        # 모든 process 동기화 — 최종 artifact 쓰기 전 epoch 마무리 보장
        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()

        # 최종 아티팩트 — 모델 weight, config (git sha 포함), metrics history.
        # multi-process accelerate 환경에서는 main process 만 디스크 쓰기.
        if self._is_main_process():
            torch.save(
                self._unwrap(self.model).state_dict(),
                self.output_dir / "model.pt",
            )
            # provenance 메타를 self.cfg 에 주입 → build_run_summary 도 읽을 수 있게
            self.cfg["_git_sha"] = _git_sha()
            self.cfg["_pcq_version"] = _pcq_version()
            with open(
                self.output_dir / "config.json", "w", encoding="utf-8"
            ) as f:
                json.dump(dict(self.cfg), f, indent=2, default=str)
            metrics_payload: dict[str, Any] = {"history": self.history}
            if self._early_stopped_at is not None:
                metrics_payload["early_stopped_at_epoch"] = (
                    self._early_stopped_at
                )
            with open(
                self.output_dir / "metrics.json", "w", encoding="utf-8"
            ) as f:
                json.dump(metrics_payload, f, indent=2)

            # v1.7: run_summary.json — agent 가 학습 결과를 한 눈에 요약 가능하게.
            # manifest 작성 직전에 합성·기록 (manifest 가 run_summary 도 추적).
            from pcq.agent.summary import build_run_summary

            run_summary = build_run_summary(self)
            with open(
                self.output_dir / "run_summary.json", "w", encoding="utf-8"
            ) as f:
                json.dump(run_summary.to_dict(), f, indent=2)

            # Manifest — output_dir 의 artifact 인덱스 (cq worker 가 사용)
            # v1.14: schema v2 (sha256 + size_bytes + created_at) default.
            # cfg["manifest_checksums"]=False 면 v1 fallback (대형 weight opt-out).
            # contract.save_manifest 우회: in-memory self.cfg 권위 + import 순환 회피.
            from pcq.contract import _file_metadata

            enrich = bool(self.cfg.get("manifest_checksums", True))
            manifest_entries: list[dict[str, Any]] = []
            base_files: list[tuple[str, str]] = [
                ("model.pt", "weights"),
                ("config.json", "config"),
                ("metrics.json", "metrics"),
                ("run_summary.json", "summary"),
            ]
            for ckpt_name in ("last.ckpt", "best.ckpt"):
                if (self.output_dir / ckpt_name).exists():
                    base_files.append((ckpt_name, "checkpoint"))
            for rel_path, kind in base_files:
                entry: dict[str, Any] = {"path": rel_path, "kind": kind}
                if enrich:
                    meta = _file_metadata(self.output_dir, rel_path)
                    if meta:
                        entry.update(meta)
                manifest_entries.append(entry)
            manifest: dict[str, Any] = {
                "schema_version": 2 if enrich else 1,
                "files": manifest_entries,
            }
            # manifest.json 자체는 자기참조 회피로 포함하지 않음
            with open(
                self.output_dir / "manifest.json", "w", encoding="utf-8"
            ) as f:
                json.dump(manifest, f, indent=2)

            # v1.16: run_record.json + validation_report.json 자동 생성.
            # finalize_run 은 core.config() (CQ_CONFIG_JSON) 을 읽으므로
            # in-memory self.cfg 를 임시 파일로 inject. finalize 실패는 학습
            # 결과를 막지 않음 — warn 만 띄움.
            self._finalize_run_artifacts()

    def _finalize_run_artifacts(self) -> None:
        """v1.16: run_record.json + validation_report.json 작성 helper.

        v2.5: chdir/env tmp-file 트릭 제거. finalize_run(output_dir=self.output_dir)
        직접 호출 — RunContext 가 cfg/project_root 결정.

        학습 끝난 후 호출. finalize 실패는 학습 결과를 막지 않음 — warn만.
        """
        try:
            from pcq.contract import finalize_run

            finalize_run(
                history=self.history,
                status="completed",
                plan_id=self.cfg.get("_plan_id"),
                intent=self.cfg.get("_plan_intent"),
                output_dir=self.output_dir,
            )
        except Exception as e:  # noqa: BLE001 — finalize 실패는 학습 결과 막지 않음
            print(
                f"[cq] warning: finalize_run failed: {e}", file=sys.stderr
            )

    # ── Internals ────────────────────────────────────────────────────────────
    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        # train=True 면 backward+step (with AMP/grad_accum), 아니면 no_grad eval.
        self.model.train(train)
        agg: dict[str, float] = {}
        weight_total: float = 0.0
        if train:
            # epoch 시작 시 grad 초기화 (이전 epoch 의 미완료 accum 잔존 방지)
            self.optimizer.zero_grad()
        batches_in_epoch = len(loader)
        for step, batch in enumerate(loader):
            # accelerate 사용 시 로더가 이미 device 이동을 처리함
            if self._accelerator is None:
                batch = self._move_to_device(batch)
            if train:
                metrics = self._train_step_with_amp(batch, step, batches_in_epoch)
            else:
                with torch.no_grad():
                    metrics = self.eval_step(batch)
                if isinstance(metrics, tuple):
                    raise TypeError(
                        "eval_step must return a metrics dict (not tuple). "
                        "training_step returns (loss, metrics) but eval_step "
                        "returns just metrics."
                    )
            # weighted_mean 모드는 batch_size 를 가중치로 사용
            weight = self._infer_batch_size(batch) if self._agg_mode == "weighted_mean" else 1
            for k, v in metrics.items():
                vf = float(v.item() if hasattr(v, "item") else v)
                agg[k] = agg.get(k, 0.0) + vf * weight
            weight_total += weight
        return {k: v / max(1.0, weight_total) for k, v in agg.items()}

    def _train_step_with_amp(
        self, batch, step: int, batches_in_epoch: int
    ) -> dict:
        # AMP autocast 안에서 forward + backward + (조건부) optimizer.step.
        # grad_accum 만큼 누적 후 한 번 step.
        autocast_ctx: Any
        if self._amp_enabled and self._accelerator is None:
            dtype = (
                torch.float16
                if self._amp_dtype == "fp16"
                else torch.bfloat16
            )
            device_type = self.device.split(":")[0]
            autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)
        else:
            autocast_ctx = contextlib.nullcontext()
        with autocast_ctx:
            result = self.training_step(batch)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "training_step must return (loss_tensor, metrics_dict). "
                    f"Got: {type(result).__name__}"
                )
            loss, metrics = result
        # 편의: metrics 에 'loss' 없으면 자동 추가 (unscaled, logging 일관성)
        if "loss" not in metrics:
            metrics = {"loss": float(loss.detach().item()), **metrics}
        # grad_accum: scale-down 후 backward (logging 용 loss 는 unscaled 유지)
        loss_for_back = loss / self._grad_accum if self._grad_accum > 1 else loss
        if self._scaler is not None:
            self._scaler.scale(loss_for_back).backward()
        elif self._accelerator is not None:
            self._accelerator.backward(loss_for_back)
        else:
            loss_for_back.backward()
        # optimizer.step() 조건: accum_steps 마다 또는 마지막 step
        is_last_step = step == batches_in_epoch - 1
        should_step = (step + 1) % self._grad_accum == 0 or is_last_step
        if should_step:
            if self._scaler is not None:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        return metrics

    def _infer_batch_size(self, batch) -> int:
        # weighted_mean 집계용 batch_size 추정.
        # tuple/list 면 첫 원소의 첫 차원, tensor 면 첫 차원, 그 외엔 1.
        if isinstance(batch, (list, tuple)) and len(batch) > 0 and hasattr(batch[0], "shape"):
            return int(batch[0].shape[0])
        if hasattr(batch, "shape"):
            return int(batch.shape[0])
        return 1

    def _move_to_device(self, batch):
        # tuple/list 배치는 원소별로, tensor 는 직접 device 이동.
        if isinstance(batch, (list, tuple)):
            return type(batch)(
                b.to(self.device) if hasattr(b, "to") else b for b in batch
            )
        if hasattr(batch, "to"):
            return batch.to(self.device)
        return batch

    def _unwrap(self, model: nn.Module) -> nn.Module:
        # accelerate wrapper 제거 (state_dict 저장/로드 호환)
        if self._accelerator is not None:
            return self._accelerator.unwrap_model(model)
        return model

    def _extract_monitor_value(self, entry: dict) -> Optional[float]:
        """Find monitor key in epoch entry. Warn once on stderr if missing."""
        if self._monitor_key in entry:
            return float(entry[self._monitor_key])
        if not self._monitor_warned:
            self._monitor_warned = True
            print(
                f"[cq] warning: monitor key {self._monitor_key!r} not in metrics; "
                f"best.ckpt will not be updated. Available: {sorted(entry.keys())}",
                file=sys.stderr,
            )
        return None

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        # multi-process accelerate: main process 만 ckpt 디스크 쓰기 (race 방지)
        if not self._is_main_process():
            return
        ckpt: dict[str, Any] = {
            "epoch": epoch,
            "model": self._unwrap(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_monitored": self._best_monitored,
            "no_improve_count": self._no_improve_count,
            "scaler": (
                self._scaler.state_dict() if self._scaler is not None else None
            ),
        }
        if self.scheduler is not None:
            with contextlib.suppress(Exception):
                ckpt["scheduler"] = self.scheduler.state_dict()
        torch.save(ckpt, self.output_dir / name)
