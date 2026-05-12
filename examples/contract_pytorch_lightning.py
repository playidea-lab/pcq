"""PyTorch Lightning contract example.

Requires: uv add lightning torch
Run:      uv run python examples/contract_pytorch_lightning.py

Shows the pcq integration pattern for a `pl.LightningModule` + `Trainer`.
The trick is a tiny `PcqCallback` that hooks `on_validation_epoch_end`
and calls `pcq.log()` with the metrics Lightning has already aggregated.
The LightningModule stays free of pcq imports — `pcq.config()` and
`pcq.save_all()` bookend the run from the outside.
"""
from __future__ import annotations

import pcq


def main() -> None:
    cfg = pcq.config()
    out = pcq.output_dir()
    pcq.seed_everything(cfg.get("seed", 42))

    try:
        import lightning.pytorch as pl
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        pcq.save_all(
            history=[],
            status="failed",
            failure={
                "category": "missing_dependency",
                "message": str(e),
                "suggested_fix": "uv add lightning torch",
            },
        )
        raise

    in_dim = int(cfg.get("in_dim", 16))
    hidden = int(cfg.get("hidden_dim", 32))
    epochs = int(cfg.get("epochs", 2))
    lr = float(cfg.get("lr", 1e-3))
    batch_size = int(cfg.get("batch_size", 32))
    n_train = int(cfg.get("n_train", 256))
    n_eval = int(cfg.get("n_eval", 64))

    torch.manual_seed(cfg.get("seed", 42))

    # 합성 데이터셋
    w_true = torch.randn(in_dim)
    x_train = torch.randn(n_train, in_dim)
    y_train = (x_train @ w_true > 0).float().unsqueeze(1)
    x_eval = torch.randn(n_eval, in_dim)
    y_eval = (x_eval @ w_true > 0).float().unsqueeze(1)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(x_eval, y_eval), batch_size=batch_size)

    class MLP(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
            )
            self.loss_fn = nn.BCEWithLogitsLoss()

        def training_step(self, batch, _batch_idx):
            x, y = batch
            loss = self.loss_fn(self.net(x), y)
            self.log("train_loss", loss, on_epoch=True, prog_bar=False)
            return loss

        def validation_step(self, batch, _batch_idx):
            x, y = batch
            logits = self.net(x)
            loss = self.loss_fn(logits, y)
            acc = ((torch.sigmoid(logits) >= 0.5).float() == y).float().mean()
            self.log("val_loss", loss, on_epoch=True)
            self.log("val_acc", acc, on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=lr)

    history: list[dict] = []

    class PcqCallback(pl.Callback):
        """Lightning이 epoch 끝에서 집계한 메트릭을 pcq.log로 emit."""

        def on_validation_epoch_end(self, trainer, _pl_module):
            row = {
                "epoch": int(trainer.current_epoch),
                "train_loss": float(trainer.callback_metrics.get("train_loss", 0.0)),
                "val_loss": float(trainer.callback_metrics.get("val_loss", 0.0)),
                "val_acc": float(trainer.callback_metrics.get("val_acc", 0.0)),
            }
            history.append(row)
            pcq.log(**row)

    model = MLP()
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        devices=1,
        callbacks=[PcqCallback()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_loader, val_loader)

    ckpt_path = out / "lightning.ckpt"
    trainer.save_checkpoint(str(ckpt_path))

    pcq.save_all(
        history=history,
        artifacts={"model": "lightning.ckpt"},
    )


if __name__ == "__main__":
    main()
