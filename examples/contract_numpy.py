"""Framework-neutral contract script example using NumPy only.

Requires no pcq Trainer, Experiment, atom, or framework adapter. The training
code owns the workflow; pcq only provides config, logging, output directory,
and standard artifact finalization.
"""
from __future__ import annotations

import json

import numpy as np

import pcq


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    cfg = pcq.config()
    out = pcq.output_dir()
    seed = int(cfg.get("seed", 42))
    epochs = int(cfg.get("epochs", 3))
    lr = float(cfg.get("lr", 0.2))
    rng = np.random.default_rng(seed)

    x_train = rng.normal(size=(64, 4))
    x_eval = rng.normal(size=(32, 4))
    true_w = np.array([0.7, -1.2, 0.4, 0.9])
    y_train = (x_train @ true_w > 0).astype(float)
    y_eval = (x_eval @ true_w > 0).astype(float)

    w = rng.normal(scale=0.1, size=4)
    history: list[dict] = []
    for epoch in range(epochs):
        train_prob = _sigmoid(x_train @ w)
        eps = 1e-8
        train_loss = float(
            -np.mean(
                y_train * np.log(train_prob + eps)
                + (1.0 - y_train) * np.log(1.0 - train_prob + eps)
            )
        )
        grad = x_train.T @ (train_prob - y_train) / len(x_train)
        w -= lr * grad

        eval_prob = _sigmoid(x_eval @ w)
        eval_acc = float(((eval_prob >= 0.5) == y_eval).mean())
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_acc": eval_acc,
        }
        history.append(row)
        pcq.log(**row)

    np.savez(out / "model.npz", weights=w)
    (out / "framework_result.json").write_text(
        json.dumps({
            "framework": "numpy",
            "epochs": epochs,
            "eval_acc": history[-1]["eval_acc"],
        }),
        encoding="utf-8",
    )
    pcq.save_all(
        history=history,
        artifacts={
            "model": "model.npz",
            "framework_result": "framework_result.json",
        },
    )


if __name__ == "__main__":
    main()
