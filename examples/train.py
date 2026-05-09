"""pcq smoke experiment — contract script (v4.0).

CQ_CONFIG_JSON 을 읽어 minimal contract artifacts 를 생성한다.
Trainer 등 framework code 없음 — contract surface 만 검증.
산출물은 cfg["output_dir"] (기본값: 'output/') 에 저장된다.
"""
import random

import pcq

cfg = pcq.config()
pcq.seed_everything(cfg.get("seed", 42))

# 합성 metric history — 실제 학습 없이 contract 만 시연.
epochs = int(cfg.get("epochs", 2))
history = []
for epoch in range(epochs):
    train_loss = 1.0 - 0.3 * epoch + random.random() * 0.05
    train_acc = 0.5 + 0.2 * epoch + random.random() * 0.05
    eval_loss = 1.1 - 0.3 * epoch + random.random() * 0.05
    eval_acc = 0.45 + 0.2 * epoch + random.random() * 0.05
    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
    })
    pcq.log(
        epoch=epoch,
        train_loss=train_loss,
        train_acc=train_acc,
        eval_loss=eval_loss,
        eval_acc=eval_acc,
    )

pcq.save_all(history=history, status="completed")
