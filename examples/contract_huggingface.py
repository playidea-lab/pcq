"""HuggingFace Trainer contract example.

Requires: uv add transformers datasets torch
Run:      uv run python examples/contract_huggingface.py

Shows the pcq integration pattern for `transformers.Trainer`. The
trick is a tiny `TrainerCallback` that hooks `on_evaluate` and calls
`pcq.log()` with whatever Trainer emits in `metrics` at the end of an
evaluation epoch. Model, tokenizer, and TrainingArguments stay
framework-idiomatic — pcq only sits at the boundary.

A very small pretrained model is used by default
(`sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english`,
~5MB) so first-time download is quick. Override with `model_name`
config to point at any HF Hub model.
"""
from __future__ import annotations

import pcq


def main() -> None:
    cfg = pcq.config()
    out = pcq.output_dir()
    pcq.seed_everything(cfg.get("seed", 42))

    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as e:
        pcq.save_all(
            history=[],
            status="failed",
            failure={
                "category": "missing_dependency",
                "message": str(e),
                "suggested_fix": "uv add transformers datasets torch",
            },
        )
        raise

    model_name = cfg.get(
        "model_name", "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"
    )
    epochs = int(cfg.get("epochs", 1))
    lr = float(cfg.get("lr", 5e-5))
    batch_size = int(cfg.get("batch_size", 8))
    n_train = int(cfg.get("n_train", 32))
    n_eval = int(cfg.get("n_eval", 16))
    seed = int(cfg.get("seed", 42))

    torch.manual_seed(seed)

    # 합성 binary text classification — datasets.Dataset.from_dict
    pos = ["this movie was great", "i loved the acting", "wonderful experience"]
    neg = ["terrible film", "wasted my time", "boring and slow"]
    train_texts, train_labels = [], []
    for i in range(n_train):
        if i % 2 == 0:
            train_texts.append(pos[i % len(pos)])
            train_labels.append(1)
        else:
            train_texts.append(neg[i % len(neg)])
            train_labels.append(0)
    eval_texts = ["amazing story", "horrible plot", "great direction", "bad ending"]
    eval_labels = [1, 0, 1, 0]
    while len(eval_texts) < n_eval:
        eval_texts.append("good" if len(eval_texts) % 2 == 0 else "bad")
        eval_labels.append(1 if eval_texts[-1] == "good" else 0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=32)

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(
        tok, batched=True
    )
    eval_ds = Dataset.from_dict({"text": eval_texts, "label": eval_labels}).map(
        tok, batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir=str(out / "hf_trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        seed=seed,
        disable_tqdm=True,
    )

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        acc = float((preds == pred.label_ids).mean())
        return {"accuracy": acc}

    history: list[dict] = []

    class PcqCallback(TrainerCallback):
        """Trainer가 epoch 끝에 evaluate 끝낸 시점에 pcq.log 호출."""

        def on_evaluate(self, _args, state, _control, metrics=None, **_kwargs):
            row = {
                "epoch": int(state.epoch or 0),
                "eval_loss": float(metrics.get("eval_loss", 0.0)) if metrics else 0.0,
                "eval_accuracy": float(metrics.get("eval_accuracy", 0.0)) if metrics else 0.0,
            }
            history.append(row)
            pcq.log(**row)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[PcqCallback()],
    )
    trainer.train()

    # 최종 모델 저장 — save_strategy=no 였으므로 명시적으로
    model.save_pretrained(out / "model")
    tokenizer.save_pretrained(out / "model")

    pcq.save_all(
        history=history,
        artifacts={"model": "model"},
    )


if __name__ == "__main__":
    main()
