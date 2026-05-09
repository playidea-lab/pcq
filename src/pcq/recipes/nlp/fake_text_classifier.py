"""Recipe: tokenizer-free text classifier (NLP smoke).

v1.9: RecipeSpec metadata-first 형태. 외부 dep 0, smoke_safe=True.

NLP 카탈로그의 minimal 예시. 실제 NLP 는 pcq[nlp] extras 로 transformers
사용 시 BERT-class recipe 추가 (v2 roadmap).
"""
from __future__ import annotations

import pcq
from pcq.agent.schema import RecipeSpec


SPEC = RecipeSpec(
    name="nlp/fake_text_classifier",
    task="text_classification",
    description=(
        "Tokenizer-free fake-text classifier — torch-only NLP smoke."
    ),
    metrics=["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc"],
    monitor_candidates=[
        {"name": "eval_acc", "mode": "max"},
        {"name": "eval_loss", "mode": "min"},
    ],
    requires_extras=[],
    smoke_safe=True,
    atoms={
        "model": pcq.model_ref(
            "text_classifier",
            {"vocab_size": 1000, "embed_dim": 64, "num_classes": 2},
        ),
        "dataset_train": pcq.dataset_ref(
            "fake_text", {"num_samples": 128, "num_classes": 2},
        ),
        "dataset_eval": pcq.dataset_ref(
            "fake_text", {"num_samples": 64, "num_classes": 2},
        ),
        "loss": pcq.loss_ref("cross_entropy"),
        "optim": pcq.optim_ref("adamw", {"lr": 1e-3}),
    },
    defaults={"epochs": 3, "batch_size": 32},
)


def fake_text_classifier() -> dict:
    """Tokenizer-free 텍스트 분류 smoke recipe — 외부 dep 0."""
    return SPEC.build()
