"""pcq smoke experiment.

CQ_CONFIG_JSON 을 읽어 Trainer 로 fake 데이터셋 + mlp 모델을 학습한다.
산출물은 cfg["output_dir"] (기본값: 'output/') 에 저장된다.
"""
import pcq

cfg = pcq.config()
pcq.seed_everything(cfg["seed"])

trainer = pcq.Trainer(
    task="classification",
    dataset="fake",
    model="mlp",
    cfg=cfg,
)
trainer.fit()
