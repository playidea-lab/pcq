"""Stub training script — never invoked under --config-only.

Conformance tests run pcq with --config-only, which materializes
runtime_cfg.json and emits the envelope without executing cq.yaml.cmd.
This file exists only so cq.yaml resolves to a real file.
"""
import pcq


def main() -> None:
    cfg = pcq.config()
    pcq.log(epoch=0, eval_acc=1.0)
    pcq.save_all(history=[{"epoch": 0, "eval_acc": 1.0}])


if __name__ == "__main__":
    main()
