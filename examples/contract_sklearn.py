"""sklearn contract script example — v1.13.

Requires: uv add scikit-learn joblib
Run:      CQ_CONFIG_JSON=examples/contract_sklearn_cfg.json \
          uv run python examples/contract_sklearn.py

Demonstrates 'no adapter requirement' — sklearn 코드를 pcq-compatible 하게
만드는 데 필요한 것은 pcq.config / pcq.log / pcq.save_all 호출 뿐.
"""
import pcq


def main() -> None:
    cfg = pcq.config()
    out = pcq.output_dir()
    pcq.seed_everything(cfg.get("seed", 42))

    try:
        import joblib
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        # 의존성 누락도 contract 에 따라 graceful 실패 보고
        pcq.save_all(
            history=[],
            status="failed",
            failure={
                "category": "missing_dependency",
                "message": str(e),
                "suggested_fix": "uv add scikit-learn joblib",
            },
        )
        raise

    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=cfg.get("seed", 42),
    )
    model = RandomForestClassifier(n_estimators=cfg.get("n_estimators", 100))
    model.fit(X_tr, y_tr)
    acc = float(model.score(X_te, y_te))

    # CQ Hub 가 stdout 에서 자동 파싱
    pcq.log(epoch=0, eval_acc=acc)
    joblib.dump(model, out / "model.pkl")

    pcq.save_all(
        history=[{"epoch": 0, "eval_acc": acc}],
        artifacts={"model": "model.pkl"},
    )


if __name__ == "__main__":
    main()
