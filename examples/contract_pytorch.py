"""PyTorch contract example.

Requires: uv add 'pcq[torch]' or `uv add torch`.
Run:      uv run python examples/contract_pytorch.py

Shows the pcq integration pattern for a raw `torch.nn` training loop —
no adapter, no Trainer subclass. Three pcq calls (config / log /
save_all) sit at the boundary while the model, optimizer, loss, and
data loop all stay user-owned.
"""
import pcq


def main() -> None:
    cfg = pcq.config()
    out = pcq.output_dir()
    pcq.seed_everything(cfg.get("seed", 42))

    try:
        import torch
        from torch import nn
    except ImportError as e:
        pcq.save_all(
            history=[],
            status="failed",
            failure={
                "category": "missing_dependency",
                "message": str(e),
                "suggested_fix": "uv add torch",
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

    # 합성 binary classification — torch.randn으로 input, sign에 따라 label.
    w_true = torch.randn(in_dim)
    x_train = torch.randn(n_train, in_dim)
    y_train = (x_train @ w_true > 0).float().unsqueeze(1)
    x_eval = torch.randn(n_eval, in_dim)
    y_eval = (x_eval @ w_true > 0).float().unsqueeze(1)

    model = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        # 미니배치 루프
        perm = torch.randperm(n_train)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(idx)
        train_loss = total_loss / n_train

        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(x_eval)) >= 0.5).float()
            eval_acc = float((pred == y_eval).float().mean())

        row = {"epoch": epoch, "train_loss": train_loss, "eval_acc": eval_acc}
        history.append(row)
        pcq.log(**row)

    torch.save(model.state_dict(), out / "model.pt")
    pcq.save_all(
        history=history,
        artifacts={"model": "model.pt"},
    )


if __name__ == "__main__":
    main()
