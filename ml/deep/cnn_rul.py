"""1D-CNN RUL regressor on 30-cycle sensor windows (optional benchmark; needs torch).

Same protocol as the tree models: target is RUL capped at 125, selection uses
engine-wise validation (20 training engines held out for early stopping), and the
official test engines are used once, for the final number at each engine's last
cycle. Short histories are left-padded by repeating the first reading.

Run:  python -m ml.deep.cnn_rul          (needs: pip install -r requirements-deep.txt)
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.data.cmapss import DEFAULT_DIR, RUL_CAP, SENSORS, download, load_test, load_train

WINDOW = 30
SEED = 42
RESULTS_PATH = Path("evaluation/results/deep_rul.json")
MODEL_PATH = Path("models/cnn_rul.pt")


def nasa_score(y_true, y_pred) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def make_windows(df, window: int = WINDOW, last_only: bool = False):
    """(n, window, n_sensors) float32 windows ending at each cycle (or only each
    engine's last cycle), the capped-RUL labels, and each window's engine id."""
    X, y, ids = [], [], []
    for engine_id, g in df.groupby("engine_id"):
        values = g.sort_values("cycle")[SENSORS].to_numpy(dtype=np.float32)
        labels = g.sort_values("cycle")["rul_capped"].to_numpy(dtype=np.float32)
        padded = np.vstack([np.repeat(values[:1], window - 1, axis=0), values])
        ends = [len(values) - 1] if last_only else range(len(values))
        for i in ends:
            X.append(padded[i:i + window])
            y.append(labels[i])
            ids.append(engine_id)
    return np.stack(X), np.asarray(y, dtype=np.float32), np.asarray(ids)


class CNN(nn.Module):
    def __init__(self, n_sensors: int = len(SENSORS), window: int = WINDOW):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_sensors, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Flatten(),  # keep the time axis: where in the window the change happens matters
            nn.Linear(128 * window, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1),
        )

    def forward(self, x):  # x: (batch, window, sensors)
        return self.net(x.transpose(1, 2)).squeeze(-1)


def _predict(model, X, mean, std) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy((X - mean) / std)).numpy()
    return np.clip(out, 0, RUL_CAP)


def train(df_train, epochs: int = 120, patience: int = 15, val_engines: int = 20, batch: int = 256):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    engines = df_train["engine_id"].unique()
    val_ids = set(rng.choice(engines, size=min(val_engines, max(1, len(engines) // 5)), replace=False))
    tr, va = df_train[~df_train["engine_id"].isin(val_ids)], df_train[df_train["engine_id"].isin(val_ids)]
    X_tr, y_tr, _ = make_windows(tr)
    X_va, y_va, _ = make_windows(va)
    mean, std = X_tr.mean(axis=(0, 1)), X_tr.std(axis=(0, 1)) + 1e-6

    model, loss_fn = CNN(), nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    Xt, yt = torch.from_numpy((X_tr - mean) / std), torch.from_numpy(y_tr)
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        val_rmse = float(np.sqrt(np.mean((_predict(model, X_va, mean, std) - y_va) ** 2)))
        sched.step(val_rmse)
        if val_rmse < best:
            best, best_state, stale = val_rmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, mean, std, {"val_rmse": best, "epochs_run": epoch + 1, "val_engines": len(val_ids)}


def main():
    data_dir = download(DEFAULT_DIR)
    model, mean, std, info = train(load_train(data_dir))
    X, y, _ = make_windows(load_test(data_dir), last_only=True)
    pred = _predict(model, X, mean, std)
    result = {"model": "1D-CNN (30-cycle windows)", "protocol": "FD001 official test engines, last cycle, capped RUL",
              "rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "mae": float(np.mean(np.abs(pred - y))),
              "nasa_score": nasa_score(y, pred), **info}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std}, MODEL_PATH)
    RESULTS_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
