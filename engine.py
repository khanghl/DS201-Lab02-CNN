import copy
import csv
from pathlib import Path
import time

from sklearn.metrics import f1_score
import torch
import torch.nn as nn


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() #bat dropout, BN dung thong ke batch
    total, correct, running = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return running / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval() #tat dropout, BN dung thong ke tich luy 
    total, correct, running = 0, 0, 0.0
    y_true, y_pred = [], []
    n_cls = None
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if n_cls is None:
            n_cls = logits.shape[1]
        running += criterion(logits, yb).item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        y_true += yb.cpu().tolist()
        y_pred += pred.cpu().tolist()

    labels = list(range(n_cls)) if n_cls is not None else None
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return running / total, correct / total, macro_f1, y_true, y_pred


def fit(
    model,
    train_dl,
    val_dl,
    epochs: int,
    lr: float,
    device,
    out_dir: str | Path,
    classes: list[str],
    img_size: int,
    criterion=None,
    optimizer=None,
):
    """Vong lap huan luyen:
        - Optimizer chi nhan cac tham so co requires_grad=True
        - Do thoi gian thuc thi moi epoch (giay/epoch)
        - Giu lai checkpoint co macro-F1 tren tap val cao nhat (best.pt) và luu history.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(trainable, lr=lr)

    history = []
    best_f1, best_epoch = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    ckpt_path = out_dir / "best.pt"

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_dl, criterion, device)
        ep_sec = time.perf_counter() - t0

        history.append(
            dict(
                epoch=ep,
                train_loss=tr_loss,
                train_acc=tr_acc,
                val_loss=va_loss,
                val_acc=va_acc,
                val_f1=va_f1,
                sec=ep_sec,
            )
        )

        mark = ""
        if va_f1 > best_f1: #giu checkpoint co macro-F1 tren val cao nhat
            best_f1 = va_f1
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model": best_state,
                    "classes": classes,
                    "img_size": img_size,
                    "epoch": ep,
                    "val_f1": va_f1,
                },
                ckpt_path,
            )
            mark = "  *"

        print(
            f"epoch {ep:>2}/{epochs:<2}  train {tr_loss:.4f}/{tr_acc:.4f}"
            f"   val {va_loss:.4f}/{va_acc:.4f}  F1 {va_f1:.4f}"
            f"   ({ep_sec:.1f}s){mark}"
        )

    if history:
        with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    avg_sec = sum(h["sec"] for h in history) / max(len(history), 1)
    model.load_state_dict(best_state)
    print(
        f"-> best val macro-F1: {best_f1:.4f} (epoch {best_epoch}) | "
        f"trung binh: {avg_sec:.2f}s/epoch | da luu: {ckpt_path}\n"
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "sec_per_epoch": avg_sec,
        "ckpt_path": str(ckpt_path),
    }
    return history, summary
