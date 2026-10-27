from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn

from engine import evaluate
from models import n_params, n_trainable_params


def set_seed(seed: int = 42) -> None:
    #Co dinh seed toan cuc de dam bao tinh tai lap giua cac bai
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_class_distribution(classes: list[str], train_counts: list[int], test_counts: list[int]) -> None:
    #Truc quan hoa phan bo so luong anh theo tung lop cua vinafood21
    x = np.arange(len(classes))
    width = 0.42
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width / 2, train_counts, width, label=f"Train ({sum(train_counts):,} anh)", color="#2b6cb0")
    ax.bar(x + width / 2, test_counts, width, label=f"Test ({sum(test_counts):,} anh)", color="#ed8936")
    ax.set_xticks(x, classes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("So luong anh")
    ax.set_title("Phan bo 21 lop mon an trong bo du lieu vinaFood21", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def find_divergence_epoch(history: list[dict]) -> int:
    #xac dinh epoch ma duong train loss & val loss bat dau tach nhau (val loss dat day truoc khi tang/chung)
    if not history:
        return 1
    best_idx = int(np.argmin([h["val_loss"] for h in history]))
    return int(history[best_idx]["epoch"])


def plot_history(
    history: list[dict],
    title: str = "",
    save_path: str | Path | None = None,
    mark_divergence: bool = False,
) -> int | None:
    #Ve duong cong loss (train&val) va acc/val macrof1 theo epoch
    if not history:
        return None

    ep = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(ep, [h["train_loss"] for h in history], "b-o", markersize=4, label="train loss")
    ax1.plot(ep, [h["val_loss"] for h in history], "r--s", markersize=4, label="val loss")
    div_ep = None
    if mark_divergence:
        div_ep = find_divergence_epoch(history)
        ax1.axvline(div_ep, color="darkgreen", linestyle=":", linewidth=1.8, label=f"bat dau tach (ep {div_ep})")
    ax1.set_title(f"{title} ·Loss" if title else "Loss theo epoch", fontweight="bold")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("CrossEntropy loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(ep, [h["train_acc"] for h in history], "b-o", markersize=4, label="train acc")
    ax2.plot(ep, [h["val_acc"] for h in history], "g--^", markersize=4, label="val acc")
    ax2.plot(ep, [h["val_f1"] for h in history], "m-s", markersize=4, label="val macro-F1")
    ax2.set_title(f"{title} ·Accuracy & macro-F1" if title else "Accuracy & macro-F1", fontweight="bold")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    if mark_divergence and div_ep is not None:
        print(f"Nhan xet duong Loss ({title}): val_loss thap nhat tai epoch {div_ep}, "
              f"sau thoi diem nay hai duong train_loss va val_loss bat dau tach xa nhau")
    return div_ep


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    title: str = "Confusion Matrix (21x21)",
    save_path: str | Path | None = None,
) -> None:
    #Ve va luu confusion matrix 21x21 voi ten lop tu train_ds.classes
    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8.5)
    ax.set_yticks(range(len(classes)), classes, fontsize=8.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            if val > 0:
                ax.text(
                    j, i, str(val),
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=7,
                )

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def top_confused_pairs(cm: np.ndarray, classes: list[str], top_k: int = 5):
    #Trich xuat top k cap lop bi du doan nham nhieu nhat
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), classes[i], classes[j]))
    pairs.sort(reverse=True)
    return pairs[:top_k]


def evaluate_test(
    model: nn.Module,
    test_dl,
    classes: list[str],
    device,
    model_name: str,
    img_size: int,
    sec_per_epoch: float | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Danh gia day du tren tap test:
        - Bang accuracy, precision (macro), recall (macro), macro-F1 (do do chinh)
        - Bang F1 tung lop (classification_report 21 lop)
        - Confusion matrix 21x21 + top cap lop nham lan nhieu nhat
    """
    loss, acc, macro_f1, y_true, y_pred = evaluate(model, test_dl, nn.CrossEntropyLoss(), device)
    assert not model.training, "model.training phai la False trong khi danh gia!"

    label_ids = list(range(len(classes)))
    prec = precision_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    total_p = n_params(model)
    train_p = n_trainable_params(model)

    print(f"KET QUA TAP TEST {model_name}:")
    print(f"    - Kich thuoc anh: {img_size}x{img_size}")
    print(f"    - Tong so tham so: {total_p:,} (cap nhat: {train_p:,})")
    if sec_per_epoch is not None:
        print(f" Thoi gian / epoch : {sec_per_epoch:.2f}s")
    print(f"    - Test Loss         : {loss:.4f}")
    print(f"    - Accuracy          : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"    - Precision (macro) : {prec:.4f}")
    print(f"    - Recall (macro)    : {rec:.4f}")
    print(f"    - macro-F1 (chinh)  : {macro_f1:.4f}\n")

    print("Bang Precision, Recall, F1 tung lop (21 lop vinaFood21)")
    print(classification_report(
        y_true, y_pred, labels=label_ids, target_names=classes, digits=4, zero_division=0
    ))

    confused = top_confused_pairs(cm, classes, top_k=5)
    if confused:
        print("Top 5 cap mon an bi nham lan nhieu nhat (Nhan that -> Du doan)")
        for cnt, true_cls, pred_cls in confused:
            print(f"  {true_cls:<18} -> {pred_cls:<18} : {cnt} mau")
        print()

    cm_path = (Path(out_dir) / "confusion_matrix.png") if out_dir is not None else None
    plot_confusion_matrix(cm, classes, title=f"Confusion Matrix (21x21) · {model_name}", save_path=cm_path)

    return {
        "model": model_name,
        "resize": f"{img_size}x{img_size}",
        "params": total_p,
        "trainable_params": train_p,
        "sec_per_epoch": sec_per_epoch,
        "test_loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(classes, per_class_f1.tolist())),
        "cm": cm,
    }


def print_comparison_table(records: list[dict], title: str = "BANG SO SANH CAC MO HINH") -> None:
    #in bang tong hop ket qua
    print(f"=== {title} ===")
    header = (
        f"{'Mo hiinh':<32} | {'Resize':^9} | {'Tham so':>12} | "
        f"{'s/ep':>8} | {'Accuracy':>9} | {'Precision':>9} | {'Recall':>8} | {'macro-F1':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        sec_str = f"{r['sec_per_epoch']:.2f}s" if r.get("sec_per_epoch") is not None else "—"
        print(
            f"{r['model']:<32} | {r['resize']:^9} | {r['params']:>12,} | "
            f"{sec_str:>8} | {r['accuracy']:>9.4f} | {r['precision']:>9.4f} | "
            f"{r['recall']:>8.4f} | {r['macro_f1']:>9.4f}"
        )
    print()


def plot_model_comparison(records: list[dict], save_path: str | Path | None = None) -> None:
    #ve bieu do cot truc quan so sanh acc & macrof1 tren test set giua cac model
    if not records:
        return

    names = [r["model"] for r in records]
    accs = [r["accuracy"] for r in records]
    f1s = [r["macro_f1"] for r in records]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4299e1")
    bars2 = ax.bar(x + width / 2, f1s, width, label="macro-F1 (chinh)", color="#dd6b20")

    ax.set_xticks(x, names, rotation=20, ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("So sanh Accuracy va macro-F1 tren tap Test VinaFood21", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)

