"""
ĐÁNH GIÁ: báo cáo 5 lớp + bài toán nhị phân hợp lệ/không hợp lệ + ma trận nhầm lẫn.
Với bầu cử, Recall của lớp "không hợp lệ" là chỉ số quan trọng nhất (bỏ sót phiếu sai nghiêm trọng hơn).
"""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import CFG

VALID_IDX = 0   # vị trí của "valid" trong CFG["labels"]


def summarize(name, y_true, y_pred):
    labels = CFG["labels"]
    K = len(labels)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=range(K))

    inv_t = y_true != VALID_IDX
    inv_p = y_pred != VALID_IDX
    tp = int((inv_t & inv_p).sum())
    inv_recall = tp / max(int(inv_t.sum()), 1)
    inv_precision = tp / max(int(inv_p.sum()), 1)

    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred, labels=range(K), target_names=labels, zero_division=0))
    print("Ma trận nhầm lẫn (hàng = thực tế, cột = dự đoán):")
    print(" " * 12 + " ".join(f"{l[:9]:>9s}" for l in labels))
    for i, row in enumerate(cm):
        print(f"{labels[i]:>12s}" + " ".join(f"{v:>9d}" for v in row))
    print(f"Hợp lệ/không hợp lệ: Recall(không hợp lệ)={inv_recall:.4f}  "
          f"Precision(không hợp lệ)={inv_precision:.4f}")
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "invalid_recall": round(inv_recall, 4),
        "invalid_precision": round(inv_precision, 4),
        "confusion_matrix": cm.tolist(),
    }


def clopper_pearson_upper(k, n, conf=0.95):
    """Cận trên Clopper–Pearson MỘT PHÍA. Chốt trong BA v15 mục 3.4 (NFR-12)."""
    from scipy.stats import beta
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))
