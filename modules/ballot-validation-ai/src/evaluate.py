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
    """
    Cận trên Clopper–Pearson MỘT PHÍA, công thức exact: beta.ppf(conf, k+1, n-k).
    Hàm DUY NHẤT trong dự án (decision.py dùng lại). n ≤ 0 → None (mẫu số bằng 0, hiển thị N/A).
    k ≥ n → 1.0.
    """
    from scipy.stats import beta
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def _ratio(num, den):
    return None if den <= 0 else num / den


def invalid_to_valid_report(y, pred, pending, labels=None):
    """
    Invalid → Valid Auto. KHÔNG nhầm với Selective Risk.
      N_inv_to_valid = #(nhãn thật ≠ valid, AUTO_ACCEPT, pred = valid)
      R1 = N_inv_to_valid / N_invalid        (mẫu số: tổng phiếu không hợp lệ)
      R2 = N_inv_to_valid / N_AUTO_ACCEPT    (mẫu số: tổng phiếu tự động chấp nhận)
    Theo nhãn k: R1_k = accept_k / n_k, R2_k = accept_k / N_AUTO_ACCEPT.
    Nhãn loại trừ lẫn nhau nên Σ_k accept_k = N_inv_to_valid → Σ R2_k = R2.
    Tổng R1 KHÔNG được tính bằng Σ R1_k (mẫu số khác nhau).
    """
    labels = labels or CFG["labels"]
    y, pred, pend = np.asarray(y), np.asarray(pred), np.asarray(pending, dtype=bool)
    accept = (~pend) & (pred == VALID_IDX)
    inv = y != VALID_IDX
    n_inv, n_acc = int(inv.sum()), int(accept.sum())
    k = int((inv & accept).sum())
    per = {}
    for idx, name in enumerate(labels):
        if idx == VALID_IDX:
            continue
        m = y == idx
        n_k, ka = int(m.sum()), int((m & accept).sum())
        per[name] = {"n": n_k, "auto_accept": ka,
                     "R1": _ratio(ka, n_k), "CP_upper_R1": clopper_pearson_upper(ka, n_k),
                     "R2": _ratio(ka, n_acc), "CP_upper_R2": clopper_pearson_upper(ka, n_acc)}
    return {"N_inv_to_valid": k, "N_invalid": n_inv, "N_AUTO_ACCEPT": n_acc,
            "R1": _ratio(k, n_inv), "CP_upper_R1": clopper_pearson_upper(k, n_inv),
            "R2": _ratio(k, n_acc), "CP_upper_R2": clopper_pearson_upper(k, n_acc),
            "sum_by_label_equals_total": sum(v["auto_accept"] for v in per.values()) == k,
            "by_label": per}
