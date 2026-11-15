"""
QUYẾT ĐỊNH TỰ ĐỘNG / CHUYỂN CẦN XÁC MINH (BA v15: NFR-09, NFR-10, mục 3.1).

Giai đoạn 1 — CHƯA có: BR-07 (mã R03–R10 từ ảnh), mục 3.4 (legal_label, chờ sổ tay đã duyệt).
  - conf_cnn = độ chắc chắn kém nhất trong 5 ô: min_ô max(p, 1-p)
  - conf_c45 = xác suất lá cao nhất của C4.5
  - OR rule: PENDING_REVIEW nếu conf_cnn < τ_cnn HOẶC conf_c45 < τ_c45
  - τ chọn trên OOF của tập Train (StratifiedGroupKFold theo group_id). Không dùng Test.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from c45 import C45Classifier
from config import CFG

VALID_IDX = 0   # vị trí "valid" trong CFG["labels"]


def confidences(p_marks, proba):
    conf_cnn = np.maximum(p_marks, 1.0 - p_marks).min(axis=1)
    conf_c45 = np.asarray(proba).max(axis=1)
    return conf_cnn, conf_c45


def is_pending(conf_cnn, conf_c45, tau_cnn, tau_c45):
    return (np.asarray(conf_cnn) < tau_cnn) | (np.asarray(conf_c45) < tau_c45)


def screening_report(y, pred, pending):
    """Mục 3.1. pending: True = PENDING_REVIEW. Nhãn dự đoán `pred`, nhãn thật `y` (chỉ số lớp)."""
    y, pred, pending = np.asarray(y), np.asarray(pred), np.asarray(pending, dtype=bool)
    inv = y != VALID_IDX
    n_inv = max(int(inv.sum()), 1)
    auto = ~pending
    n_auto = int(auto.sum())
    flagged = inv & (pending | (pred != VALID_IDX))            # phiếu không hợp lệ được phát hiện (kể cả chờ xác minh)
    auto_inv = inv & auto & (pred != VALID_IDX)                # tự động kết luận không hợp lệ
    wrong_auto = auto & (pred != y)
    return {
        "invalid_screening_recall": round(float(flagged.sum()) / n_inv, 4),
        "automatic_invalid_recall": round(float(auto_inv.sum()) / n_inv, 4),
        "coverage": round(float(auto.mean()), 4),
        "referral_rate": round(float(pending.mean()), 4),
        "selective_risk": round(float(wrong_auto.sum()) / max(n_auto, 1), 4),
        "n": int(len(y)), "n_auto": n_auto, "n_pending": int(pending.sum()),
    }


def select_tau(y, pred, conf_cnn, conf_c45, target=None, step=None):
    """
    Lưới (τ_cnn, τ_c45) trên OOF của Train.
    Ưu tiên: Invalid Screening Recall ≥ target, rồi Coverage cao nhất.
    Không đạt target với mọi τ: chọn τ có Invalid Screening Recall cao nhất, rồi Coverage.
    """
    target = CFG["decision"]["nfr10_target"] if target is None else target
    step = CFG["decision"]["tau_step"] if step is None else step
    grid_cnn = np.round(np.arange(0.5, 1.0 + 1e-9, step), 4)
    grid_c45 = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    cands = []
    for tc in grid_cnn:
        for td in grid_c45:
            r = screening_report(y, pred, is_pending(conf_cnn, conf_c45, tc, td))
            cands.append((tc, td, r))
    feasible = [c for c in cands if c[2]["invalid_screening_recall"] >= target]
    if feasible:
        tc, td, r = max(feasible, key=lambda c: (c[2]["coverage"], -c[0], -c[1]))
    else:
        tc, td, r = max(cands, key=lambda c: (c[2]["invalid_screening_recall"], c[2]["coverage"]))
    return {"tau_cnn": float(tc), "tau_c45": float(td), "target_met_oof": bool(feasible), "oof": r}


def oof_c45_proba(H, y, groups, k=None, seed=42):
    """C4.5 out-of-fold trên đặc trưng lai H, chia theo group_id."""
    k = CFG["decision"]["cv_folds"] if k is None else k
    hp = CFG["c45"]
    n_cls = len(CFG["labels"])
    proba = np.zeros((len(y), n_cls))
    for tr_i, va_i in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(H, y, groups):
        clf = C45Classifier(max_depth=hp["max_depth"], min_leaf=hp["min_leaf"], cf=hp["cf"])
        clf.fit(H[tr_i], y[tr_i], class_names=CFG["labels"])
        proba[va_i] = clf.predict_proba(H[va_i])
    return proba
