"""
QUYẾT ĐỊNH TỰ ĐỘNG / CHUYỂN CẦN XÁC MINH (BA v15: NFR-09, NFR-10, mục 3.1).

Giai đoạn 1 — CHƯA có: BR-07 (mã R03–R10 từ ảnh), mục 3.4 (legal_label, chờ sổ tay đã duyệt).
  - conf_cnn = độ chắc chắn kém nhất trong 5 ô: min_ô max(p, 1-p)
  - conf_c45 = xác suất lá cao nhất của C4.5
  - leaf_n   = số mẫu train tại lá C4.5 (lá ít mẫu → không tin được confidence)
  - PENDING_REVIEW nếu conf_cnn < τ_cnn HOẶC conf_c45 < τ_c45 HOẶC leaf_n < leaf_min_n
  - τ, min_leaf, cf, leaf_min_n đều chọn trên OOF của tập Train (StratifiedGroupKFold theo group_id). Không dùng Test.
  - Ràng buộc: Recall ≥ 0.98, Coverage ≥ 0.80 (đề xuất, chưa phê duyệt), Selective Risk ≤ 0.02.
  Chọn cấu hình đạt theo Recall → Selective Risk → Coverage. Không đạt: ghi nhận, không hạ ngưỡng.
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


def is_pending(conf_cnn, conf_c45, tau_cnn, tau_c45, leaf_n=None, leaf_min_n=0):
    p = (np.asarray(conf_cnn) < tau_cnn) | (np.asarray(conf_c45) < tau_c45)
    if leaf_n is not None:
        p = p | (np.asarray(leaf_n) < leaf_min_n)
    return p


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
        "selective_risk": round(float(wrong_auto.sum()) / n_auto, 4) if n_auto > 0 else 1.0,
        "n": int(len(y)), "n_auto": n_auto, "n_pending": int(pending.sum()),
    }


def feasible(r):
    """Đạt nghiệm thu: Recall ≥ 0.98 (NFR-10), Coverage ≥ coverage_min, Selective Risk ≤ selective_risk_max."""
    d = CFG["decision"]
    return (r["invalid_screening_recall"] >= d["nfr10_target"]
            and r["coverage"] >= d["coverage_min"]
            and r["selective_risk"] <= d["selective_risk_max"])


def rank(r):
    """
    Cấu hình đạt: ưu tiên Recall → Selective Risk thấp → Coverage cao.
    Cấu hình không đạt: ít vi phạm ràng buộc hơn được xếp trên. Không hạ ngưỡng để tăng Coverage.
    """
    d = CFG["decision"]
    if feasible(r):
        return (2, r["invalid_screening_recall"], -r["selective_risk"], r["coverage"])
    viol = (r["invalid_screening_recall"] < d["nfr10_target"]) + (r["coverage"] < d["coverage_min"]) \
        + (r["selective_risk"] > d["selective_risk_max"])
    return (0, -viol, r["invalid_screening_recall"], r["coverage"])


def select_tau(y, pred, conf_cnn, conf_c45, leaf_n=None, leaf_min_n=0, step=None):
    """Lưới (τ_cnn, τ_c45) trên OOF của Train, chọn theo rank()."""
    step = CFG["decision"]["tau_step"] if step is None else step
    grid_cnn = np.round(np.arange(0.5, 1.0 + 1e-9, step), 4)
    grid_c45 = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    best = None
    for tc in grid_cnn:
        for td in grid_c45:
            r = screening_report(y, pred, is_pending(conf_cnn, conf_c45, tc, td, leaf_n, leaf_min_n))
            if best is None or rank(r) > rank(best[2]):
                best = (tc, td, r)
    tc, td, r = best
    ok = feasible(r)
    return {"tau_cnn": float(tc), "tau_c45": float(td), "leaf_min_n": leaf_min_n,
            "tier": 2 if ok else 0, "target_met_oof": ok, "oof_target_met": ok, "oof": r}


def oof_c45(H, y, groups, hp, k=None, seed=42):
    """C4.5 out-of-fold trên đặc trưng lai H, chia theo group_id. Trả về (xác suất lá, số mẫu lá)."""
    k = CFG["decision"]["cv_folds"] if k is None else k
    proba = np.zeros((len(y), len(CFG["labels"])))
    leaf_n = np.zeros(len(y), dtype=int)
    for tr_i, va_i in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(H, y, groups):
        clf = C45Classifier(max_depth=CFG["c45"]["max_depth"], min_leaf=hp["min_leaf"], cf=hp["cf"])
        clf.fit(H[tr_i], y[tr_i], class_names=CFG["labels"])
        proba[va_i] = clf.predict_proba(H[va_i])
        leaf_n[va_i] = clf.leaf_size(H[va_i])
    return proba, leaf_n


def grid_search(H, y, groups, p_marks, grid=None):
    """Lưới min_leaf × cf × leaf_min_n (chọn trên OOF Train). Trả về cấu hình tốt nhất + τ."""
    grid = grid or CFG["decision"]
    best = None
    for min_leaf in grid["grid_min_leaf"]:
        for cf in grid["grid_cf"]:
            proba, leaf_n = oof_c45(H, y, groups, {"min_leaf": min_leaf, "cf": cf})
            cc, cd = confidences(p_marks, proba)
            for lmn in grid["grid_leaf_min_n"]:
                t = select_tau(y, proba.argmax(1), cc, cd, leaf_n=leaf_n, leaf_min_n=lmn)
                t.update(min_leaf=min_leaf, cf=cf)
                print(f"[grid] min_leaf={min_leaf} cf={cf} leaf_min_n={lmn} | "
                      f"τ=({t['tau_cnn']},{t['tau_c45']}) tier={t['tier']} oof={t['oof']}")
                if best is None or rank(t["oof"]) > rank(best["oof"]):
                    best = t
    return best


# ---------------------------------------------------------------------------
# ĐỀ XUẤT PHÁP LÝ cho Tổ bầu cử (BA v15: BR-01, BR-06, BR-07b, P-01)
# Hệ thống chỉ ĐỀ XUẤT. Quyết định cuối cùng do Tổ bầu cử ghi nhận (FR-09).
# Theo quyết định của nhóm: BLANK được tính là KHÔNG HỢP LỆ (mọi nhãn khác `valid`).
# Lưu ý: điều này khác BR-06 hiện hành ("không mặc định là không hợp lệ về pháp lý"); BA cần sửa BR-06.
# ---------------------------------------------------------------------------
PROPOSED_VALID = "PROPOSED_LEGAL_VALID"
PROPOSED_INVALID = "PROPOSED_LEGAL_INVALID"
PENDING = "PENDING_REVIEW"


def propose_legal(pred, pending):
    """Trả về trạng thái đề xuất cho từng phiếu. pred: chỉ số lớp (0 = valid, 1 = blank, ...)."""
    pred, pending = np.asarray(pred), np.asarray(pending, dtype=bool)
    out = np.where(pred == VALID_IDX, PROPOSED_VALID, PROPOSED_INVALID).astype(object)
    out[pending] = PENDING
    return out
