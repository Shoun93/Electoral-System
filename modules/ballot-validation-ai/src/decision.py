"""
QUYẾT ĐỊNH TỰ ĐỘNG / CHUYỂN CẦN XÁC MINH (BA v15: NFR-09, NFR-10, mục 3.1).

Giai đoạn 1 — CHƯA có: BR-07 (mã R03–R10 từ ảnh), mục 3.4 (legal_label, chờ sổ tay đã duyệt).
  - conf_cnn = độ chắc chắn kém nhất trong 5 ô: min_ô max(p, 1-p), p = sigmoid(logit / T)
  - conf_c45 = xác suất lá cao nhất của C4.5
  - leaf_n   = số mẫu train tại lá C4.5 (lá ít mẫu → không tin được confidence)
               OOF: nhân k/(k-1) để cùng thang với mô hình fit trên toàn Train
  - PENDING_REVIEW nếu conf_cnn < τ_cnn HOẶC conf_c45 < τ_c45 HOẶC leaf_n < leaf_min_n
  - τ, min_leaf, cf, leaf_min_n, T đều chọn/fit trên OOF của Train (StratifiedGroupKFold theo group_id). Không dùng Test.
  - Ràng buộc: Recall ≥ 0.98, Coverage ≥ 0.80 (đề xuất, chưa phê duyệt), Selective Risk ≤ 0.02.
  Chọn cấu hình đạt theo Recall → Selective Risk → Coverage. Không đạt: xếp theo mức vi phạm, ghi nhận, không hạ ngưỡng.
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


def _counts(y, pred, pending):
    y, pred, pending = np.asarray(y), np.asarray(pred), np.asarray(pending, dtype=bool)
    inv = y != VALID_IDX
    auto = ~pending
    return {
        "n": int(len(y)), "n_inv": int(inv.sum()), "n_auto": int(auto.sum()), "n_pending": int(pending.sum()),
        # phiếu không hợp lệ được phát hiện (kể cả chờ xác minh)
        "flagged": int((inv & (pending | (pred != VALID_IDX))).sum()),
        # tự động kết luận không hợp lệ
        "auto_inv": int((inv & auto & (pred != VALID_IDX)).sum()),
        "wrong_auto": int((auto & (pred != y)).sum()),
    }


def screening_report(y, pred, pending):
    """Mục 3.1. pending: True = PENDING_REVIEW. Nhãn dự đoán `pred`, nhãn thật `y` (chỉ số lớp)."""
    c = _counts(y, pred, pending)
    n_inv = max(c["n_inv"], 1)
    return {
        "invalid_screening_recall": round(c["flagged"] / n_inv, 4),
        "automatic_invalid_recall": round(c["auto_inv"] / n_inv, 4),
        "coverage": round(c["n_auto"] / c["n"], 4),
        "referral_rate": round(c["n_pending"] / c["n"], 4),
        "selective_risk": round(c["wrong_auto"] / c["n_auto"], 4) if c["n_auto"] > 0 else 1.0,
        "n": c["n"], "n_auto": c["n_auto"], "n_pending": c["n_pending"],
    }


def cp_lower(x, n, alpha=0.05):
    """Clopper–Pearson cận dưới một phía, mức (1 - alpha)."""
    from scipy.stats import beta
    return 0.0 if x <= 0 else float(beta.ppf(alpha, x, n - x + 1))


def cp_upper(x, n, alpha=0.05):
    """Clopper–Pearson cận trên một phía, mức (1 - alpha)."""
    from scipy.stats import beta
    return 1.0 if x >= n else float(beta.ppf(1 - alpha, x + 1, n - x))


def ci_report(y, pred, pending, alpha=0.05):
    """Khoảng tin cậy 95% một phía (Clopper–Pearson) cho Invalid Screening Recall và Selective Risk."""
    c = _counts(y, pred, pending)
    return {
        "invalid_screening_recall_lower95": round(cp_lower(c["flagged"], c["n_inv"], alpha), 4),
        "selective_risk_upper95": round(cp_upper(c["wrong_auto"], c["n_auto"], alpha), 4) if c["n_auto"] > 0 else 1.0,
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
    Cấu hình không đạt: ít vi phạm hơn xếp trên; cùng số vi phạm thì tổng mức vượt ngưỡng nhỏ hơn xếp trên.
    Không hạ ngưỡng để tăng Coverage.
    """
    d = CFG["decision"]
    if feasible(r):
        return (2, r["invalid_screening_recall"], -r["selective_risk"], r["coverage"])
    gap_recall = max(0.0, d["nfr10_target"] - r["invalid_screening_recall"])
    gap_cov = max(0.0, d["coverage_min"] - r["coverage"])
    gap_risk = max(0.0, r["selective_risk"] - d["selective_risk_max"])
    viol = int(gap_recall > 0) + int(gap_cov > 0) + int(gap_risk > 0)
    return (0, -viol, -(gap_recall + gap_cov + gap_risk), r["invalid_screening_recall"], r["coverage"])


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
    """
    C4.5 out-of-fold trên đặc trưng lai H, chia theo group_id.
    Trả về (xác suất lá, số mẫu lá). Số mẫu lá nhân k/(k-1): mỗi fold fit trên (k-1)/k dữ liệu,
    nhân lại để cùng thang với mô hình fit trên toàn Train (xấp xỉ, kiểm chứng trên Val).
    """
    k = CFG["decision"]["cv_folds"] if k is None else k
    scale = k / (k - 1)
    proba = np.zeros((len(y), len(CFG["labels"])))
    leaf_n = np.zeros(len(y), dtype=float)
    for tr_i, va_i in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(H, y, groups):
        clf = C45Classifier(max_depth=CFG["c45"]["max_depth"], min_leaf=hp["min_leaf"], cf=hp["cf"])
        clf.fit(H[tr_i], y[tr_i], class_names=CFG["labels"])
        proba[va_i] = clf.predict_proba(H[va_i])
        leaf_n[va_i] = clf.leaf_size(H[va_i]) * scale
    return proba, leaf_n


def fit_temperature(logits, M, lo=0.05, hi=20.0):
    """
    Temperature scaling: T > 0 tối thiểu BCE(sigmoid(logit / T), M) trên OOF Train.
    Nếu OOF phân tách hoàn hảo (không có lỗi để hiệu chỉnh) thì BCE giảm đơn điệu về T→0 và nghiệm nằm ở biên.
    Khi đó dùng quy tắc không nhãn: T = std(logit OOF). Trả về (T, phương pháp).
    """
    from scipy.optimize import minimize_scalar
    z = np.asarray(logits, dtype=np.float64).ravel()
    t = np.asarray(M, dtype=np.float64).ravel()

    def bce(log_t):
        s = z / np.exp(log_t)
        return float(np.mean(np.logaddexp(0.0, s) - t * s))

    res = minimize_scalar(bce, bounds=(np.log(lo), np.log(hi)), method="bounded",
                          options={"xatol": 1e-6})
    T = float(np.exp(res.x))
    if T <= lo * 1.01 or T >= hi * 0.99:
        return float(np.std(z)), "std_fallback (OOF phân tách hoàn hảo, NLL không xác định)"
    return T, "nll"


def _search(H, y, groups, p_marks, min_leafs, grid):
    best = None
    for min_leaf in min_leafs:
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


def grid_search(H, y, groups, p_marks, grid=None):
    """
    Lưới min_leaf × cf × leaf_min_n (chọn trên OOF Train). Nếu nghiệm nằm ở biên max(grid_min_leaf),
    mở rộng bằng grid_min_leaf_ext và giữ nghiệm tốt hơn. Trả về cấu hình tốt nhất + τ.
    """
    grid = grid or CFG["decision"]
    best = _search(H, y, groups, p_marks, grid["grid_min_leaf"], grid)
    best["grid_extended"] = False
    ext = grid.get("grid_min_leaf_ext") or []
    if best["min_leaf"] == max(grid["grid_min_leaf"]) and ext:
        print(f"[grid] nghiệm ở biên min_leaf={best['min_leaf']} → mở rộng {ext}")
        cand = _search(H, y, groups, p_marks, ext, grid)
        cand["grid_extended"] = True
        if rank(cand["oof"]) > rank(best["oof"]):
            best = cand
    return best
