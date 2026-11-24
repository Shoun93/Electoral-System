"""Selective-decision metrics with explicit business-policy gates.

No class is implicitly treated as INVALID. No threshold is selected until the
PM-approved policy and the separate OOF authorization are present in config.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from c45 import C45Classifier
from config import CFG

VALID_IDX = 0


class PolicyNotApprovedError(RuntimeError):
    """Raised when code is asked to make a selection under unresolved policy."""


def _as_1d(name, values, dtype=None):
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def confidences(p_marks, proba):
    p_marks = np.asarray(p_marks, dtype=float)
    proba = np.asarray(proba, dtype=float)
    if p_marks.ndim != 2 or proba.ndim != 2 or len(p_marks) != len(proba):
        raise ValueError("p_marks and proba must be 2-D arrays with equal row counts")
    if not np.isfinite(p_marks).all() or not np.isfinite(proba).all():
        raise ValueError("probabilities must be finite")
    if ((p_marks < 0) | (p_marks > 1)).any() or ((proba < 0) | (proba > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("each class-probability row must sum to 1")
    return np.maximum(p_marks, 1.0 - p_marks).min(axis=1), proba.max(axis=1)


def is_pending(conf_cnn, conf_c45, tau_cnn, tau_c45, leaf_n=None, leaf_min_n=0):
    cc = _as_1d("conf_cnn", conf_cnn, float)
    cd = _as_1d("conf_c45", conf_c45, float)
    if len(cc) != len(cd) or not np.isfinite(cc).all() or not np.isfinite(cd).all():
        raise ValueError("confidence arrays must be finite and have equal length")
    if not (0 <= tau_cnn <= 1 and 0 <= tau_c45 <= 1):
        raise ValueError("thresholds must be in [0, 1]")
    pending = (cc < tau_cnn) | (cd < tau_c45)
    if leaf_n is not None:
        leaves = _as_1d("leaf_n", leaf_n, int)
        if len(leaves) != len(cc) or (leaves < 0).any():
            raise ValueError("leaf_n must be non-negative and match confidence length")
        if leaf_min_n < 0:
            raise ValueError("leaf_min_n must be non-negative")
        pending |= leaves < leaf_min_n
    return pending


def screening_report(y, pred, pending, invalid_labels=None, valid_label=VALID_IDX):
    """Report descriptive selective metrics.

    invalid_labels must be an explicit PM-approved iterable of class indices.
    PENDING_REVIEW is reported as referral, never counted as automatic detection.
    """
    y = _as_1d("y", y)
    pred = _as_1d("pred", pred)
    pending = _as_1d("pending", pending, bool)
    if not (len(y) == len(pred) == len(pending)):
        raise ValueError("y, pred and pending must have equal lengths")
    if len(y) == 0:
        raise ValueError("cannot calculate metrics for an empty sample")
    if not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer):
        raise ValueError("y and pred must contain integer class indices")

    auto = ~pending
    n_auto = int(auto.sum())
    wrong_auto = auto & (pred != y)
    result = {
        "coverage": round(float(auto.mean()), 4),
        "referral_rate": round(float(pending.mean()), 4),
        "selective_risk": round(float(wrong_auto.sum()) / n_auto, 4) if n_auto else None,
        "n": int(len(y)),
        "n_auto": n_auto,
        "n_pending": int(pending.sum()),
        "selective_risk_denominator": n_auto,
        "invalid_metrics_status": "BLOCKED_POLICY_NOT_APPROVED",
    }

    if invalid_labels is not None:
        invalid_set = {int(label) for label in invalid_labels}
        if not invalid_set or valid_label in invalid_set:
            raise ValueError("invalid_labels must be non-empty and exclude valid_label")
        invalid = np.isin(y, list(invalid_set))
        n_invalid = int(invalid.sum())
        if n_invalid == 0:
            result.update({
                "invalid_metrics_status": "UNDEFINED_NO_INVALID_SAMPLES",
                "n_invalid": 0,
                "automatic_invalid_recall": None,
                "invalid_referral_rate": None,
                "invalid_interception_rate": None,
            })
        else:
            auto_invalid = invalid & auto & (pred != valid_label)
            referred_invalid = invalid & pending
            intercepted_invalid = auto_invalid | referred_invalid
            result.update({
                "invalid_metrics_status": "DESCRIPTIVE_POLICY_SCOPED",
                "n_invalid": n_invalid,
                "automatic_invalid_recall": round(float(auto_invalid.sum()) / n_invalid, 4),
                "invalid_referral_rate": round(float(referred_invalid.sum()) / n_invalid, 4),
                "invalid_interception_rate": round(float(intercepted_invalid.sum()) / n_invalid, 4),
                "invalid_denominator": n_invalid,
                "pending_counted_as_automatic_detection": False,
            })
    return result


def _approved_acceptance_policy():
    d = CFG.get("decision", {})
    required = ("acceptance_policy_approved", "nfr10_target", "coverage_min",
                "selective_risk_max", "invalid_labels")
    if not all(key in d for key in required):
        return False
    if d["acceptance_policy_approved"] is not True:
        return False
    if d["nfr10_target"] is None or d["coverage_min"] is None or d["selective_risk_max"] is None:
        return False
    if d["invalid_labels"] is None:
        return False
    return True


def _require_oof_authorized():
    d = CFG.get("decision", {})
    if not _approved_acceptance_policy():
        raise PolicyNotApprovedError("PM-approved label and acceptance policy is missing; threshold selection is blocked")
    if d.get("oof_selection_authorized") is not True:
        raise PolicyNotApprovedError("OOF selection requires separate explicit PM authorization")


def feasible(report):
    """Acceptance decision is unavailable until PM policy is approved."""
    if not _approved_acceptance_policy():
        return False
    d = CFG["decision"]
    recall = report.get("automatic_invalid_recall")
    return (
        recall is not None
        and recall >= d["nfr10_target"]
        and report.get("coverage") is not None
        and report["coverage"] >= d["coverage_min"]
        and report.get("selective_risk") is not None
        and report["selective_risk"] <= d["selective_risk_max"]
    )


def rank(report):
    """Do not rank configurations while policy or OOF authorization is unresolved."""
    _require_oof_authorized()
    if feasible(report):
        return (2, report["automatic_invalid_recall"], -report["selective_risk"], report["coverage"])
    d = CFG["decision"]
    recall = report.get("automatic_invalid_recall")
    violations = (
        recall is None or recall < d["nfr10_target"],
        report.get("coverage") is None or report["coverage"] < d["coverage_min"],
        report.get("selective_risk") is None or report["selective_risk"] > d["selective_risk_max"],
    )
    return (0, -sum(violations), recall if recall is not None else -1.0,
            report.get("coverage") if report.get("coverage") is not None else -1.0)


def select_tau(y, pred, conf_cnn, conf_c45, leaf_n=None, leaf_min_n=0, step=None):
    """Select thresholds only after both policy and OOF approvals are recorded."""
    _require_oof_authorized()
    step = CFG["decision"]["tau_step"] if step is None else step
    if not isinstance(step, (int, float)) or not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")
    grid_cnn = np.round(np.arange(0.5, 1.0 + 1e-9, step), 4)
    grid_c45 = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    best = None
    for tc in grid_cnn:
        for td in grid_c45:
            pending = is_pending(conf_cnn, conf_c45, tc, td, leaf_n, leaf_min_n)
            report = screening_report(y, pred, pending, invalid_labels=CFG["decision"]["invalid_labels"])
            if best is None or rank(report) > rank(best[2]):
                best = (tc, td, report)
    tc, td, report = best
    ok = feasible(report)
    return {"tau_cnn": float(tc), "tau_c45": float(td), "leaf_min_n": int(leaf_min_n),
            "tier": 2 if ok else 0, "target_met_oof": ok, "oof_target_met": ok, "oof": report}


def _validate_group_folds(y, groups, k, seed=42):
    y = _as_1d("y", y)
    groups = _as_1d("groups", groups)
    if len(y) != len(groups) or len(y) == 0:
        raise ValueError("y and groups must be non-empty and have equal length")
    if any(value is None or not str(value).strip() for value in groups):
        raise ValueError("group IDs must be non-empty")
    unique_groups = np.unique(groups.astype(str))
    if not isinstance(k, int) or k < 2 or len(unique_groups) < k:
        raise ValueError(f"Need at least {k} distinct groups for {k}-fold OOF; found {len(unique_groups)}")
    # Validate actual generated folds rather than assuming n_groups >= k*k.
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    try:
        splits = list(splitter.split(np.zeros(len(y)), y, groups.astype(str)))
    except ValueError as exc:
        raise ValueError(f"Cannot construct StratifiedGroupKFold({k}): {exc}") from exc
    coverage = np.zeros(len(y), dtype=int)
    all_classes = set(np.unique(y).tolist())
    for fold, (train_idx, held_idx) in enumerate(splits):
        if not len(train_idx) or not len(held_idx):
            raise ValueError(f"Fold {fold} has an empty partition")
        if set(groups[train_idx].astype(str)) & set(groups[held_idx].astype(str)):
            raise ValueError(f"Group leakage in fold {fold}")
        missing_train = all_classes - set(np.unique(y[train_idx]).tolist())
        if missing_train:
            raise ValueError(f"Fold {fold} training partition misses classes: {sorted(missing_train)}")
        coverage[held_idx] += 1
    if not np.all(coverage == 1):
        raise ValueError("OOF folds must predict every sample exactly once")
    return splits


def oof_c45(H, y, groups, hp, k=None, seed=42):
    _require_oof_authorized()
    k = CFG["decision"]["cv_folds"] if k is None else k
    H, y, groups = np.asarray(H), _as_1d("y", y), _as_1d("groups", groups)
    if H.ndim != 2 or len(H) != len(y) or len(y) != len(groups):
        raise ValueError("H, y and groups must have consistent sample dimensions")
    splits = _validate_group_folds(y, groups, k, seed=seed)
    proba = np.zeros((len(y), len(CFG["labels"])), dtype=float)
    leaf_n = np.zeros(len(y), dtype=int)
    for fold, (tr_i, va_i) in enumerate(splits):
        if set(groups[tr_i].astype(str)) & set(groups[va_i].astype(str)):
            raise ValueError(f"Group leakage in OOF fold {fold}")
        clf = C45Classifier(max_depth=CFG["c45"]["max_depth"], min_leaf=hp["min_leaf"], cf=hp["cf"])
        clf.fit(H[tr_i], y[tr_i], class_names=CFG["labels"])
        fold_proba = clf.predict_proba(H[va_i])
        if fold_proba.shape != (len(va_i), len(CFG["labels"])):
            raise ValueError(f"Unexpected probability shape in fold {fold}")
        proba[va_i] = fold_proba
        leaf_n[va_i] = clf.leaf_size(H[va_i])
    if not np.isfinite(proba).all() or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("OOF predictions contain missing or invalid probabilities")
    return proba, leaf_n


def grid_search(H, y, groups, p_marks, grid=None):
    """Legacy flat-OOF selection is intentionally disabled.

    Reusing the same OOF predictions to tune hyperparameters/thresholds and report
    performance is selection-biased. A nested group-aware selection protocol must
    be implemented and separately approved before this function can be enabled.
    """
    raise PolicyNotApprovedError(
        "Flat OOF grid search is disabled: nested group-aware selection is not implemented/approved."
    )
    _require_oof_authorized()
    grid = grid or CFG["decision"]
    best = None
    for min_leaf in grid["grid_min_leaf"]:
        for cf in grid["grid_cf"]:
            proba, leaf_n = oof_c45(H, y, groups, {"min_leaf": min_leaf, "cf": cf})
            cc, cd = confidences(p_marks, proba)
            for leaf_min_n in grid["grid_leaf_min_n"]:
                result = select_tau(y, proba.argmax(1), cc, cd, leaf_n=leaf_n, leaf_min_n=leaf_min_n)
                result.update(min_leaf=min_leaf, cf=cf)
                if best is None or rank(result["oof"]) > rank(best["oof"]):
                    best = result
    if best is None:
        raise ValueError("Selection grid is empty")
    return best
