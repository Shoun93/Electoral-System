"""
CÀI ĐẶT C4.5 (Quinlan, 1993) bằng numpy — không dùng CART/Gini.
  * Tiêu chí chọn thuộc tính: GAIN RATIO = Gain / SplitInfo, chỉ xét thuộc tính có Gain ≥ Gain trung bình
  * Thuộc tính liên tục: tách nhị phân theo ngưỡng giữa hai giá trị liền kề (đúng cách C4.5 xử lý số)
  * Dừng: nút thuần, đạt max_depth, hoặc một nhánh có ít hơn min_leaf mẫu
  * Cắt tỉa: PESSIMISTIC ERROR PRUNING với độ tin cậy cf (công thức U_CF của C4.5)
  * Giải thích: xuất luật IF-THEN từ mỗi lá (export_rules)
Lưu ý: C4.5 gốc tách đa nhánh cho thuộc tính rời rạc; ở đây các đặc trưng đều liên tục nên dùng tách nhị phân.
"""
import math
from statistics import NormalDist

import numpy as np


def _entropy_rows(C):
    """Entropy (bit) cho từng hàng của ma trận đếm C (shape m×K)."""
    n = C.sum(axis=1, keepdims=True)
    p = np.divide(C, n, out=np.zeros_like(C, dtype=float), where=n > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0.0)
    return -term.sum(axis=1)


def pessimistic_errors(N, E, cf):
    """Số lỗi ước lượng trên trường hợp xấu nhất với độ tin cậy (1-cf): N × U_CF(E, N)."""
    if N <= 0:
        return 0.0
    z = NormalDist().inv_cdf(1 - cf)
    f = E / N
    rad = max(f / N - f * f / N + z * z / (4 * N * N), 0.0)
    u = (f + z * z / (2 * N) + z * math.sqrt(rad)) / (1 + z * z / N)
    return N * u


def _best_split(X, y, K, min_leaf):
    """Trả về (gain, gain_ratio, feature, threshold) tốt nhất theo tiêu chí C4.5, hoặc None."""
    n = len(y)
    parent = np.bincount(y, minlength=K)
    H0 = _entropy_rows(parent[None, :])[0]
    onehot = np.eye(K)[y]
    cands = []                                   # (gain, ratio, feature, threshold)
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f], kind="stable")
        xs = X[order, f]
        cut = np.where(xs[1:] > xs[:-1])[0]     # tách sau vị trí i nếu giá trị đổi
        if cut.size == 0:
            continue
        cum = np.cumsum(onehot[order], axis=0)
        left = cum[cut]
        right = parent[None, :] - left
        nl, nr = left.sum(1), right.sum(1)
        ok = (nl >= min_leaf) & (nr >= min_leaf)
        if not ok.any():
            continue
        left, right, nl, nr, cut = left[ok], right[ok], nl[ok], nr[ok], cut[ok]
        Hl, Hr = _entropy_rows(left), _entropy_rows(right)
        gain = H0 - (nl * Hl + nr * Hr) / n
        p = nl / n
        split_info = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        ratio = gain / split_info
        thr = (xs[cut] + xs[cut + 1]) / 2
        for g, r, t in zip(gain, ratio, thr):
            cands.append((float(g), float(r), f, float(t)))
    if not cands:
        return None
    avg_gain = np.mean([c[0] for c in cands])
    eligible = [c for c in cands if c[0] >= avg_gain - 1e-12]   # quy tắc C4.5: gain ≥ trung bình
    return max(eligible, key=lambda c: c[1])


class C45Classifier:
    def __init__(self, max_depth=6, min_leaf=5, cf=0.25):
        self.max_depth, self.min_leaf, self.cf = max_depth, min_leaf, cf
        self.tree_ = None
        self.feature_names_ = None
        self.class_names_ = None
        self.n_classes_ = None

    # ---------------- huấn luyện ----------------
    def fit(self, X, y, feature_names=None, class_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes_ = len(class_names) if class_names is not None else int(y.max()) + 1
        self.feature_names_ = list(feature_names) if feature_names is not None \
            else [f"x{i}" for i in range(X.shape[1])]
        self.class_names_ = list(class_names) if class_names is not None \
            else [str(i) for i in range(self.n_classes_)]
        self.tree_ = self._build(X, y, 0)
        self._prune(self.tree_)
        return self

    def _leaf(self, y):
        counts = np.bincount(y, minlength=self.n_classes_)
        return {"leaf": True, "counts": counts, "n": int(len(y)), "pred": int(counts.argmax())}

    def _build(self, X, y, depth):
        node = self._leaf(y)
        if len(np.unique(y)) == 1 or depth >= self.max_depth or len(y) < 2 * self.min_leaf:
            return node
        best = _best_split(X, y, self.n_classes_, self.min_leaf)
        if best is None:
            return node
        _, _, f, thr = best
        mask = X[:, f] <= thr
        node.update(leaf=False, feature=f, threshold=thr)
        node["left"] = self._build(X[mask], y[mask], depth + 1)
        node["right"] = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _leaf_estimate(self, node):
        E = node["n"] - node["counts"][node["pred"]]
        return pessimistic_errors(node["n"], E, self.cf)

    def _prune(self, node):
        """Bottom-up: thay cây con bằng lá nếu lỗi ước lượng của lá không lớn hơn. Trả về lỗi ước lượng."""
        if node["leaf"]:
            return self._leaf_estimate(node)
        sub = self._prune(node["left"]) + self._prune(node["right"])
        leaf_err = self._leaf_estimate(node)
        if leaf_err <= sub + 1e-9:
            for k in ("left", "right", "feature", "threshold"):
                node.pop(k, None)
            node["leaf"] = True
            return leaf_err
        return sub

    # ---------------- suy luận ----------------
    def _route(self, x):
        node = self.tree_
        while not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros((len(X), self.n_classes_))
        for i, x in enumerate(X):
            c = self._route(x)["counts"].astype(float)
            out[i] = c / c.sum() if c.sum() > 0 else 1.0 / self.n_classes_
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def leaf_size(self, X):
        """Số mẫu train tại lá mà mỗi mẫu rơi vào."""
        return np.array([self._route(x)["n"] for x in np.asarray(X, dtype=float)])

    # ---------------- giải thích ----------------
    def export_rules(self):
        """Danh sách luật IF-THEN: (điều kiện, lớp dự đoán, số mẫu, số lỗi huấn luyện)."""
        rules = []

        def walk(node, conds):
            if node["leaf"]:
                E = node["n"] - node["counts"][node["pred"]]
                rules.append((" AND ".join(conds) or "TRUE", self.class_names_[node["pred"]],
                              node["n"], int(E)))
                return
            name = self.feature_names_[node["feature"]]
            t = node["threshold"]
            walk(node["left"], conds + [f"{name} ≤ {t:.4g}"])
            walk(node["right"], conds + [f"{name} > {t:.4g}"])

        walk(self.tree_, [])
        return rules

    def export_text(self):
        lines = []
        for cond, cls, n, e in self.export_rules():
            lines.append(f"IF {cond}\n    THEN {cls}  [n={n}, lỗi={e}]")
        return "\n".join(lines)
