"""
CNN phát hiện DẤU NGOÀI Ô (wrong_mark) trên toàn phiếu.
Đầu vào: ảnh nhị phân gốc 850×1200 thu về 213×300 bằng MAX-pool 4×4 (giữ nét mảnh 3–5 px, không để INTER_AREA làm mờ).
Nhãn: n_strays > 0 (lấy từ oracle của bộ sinh, cột `n_strays` trong labels/all.csv). Chỉ dùng TRAIN để huấn luyện.
Trả về xác suất có dấu sai; dùng làm đầu vào quyết định PENDING (không dùng Test để chọn).
"""
import numpy as np
import torch
import torch.nn as nn

from config import CFG

POOL = 4


def maxpool_input(binary01):
    """(850,1200) uint8 0/1 → (213,300) float32 0/1 bằng max-pool. Đệm 2 dòng để chia hết 4."""
    b = np.pad(binary01.astype(np.float32), ((0, 2), (0, 0)))           # 850 → 852 để chia hết cho 4
    H, W = b.shape
    H4, W4 = H // POOL, W // POOL
    return b[:H4 * POOL, :W4 * POOL].reshape(H4, POOL, W4, POOL).max(axis=(1, 3))


class StrayNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.head = nn.Conv2d(64, 1, 1)          # bản đồ điểm; lấy max để bắt "có dấu ở đâu đó"

    def forward(self, x):
        return self.head(self.f(x)).amax(dim=(2, 3)).squeeze(1)


def _fit(X, y, seed, epochs, lr=1e-3, bs=32):
    torch.manual_seed(seed)
    m = StrayNet()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    pos = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos))
    Xt, yt = torch.from_numpy(X)[:, None], torch.from_numpy(y.astype(np.float32))
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(m(Xt[idx]), yt[idx]).backward()
            opt.step()
    return m.eval()


@torch.no_grad()
def predict(m, X, bs=64):
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(m(torch.from_numpy(X[i:i + bs])[:, None])).numpy())
    return np.concatenate(out)


def oof_stray(X, y, groups, seed=42, k=None, epochs=None):
    """Out-of-fold cho TRAIN (chia theo group_id). Trả về (xác suất OOF, danh sách model fold)."""
    from sklearn.model_selection import StratifiedGroupKFold
    epochs = CFG["stray_cnn"]["epochs"] if epochs is None else epochs
    k = CFG["stray_cnn"]["folds"] if k is None else k
    probs = np.zeros(len(X), np.float32)
    models = []
    for tr_i, va_i in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y, groups):
        m = _fit(X[tr_i], y[tr_i], seed, epochs)
        probs[va_i] = predict(m, X[va_i])
        models.append(m)
    return probs, models


def predict_ensemble(models, X):
    return np.mean([predict(m, X) for m in models], axis=0)


def stray_threshold(p_oof, has_stray, recall=0.99):
    """Ngưỡng chọn TRÊN OOF Train: ít nhất `recall` số phiếu có dấu sai đạt p ≥ ngưỡng."""
    p_pos = np.asarray(p_oof)[np.asarray(has_stray, bool)]
    if p_pos.size == 0:
        raise ValueError("Không có phiếu có dấu sai để chọn ngưỡng")
    # "lower": nội suy mặc định có thể cho ngưỡng cao hơn, làm recall thấp hơn yêu cầu.
    return float(np.quantile(p_pos, 1.0 - recall, method="lower"))


def load_inputs(files, raw_dir):
    """Đọc ảnh gốc → (N,213,300) float32 max-pool. Tách riêng để cache."""
    import cv2
    import os
    from preprocess import binarize_fullres
    out = np.zeros((len(files), 213, 300), np.float32)
    for i, f in enumerate(files):
        img = cv2.imread(os.path.join(raw_dir, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(os.path.join(raw_dir, f))
        out[i] = maxpool_input(binarize_fullres(img))
    return out
