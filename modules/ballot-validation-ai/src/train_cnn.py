"""
HUẤN LUYỆN CNN (PyTorch).
Kiến trúc: 3 khối [Conv3×3 → BN → ReLU → MaxPool2] + FC(128) + Dropout(0.3) + FC(5)
  224 → 112 → 56 → 28; Flatten 64×28×28 → 128 → 5 lớp.
Tăng cường dữ liệu CHỈ khi huấn luyện (lưu đồ bước 2): xoay ±3°, dịch ±8 px, làm dày nét.
Checkpoint lưu theo accuracy trên tập VALIDATION; tập test không tham gia huấn luyện.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

from config import CFG, abspath

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BallotCNN(nn.Module):
    def __init__(self, num_classes=5, size=224):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        side = size // 8
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * side * side, 128), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def augment(x, max_shift=8):
    """Tăng cường batch (B,1,H,W) ảnh nhị phân [0,1]: xoay ±3°, dịch ±max_shift px, làm dày nét ngẫu nhiên."""
    B = x.size(0)
    ang = (torch.rand(B, device=x.device) * 2 - 1) * 3 * math.pi / 180
    tx = (torch.rand(B, device=x.device) * 2 - 1) * max_shift * 2 / x.size(-1)   # px → toạ độ chuẩn [-1,1]
    ty = (torch.rand(B, device=x.device) * 2 - 1) * max_shift * 2 / x.size(-2)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    thick = (torch.rand(B, 1, 1, 1, device=x.device) < 0.3).float()
    return thick * F.max_pool2d(x, 3, 1, 1) + (1 - thick) * x


def _to_tensor(X):
    return torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)


def train_cnn(X_tr, y_tr, X_va, y_va, cfg=None):
    hp = dict(CFG["cnn"], **(cfg or {}))
    K = len(CFG["labels"])
    torch.manual_seed(42)
    model = BallotCNN(K, size=X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = nn.CrossEntropyLoss()

    Xt, yt = _to_tensor(X_tr), torch.as_tensor(y_tr, dtype=torch.long)
    Xv, yv = _to_tensor(X_va), torch.as_tensor(y_va, dtype=torch.long)
    save_path = abspath(hp["save_path"])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best = -1.0
    for ep in range(1, hp["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), hp["batch_size"]):
            idx = perm[i:i + hp["batch_size"]]
            x, y = Xt[idx].to(DEVICE), yt[idx].to(DEVICE)
            if hp["augment"]:
                x = augment(x)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pred = model(Xv.to(DEVICE)).argmax(1).cpu()
        acc = (pred == yv).float().mean().item()
        print(f"Epoch {ep:02d} | loss={total / len(Xt):.4f} | val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    print(f"✅ Best val_acc = {best:.4f} → {save_path}")
    model.eval()
    return model


def predict_cnn(model, X, batch_size=64):
    """Trả về (nhãn dự đoán, xác suất lớp) cho mảng ảnh (N,H,W)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = _to_tensor(X[i:i + batch_size]).to(DEVICE)
            probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    return probs.argmax(1), probs


def load_cnn(path=None):
    path = abspath(path or CFG["cnn"]["save_path"])
    model = BallotCNN(len(CFG["labels"])).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# ============================================================================
# CNN PHÁT HIỆN DẤU TRÊN TỪNG Ô (nhánh CNN của mô hình lai)
# Đầu vào: crop 48×48 của một ô (ảnh nhị phân gốc). Đầu ra: xác suất ô có dấu.
# CNN chỉ "phát hiện dấu và trích xuất đặc trưng"; quyết định hợp lệ/không hợp lệ do C4.5 đưa ra.
# ============================================================================
class MarkCNN(nn.Module):
    def __init__(self, size=48):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        side = size // 8
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * side * side, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def _fit_mark_cnn(C, M, cfg, seed=42):
    """C: (n,48,48) crop; M: (n,) nhãn 0/1. Trả về mô hình đã huấn luyện."""
    torch.manual_seed(seed)
    model = MarkCNN(C.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    X = _to_tensor(C)
    y = torch.as_tensor(M, dtype=torch.float32)
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            x, t = X[idx].to(DEVICE), y[idx].to(DEVICE)
            if cfg.get("augment", True):
                x = augment(x, max_shift=3)
            opt.zero_grad()
            loss_fn(model(x), t).backward()
            opt.step()
    model.eval()
    return model


def predict_mark_logits(model, C, batch_size=256):
    """C: (n,5,48,48) crop của từng phiếu → logit có dấu (n,5). Xác suất = sigmoid(logit / T), T fit trên OOF."""
    model.eval()
    n, k = C.shape[:2]
    flat = C.reshape(n * k, *C.shape[2:])
    out = []
    with torch.no_grad():
        for i in range(0, len(flat), batch_size):
            out.append(model(_to_tensor(flat[i:i + batch_size]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out).reshape(n, k).astype(np.float64)


def train_mark_cnn(C_tr, M_tr, cfg=None):
    hp = dict(CFG["mark_cnn"], **(cfg or {}))
    model = _fit_mark_cnn(C_tr.reshape(-1, *C_tr.shape[2:]), M_tr.reshape(-1), hp)
    return model


def oof_mark_logits(C_tr, M_tr, cfg=None, k=5, seed=42, groups=None, strat=None):
    """
    Out-of-fold: mỗi phiếu train được dự đoán bởi mô hình KHÔNG thấy phiếu đó.
    Nhờ vậy C4.5 học trên xác suất CNN "thật" chứ không phải xác suất đã học thuộc.
    groups/strat: chia theo group_id (cùng một phiếu gốc không nằm ở cả hai phía), phân tầng theo nhãn.
    """
    from sklearn.model_selection import KFold, StratifiedGroupKFold
    hp = dict(CFG["mark_cnn"], **(cfg or {}))
    logits = np.zeros(M_tr.shape, dtype=np.float64)
    if groups is None:
        splits = KFold(n_splits=k, shuffle=True, random_state=seed).split(C_tr)
    else:
        splits = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(C_tr, strat, groups)
    for tr_i, te_i in splits:
        m = _fit_mark_cnn(C_tr[tr_i].reshape(-1, *C_tr.shape[2:]), M_tr[tr_i].reshape(-1), hp, seed)
        logits[te_i] = predict_mark_logits(m, C_tr[te_i])
    return logits
