"""
TIỀN XỬ LÝ ẢNH PHIẾU.
Hai nhánh dùng hai độ phân giải khác nhau:
  * Nhánh C4.5 : nhị phân hóa ở ĐỘ PHÂN GIẢI GỐC (nét dấu ~3 px sẽ mất nếu thu nhỏ trước).
  * Nhánh CNN  : cùng ảnh nhị phân, thu về 224×224 (INTER_AREA), chuẩn hóa [0,1].
Ảnh nhị phân: 1 = mực (chữ, dấu), 0 = nền.
"""
import csv
import importlib.util
import json
import os

import cv2
import numpy as np

from config import CFG, MODULE_DIR, SRC_DIR, abspath
from feature_c45 import extract_features, FEATURE_NAMES

SIZE = CFG["preprocess"]["img_size"]
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", ".."))
GENERATOR_PATH = os.path.join(REPO_ROOT, "dataset", "generate_dataset.py")


def binarize_fullres(img_bgr_or_gray):
    """CLAHE → OTSU ở độ phân giải gốc. Trả về mảng uint8 0/1 (1 = mực)."""
    gray = img_bgr_or_gray
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CFG["preprocess"]["clahe_clip"], tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (binary > 0).astype(np.uint8)


def to_cnn_input(binary01):
    """Thu về SIZE×SIZE, float32 [0,1]."""
    resized = cv2.resize(binary01.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


MARK_SIZE = 48      # kích thước crop một ô đánh dấu đưa vào CNN phát hiện dấu
MARK_MARGIN = 6     # nới rộng crop quanh ô (px ảnh gốc) để không cắt mất nét dấu


def crop_boxes(binary, boxes):
    """Cắt 5 ô (đã nới MARK_MARGIN px) từ ảnh nhị phân gốc → mảng (5, MARK_SIZE, MARK_SIZE) float32."""
    H, W = binary.shape
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(int(round(x1)) - MARK_MARGIN, 0), max(int(round(y1)) - MARK_MARGIN, 0)
        x2, y2 = min(int(round(x2)) + MARK_MARGIN, W), min(int(round(y2)) + MARK_MARGIN, H)
        patch = binary[y1:y2, x1:x2].astype(np.float32)
        out.append(cv2.resize(patch, (MARK_SIZE, MARK_SIZE), interpolation=cv2.INTER_AREA))
    return np.stack(out).astype(np.float32)


def generate_synthetic_ballots():
    """Gọi bộ sinh dữ liệu ở thư mục dataset/ của repo; ghi vào data/ của module."""
    spec = importlib.util.spec_from_file_location("generate_dataset", GENERATOR_PATH)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.generate(out_root=MODULE_DIR)


def build_processed(force=False):
    """Đọc ảnh gốc → đặc trưng C4.5 (ở độ phân giải gốc) + ảnh CNN 224×224 → một file .npz."""
    out = abspath(CFG["data"]["processed_npz"])
    if os.path.exists(out) and not force:
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    labels = CFG["labels"]
    with open(abspath(CFG["data"]["labels_csv"]), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(abspath(CFG["data"]["boxes_json"]), encoding="utf-8") as f:
        layout = json.load(f)

    X, F, C, M, y, split, files, groups = [], [], [], [], [], [], [], []
    for r in rows:
        img = cv2.imread(os.path.join(abspath(CFG["data"]["raw_dir"]), r["file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(r["file"])
        binary = binarize_fullres(img)
        lay = layout[r["file"]]
        F.append(extract_features(binary, lay))
        X.append(to_cnn_input(binary))
        C.append(crop_boxes(binary, lay["boxes"]))
        marked = {int(c) for c in r["o_co_danh_dau"].split("|") if c != ""}
        M.append([int(k in marked) for k in range(len(lay["boxes"]))])
        y.append(labels.index(r["label"]))
        split.append(r["split"])
        files.append(r["file"])
        groups.append(r["group_id"])
    np.savez_compressed(out, X=np.stack(X), F=np.asarray(F, dtype=np.float32),
                        C=np.stack(C), M=np.asarray(M, dtype=np.int64),
                        y=np.asarray(y), split=np.asarray(split), files=np.asarray(files),
                        groups=np.asarray(groups),
                        feature_names=np.asarray(FEATURE_NAMES))
    print(f"✅ Tiền xử lý {len(rows)} phiếu → {out}")
    return out


def load_dataset():
    """Trả về dict: X, F, C, M, y, split, files, groups (đã tạo nếu chưa có hoặc thiếu groups)."""
    path = build_processed()
    if "groups" not in np.load(path, allow_pickle=False).files:   # file cũ, tạo lại
        path = build_processed(force=True)
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in ["X", "F", "C", "M", "y", "split", "files", "groups"]}


def split_train_val(data, seed=42):
    """Tách validation từ Train THEO GROUP (không để cùng một phiếu gốc ở cả hai tập), phân tầng theo nhãn.
    Số fold = 1/val_fraction; lấy một fold làm validation. Tập test giữ nguyên."""
    from sklearn.model_selection import StratifiedGroupKFold
    tr_idx = np.where(data["split"] == "train")[0]
    te_idx = np.where(data["split"] == "test")[0]
    k = round(1.0 / CFG["data"]["val_fraction"])
    rel_tr, rel_va = next(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
                          .split(tr_idx, data["y"][tr_idx], data["groups"][tr_idx]))
    return {"train": tr_idx[rel_tr], "val": tr_idx[rel_va], "test": te_idx}
