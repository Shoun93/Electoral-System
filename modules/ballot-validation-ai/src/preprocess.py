"""
TIỀN XỬ LÝ ẢNH PHIẾU.
Hai nhánh dùng hai độ phân giải khác nhau:
  * Nhánh C4.5 : nhị phân hóa ở ĐỘ PHÂN GIẢI GỐC (nét dấu ~3 px sẽ mất nếu thu nhỏ trước).
  * Nhánh CNN  : cùng ảnh nhị phân, thu về 224×224 (INTER_AREA), chuẩn hóa [0,1].
Ảnh nhị phân: 1 = mực (chữ, dấu), 0 = nền.
"""
import csv
import hashlib
import importlib.util
import json
import os

import cv2
import numpy as np

from config import CFG, MODULE_DIR, abspath
from feature_c45 import extract_features, FEATURE_NAMES

SIZE = CFG["preprocess"]["img_size"]
REPO_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", ".."))
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


def dataset_key():
    """SHA-256 của nhãn, ROI, ảnh gốc, cấu hình và mã tiền xử lý: đổi bất kỳ thứ nào thì cache hết hiệu lực."""
    h = hashlib.sha256(json.dumps([CFG["labels"], CFG["preprocess"], CFG["features"]], sort_keys=True).encode())
    labels_csv = abspath(CFG["data"]["labels_csv"])
    with open(labels_csv, encoding="utf-8") as f:
        raw = [os.path.join(abspath(CFG["data"]["raw_dir"]), r["file"]) for r in csv.DictReader(f)]
    src = os.path.dirname(os.path.abspath(__file__))
    code = [os.path.join(src, n) for n in ("preprocess.py", "feature_c45.py")]
    for p in [labels_csv, abspath(CFG["data"]["boxes_json"]), *code, *raw]:
        with open(p, "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def mark_cache_key(data, model_path):
    """Khóa cache xác suất CNN theo ô: dữ liệu, split, cấu hình, mã huấn luyện và checkpoint."""
    h = hashlib.sha256(json.dumps([CFG["mark_cnn"], CFG["data"]["val_fraction"]], sort_keys=True).encode())
    for name in ("C", "M", "y", "groups", "split", "files"):
        a = np.ascontiguousarray(data[name])
        h.update(f"{name}{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    for p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_cnn.py"), model_path):
        with open(p, "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def build_processed():
    """Đọc ảnh gốc → đặc trưng C4.5 (ở độ phân giải gốc) + ảnh CNN 224×224 → một file .npz."""
    out = abspath(CFG["data"]["processed_npz"])
    key = dataset_key()
    if os.path.exists(out):
        with np.load(out, allow_pickle=False) as d:
            if "source_key" in d.files and str(d["source_key"]) == key:
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
                        feature_names=np.asarray(FEATURE_NAMES), source_key=np.asarray(key))
    print(f"✅ Tiền xử lý {len(rows)} phiếu → {out}")
    return out


def load_dataset():
    """Trả về dict: X, F, C, M, y, split, files, groups (tạo lại nếu nguồn đã đổi)."""
    d = np.load(build_processed(), allow_pickle=False)
    return {k: d[k] for k in ["X", "F", "C", "M", "y", "split", "files", "groups"]}


def check_group_disjoint(train_groups, val_groups):
    """D-06: dừng nếu một group_id xuất hiện ở cả Train và Validation."""
    overlap = set(train_groups) & set(val_groups)
    if overlap:
        raise ValueError(f"Group overlap Train/Validation: {len(overlap)} group, ví dụ {sorted(overlap)[:5]}")
    return len(overlap)


def split_train_val(data, seed=42):
    """Tách Validation khỏi Train bằng GroupShuffleSplit theo group_id (D-06).
    Validation = model-selection validation, KHÔNG phải Independent Test. Tập test giữ nguyên.
    Trả về indices gốc (ánh xạ 1-1 với data) và meta để ghi vào metrics."""
    from sklearn.model_selection import GroupShuffleSplit
    tr_idx = np.where(data["split"] == "train")[0]
    te_idx = np.where(data["split"] == "test")[0]
    groups = np.asarray(data["groups"])
    if any(not str(g).strip() for g in groups[tr_idx]):
        raise ValueError("group_id rỗng trong tập Train")
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=CFG["data"]["val_fraction"],
                                  random_state=seed).split(tr_idx, groups=groups[tr_idx]))
    tr_out, va_idx = tr_idx[a], tr_idx[b]
    overlap = check_group_disjoint(groups[tr_out], groups[va_idx])
    y = np.asarray(data["y"])
    meta = {"method": "GroupShuffleSplit", "seed": int(seed),
            "test_size_groups": CFG["data"]["val_fraction"],
            "validation_purpose": "model-selection validation (không phải Independent Test)",
            "n_train": int(len(tr_out)), "n_val": int(len(va_idx)),
            "n_train_groups": int(len(set(groups[tr_out]))), "n_val_groups": int(len(set(groups[va_idx]))),
            "group_overlap_train_val": int(overlap), "leakage_check": "PASS",
            "val_class_counts": {str(c): int(n) for c, n in zip(*np.unique(y[va_idx], return_counts=True))}}
    return {"train": tr_out, "val": va_idx, "test": te_idx, "meta": meta}
