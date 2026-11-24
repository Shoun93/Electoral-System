"""
TIỀN XỬ LÝ ẢNH PHIẾU.
Hai nhánh dùng hai độ phân giải khác nhau:
  * Nhánh C4.5 : nhị phân hóa ở ĐỘ PHÂN GIẢI GỐC.
  * Nhánh CNN  : cùng ảnh nhị phân, thu về SIZE×SIZE, chuẩn hóa [0,1].
Ảnh nhị phân: 1 = mực (chữ, dấu), 0 = nền.
"""
import csv
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


MARK_SIZE = 48
MARK_MARGIN = 6


def crop_boxes(binary, boxes):
    """Cắt crop quanh các ô đánh dấu từ ảnh nhị phân gốc."""
    if np.asarray(binary).ndim != 2 or not boxes:
        raise ValueError("binary phải là ảnh 2-D và boxes không được rỗng")
    height, width = binary.shape
    out = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError("Mỗi box phải có dạng (x1, y1, x2, y2)")
        x1, y1, x2, y2 = box
        x1, y1 = max(int(round(x1)) - MARK_MARGIN, 0), max(int(round(y1)) - MARK_MARGIN, 0)
        x2, y2 = min(int(round(x2)) + MARK_MARGIN, width), min(int(round(y2)) + MARK_MARGIN, height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Box không hợp lệ: {box}")
        patch = binary[y1:y2, x1:x2].astype(np.float32)
        out.append(cv2.resize(patch, (MARK_SIZE, MARK_SIZE), interpolation=cv2.INTER_AREA))
    return np.stack(out).astype(np.float32)


def generate_synthetic_ballots():
    """Gọi bộ sinh dữ liệu ở thư mục dataset/ của repo; chỉ được gọi theo lệnh chạy pipeline."""
    spec = importlib.util.spec_from_file_location("generate_dataset", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Không thể nạp bộ sinh dữ liệu giả lập")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.generate(out_root=MODULE_DIR)


def build_processed(force=False):
    """Đọc ảnh gốc → đặc trưng C4.5 + ảnh CNN + crop ô → một file .npz."""
    out = abspath(CFG["data"]["processed_npz"])
    if os.path.exists(out) and not force:
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    labels = CFG["labels"]
    with open(abspath(CFG["data"]["labels_csv"]), encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with open(abspath(CFG["data"]["boxes_json"]), encoding="utf-8") as stream:
        layout = json.load(stream)

    X, F, C, M, y, split, files, groups = [], [], [], [], [], [], [], []
    for row in rows:
        image_path = os.path.join(abspath(CFG["data"]["raw_dir"]), row["file"])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image_path)
        binary = binarize_fullres(img)
        if row["file"] not in layout:
            raise ValueError(f"Thiếu ROI cho ảnh: {row['file']}")
        lay = layout[row["file"]]
        boxes = lay.get("boxes")
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f"ROI boxes không hợp lệ: {row['file']}")
        F.append(extract_features(binary, lay))
        X.append(to_cnn_input(binary))
        C.append(crop_boxes(binary, boxes))
        marked = {int(c) for c in row["o_co_danh_dau"].split("|") if c != ""}
        if any(index < 0 or index >= len(boxes) for index in marked):
            raise ValueError(f"Chỉ số ô đánh dấu ngoài phạm vi: {row['file']}")
        M.append([int(k in marked) for k in range(len(boxes))])
        if row["label"] not in labels:
            raise ValueError(f"Nhãn không được khai báo: {row['label']}")
        y.append(labels.index(row["label"]))
        split.append(row["split"])
        files.append(row["file"])
        groups.append(row["group_id"])
    if not rows:
        raise ValueError("Manifest dữ liệu rỗng")
    np.savez_compressed(out, X=np.stack(X), F=np.asarray(F, dtype=np.float32),
                        C=np.stack(C), M=np.asarray(M, dtype=np.int64),
                        y=np.asarray(y), split=np.asarray(split), files=np.asarray(files),
                        groups=np.asarray(groups), feature_names=np.asarray(FEATURE_NAMES))
    print(f"Tiền xử lý {len(rows)} phiếu → {out}")
    return out


def load_dataset():
    """Trả về các mảng dữ liệu, bảo đảm cache tiền xử lý có trường groups."""
    path = build_processed()
    with np.load(path, allow_pickle=False) as stored:
        if "groups" not in stored.files:
            path = build_processed(force=True)
    with np.load(path, allow_pickle=False) as stored:
        required = ("X", "F", "C", "M", "y", "split", "files", "groups")
        missing = [key for key in required if key not in stored.files]
        if missing:
            raise ValueError(f"Processed dataset thiếu trường: {missing}")
        return {key: stored[key] for key in required}


def _validated_group_ids(values, name):
    groups = np.asarray(values, dtype=object)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError(f"{name} phải là mảng 1-D không rỗng")
    cleaned = []
    for index, value in enumerate(groups):
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            raise ValueError(f"{name}[{index}] rỗng")
        value = str(value).strip()
        if not value:
            raise ValueError(f"{name}[{index}] rỗng")
        cleaned.append(value)
    return np.asarray(cleaned, dtype=str)


def check_group_disjoint(train_groups, val_groups):
    """Dừng nếu group_id rỗng hoặc xuất hiện ở cả hai partition."""
    train = _validated_group_ids(train_groups, "train_groups")
    val = _validated_group_ids(val_groups, "val_groups")
    overlap = set(train.tolist()) & set(val.tolist())
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(f"Group overlap Train/Validation: {len(overlap)} group, ví dụ {examples}")
    return 0


def split_train_val(data, seed=42):
    """Tách Validation theo group_id; giữ nguyên Test và kiểm tra leakage cả ba partition."""
    from sklearn.model_selection import GroupShuffleSplit

    required = ("split", "groups", "y")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Thiếu trường dữ liệu: {missing}")

    split = np.asarray(data["split"])
    y = np.asarray(data["y"])
    if split.ndim != 1 or y.ndim != 1 or len(split) == 0:
        raise ValueError("split và y phải là mảng 1-D không rỗng")
    groups = _validated_group_ids(data["groups"], "groups")
    if not (len(split) == len(y) == len(groups)):
        raise ValueError("split, y và groups phải có cùng số phần tử")

    split = split.astype(str)
    unknown = sorted(set(split.tolist()) - {"train", "test"})
    if unknown:
        raise ValueError(f"Giá trị split không hợp lệ: {unknown}")
    tr_idx = np.flatnonzero(split == "train")
    te_idx = np.flatnonzero(split == "test")
    if len(tr_idx) == 0 or len(te_idx) == 0:
        raise ValueError("Cần có cả Train và Test không rỗng")
    if len(np.unique(groups[tr_idx])) < 2:
        raise ValueError("Cần ít nhất 2 group trong Train để tách Validation")

    train_test_overlap = set(groups[tr_idx]) & set(groups[te_idx])
    if train_test_overlap:
        raise ValueError(
            f"Group leakage Train/Test: {len(train_test_overlap)} group, "
            f"ví dụ {sorted(train_test_overlap)[:5]}"
        )

    val_fraction = CFG["data"]["val_fraction"]
    if not isinstance(val_fraction, (int, float)) or not np.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("data.val_fraction phải nằm trong khoảng (0, 1)")

    splitter = GroupShuffleSplit(n_splits=1, test_size=float(val_fraction), random_state=int(seed))
    try:
        train_local, val_local = next(splitter.split(tr_idx, y[tr_idx], groups[tr_idx]))
    except ValueError as exc:
        raise ValueError(f"Không thể tạo group-aware Train/Validation split: {exc}") from exc

    tr_out, va_idx = tr_idx[train_local], tr_idx[val_local]
    if len(tr_out) == 0 or len(va_idx) == 0:
        raise ValueError("Train hoặc Validation rỗng sau khi tách")
    check_group_disjoint(groups[tr_out], groups[va_idx])
    check_group_disjoint(groups[tr_out], groups[te_idx])
    check_group_disjoint(groups[va_idx], groups[te_idx])

    if len(np.unique(np.concatenate([tr_out, va_idx]))) != len(tr_idx):
        raise RuntimeError("Train/Validation không bảo toàn chính xác tập Train gốc")
    if not np.array_equal(np.sort(np.concatenate([tr_out, va_idx])), np.sort(tr_idx)):
        raise RuntimeError("Train/Validation không bao phủ đúng tập Train gốc")
    if not np.array_equal(te_idx, np.flatnonzero(split == "test")):
        raise RuntimeError("Tập Test đã bị thay đổi trong quá trình split")

    val_classes = np.unique(y[va_idx])
    train_classes = np.unique(y[tr_out])
    all_classes = np.unique(y)
    missing_train_classes = sorted(set(all_classes.tolist()) - set(train_classes.tolist()))
    if missing_train_classes:
        raise ValueError(
            f"Train partition misses classes after group split: {missing_train_classes}. "
            "Do not auto-reduce or silently change the split; review group/class feasibility."
        )
    missing_val_classes = sorted(set(all_classes.tolist()) - set(val_classes.tolist()))
    checks = {
        "nonempty_group_ids": True,
        "train_val_disjoint": True,
        "train_test_disjoint": True,
        "val_test_disjoint": True,
        "train_partition_preserved": True,
        "test_partition_unchanged": True,
    }
    meta = {
        "method": "GroupShuffleSplit",
        "seed": int(seed),
        "validation_fraction_of_train_groups": float(val_fraction),
        "validation_purpose": "model-selection validation (không phải Independent Test)",
        "n_train": int(len(tr_out)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "n_train_groups": int(len(np.unique(groups[tr_out]))),
        "n_val_groups": int(len(np.unique(groups[va_idx]))),
        "n_test_groups": int(len(np.unique(groups[te_idx]))),
        "group_overlap_train_val": 0,
        "group_overlap_train_test": 0,
        "group_overlap_val_test": 0,
        "leakage_checks": checks,
        "leakage_check": "PASS" if all(checks.values()) else "FAIL",
        "train_class_counts": {str(c): int(n) for c, n in zip(*np.unique(y[tr_out], return_counts=True))},
        "val_class_counts": {str(c): int(n) for c, n in zip(*np.unique(y[va_idx], return_counts=True))},
        "train_missing_classes": missing_train_classes,
        "val_missing_classes": missing_val_classes,
    }
    return {"train": tr_out, "val": va_idx, "test": te_idx, "meta": meta}
