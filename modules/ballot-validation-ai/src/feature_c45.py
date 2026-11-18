"""
TRÍCH 8 ĐẶC TRƯNG CẤU TRÚC cho C4.5 (đo trên ảnh nhị phân ở ĐỘ PHÂN GIẢI GỐC).
Tọa độ các vùng lấy từ data/labels/boxes.json (đã áp phép xoay nhiễu của từng ảnh):
  1. count_ink_boxes  — số trong 5 ô có dấu (mật độ mực > ngưỡng)      valid: 1–3, multi_mark: 4–5, blank: 0
  2. mean_box_density — mật độ mực trung bình 5 ô
  3. max_box_density  — mật độ mực cao nhất trong 5 ô                   erased: nhiều ô cao
  4. total_ink        — tỉ lệ mực trên toàn ảnh
  5. edge_density     — mật độ biên (số chuyển 0↔1 / diện tích)
  6. noise_level      — std phần nhiễu cao tần
  7. left_band_ink    — mật độ mực ở dải bên trái ô (dấu đánh sai vị trí)     wrong_mark
  8. extra_zone_ink   — mật độ mực ở vùng ghi thêm tên                      wrong_mark
Ô và vùng được thu nhỏ vào trong `inner_margin` px để bỏ viền ô.
"""
import cv2
import numpy as np

from config import CFG

FEATURE_NAMES = [
    "count_ink_boxes", "mean_box_density", "max_box_density", "total_ink",
    "edge_density", "noise_level", "left_band_ink", "extra_zone_ink",
]


def _density(binary, rect, margin):
    x1, y1, x2, y2 = (int(round(v)) for v in rect)
    x1, y1, x2, y2 = x1 + margin, y1 + margin, x2 - margin, y2 - margin
    patch = binary[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    return float(patch.mean()) if patch.size else 0.0


def extract_features(binary, layout):
    """
    binary : mảng uint8 0/1 (H×W), 1 = mực, ở độ phân giải gốc.
    layout : dict {"boxes": [5 box], "left_band": box, "extra_zone": box} của phiếu này.
    Trả về list 8 số float theo thứ tự FEATURE_NAMES.
    """
    m = CFG["features"]["inner_margin"]
    thr = CFG["features"]["box_ink_threshold"]
    H, W = binary.shape
    b = binary.astype(np.float32)

    dens = [_density(b, box, m) for box in layout["boxes"]]
    count = sum(d > thr for d in dens)
    mean_d, max_d = float(np.mean(dens)), float(np.max(dens))

    total = float(b.mean())
    bi = binary.astype(np.int16)  # uint8 làm 0-1 tràn thành 255
    edges = (np.abs(np.diff(bi, axis=1)).sum() + np.abs(np.diff(bi, axis=0)).sum()) / (H * W)
    noise = float(np.std(b - cv2.GaussianBlur(b, (3, 3), 0)))

    left = _density(b, layout["left_band"], 0)
    extra = _density(b, layout["extra_zone"], 0)
    return [float(count), mean_d, max_d, total, float(edges), noise, left, extra]
