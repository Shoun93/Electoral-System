"""
KIỂM TRA PHIẾU: nhập đường dẫn ảnh → cho biết phiếu có HỢP LỆ không và LỖI GÌ.

Chạy trong modules/ballot-validation-ai/ (cần một run hoàn tất của main.py trong results/runs/):
    python predict.py                  # chế độ nhập: gõ (hoặc kéo-thả) đường dẫn ảnh, Enter; để trống để thoát
    python predict.py anh.png          # một ảnh
    python predict.py thu_muc/ --csv ket_qua.csv

Với mỗi ảnh: in kết luận + lỗi + lý do bằng tiếng Việt, và lưu ảnh đã đánh dấu vào results/annotated/.
Giả định: thẻ cử tri cùng bố cục với bộ sinh dữ liệu (ô ở vị trí mẫu).
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import cv2
import joblib
import numpy as np
import torch

from config import CFG, abspath
from preprocess import binarize_fullres, crop_boxes
from feature_c45 import extract_features
from train_cnn import MarkCNN, predict_marks_ensemble, DEVICE
from stray_cnn import StrayNet, maxpool_input, predict_ensemble
import manifest as mf
from decision import confidences, is_pending

IMG_W, IMG_H = 1200, 850
BOX_X1, BOX_X2, BOX_SIZE = 671, 707, 36
CAND_Y0, CAND_STEP, NAME_X, N_BOX = 370, 43, 722, 5
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Tên lỗi (tiếng Việt) cho từng nhãn
ERROR_NAME = {
    "valid": "Không có lỗi",
    "blank": "Phiếu trắng (không đánh dấu ô nào)",
    "erased": "Gạch xóa toàn bộ / nhiều ô bị gạch",
    "wrong_mark": "Đánh dấu sai vị trí hoặc ghi thêm tên ngoài danh sách",
    "multi_mark": "Chọn quá số đại biểu cho phép (tối đa 3)",
}
LEFT_BAND_THR, EXTRA_ZONE_THR = 0.011, 0.022   # ngưỡng lấy từ luật C4.5 trong README


def default_layout():
    """Bố cục mẫu (ảnh chưa xoay), cùng công thức với dataset/generate_dataset.py."""
    boxes = []
    for i in range(N_BOX):
        yc = CAND_Y0 + i * CAND_STEP
        boxes.append([BOX_X1, yc - BOX_SIZE // 2, BOX_X2, yc + BOX_SIZE // 2])
    top, bottom = boxes[0][1] - 6, boxes[-1][3] + 6
    return {"boxes": boxes,
            "left_band": [BOX_X1 - 70, top, BOX_X1 - 24, bottom],
            "extra_zone": [NAME_X, boxes[-1][3] + 25, IMG_W - 40, boxes[-1][3] + 60]}


def hybrid_features(F, p_marks):
    cnn = np.stack([(p_marks > 0.5).sum(1), p_marks.mean(1), p_marks.max(1)], axis=1)
    return np.hstack([F, cnn]).astype(np.float64)


def load_models():
    """Mô hình và ngưỡng lấy từ cùng một run hoàn tất gần nhất."""
    try:
        run_dir = mf.latest_run_dir(abspath(CFG["results_dir"]))
    except OSError as e:
        sys.exit(f"Chưa có run hoàn tất ({e}). Chạy main.py trước.")
    c45_path, mark_path, stray_path = (os.path.join(run_dir, "models", n)
                                       for n in ("hybrid_c45.joblib", "mark_cnn.pt", "stray_cnn.pt"))
    for p in (c45_path, mark_path, stray_path):
        if not os.path.exists(p):
            sys.exit(f"Thiếu {p}. Chạy main.py trước.")
    th = load_thresholds(os.path.join(run_dir, "metrics.json"))
    return joblib.load(c45_path), (_load_ensemble(MarkCNN, mark_path), _load_ensemble(StrayNet, stray_path)), th


def _load_ensemble(cls, path):
    models = []
    for state in torch.load(path, map_location=DEVICE):
        model = cls()
        model.load_state_dict(state)
        models.append(model.to(DEVICE if cls is MarkCNN else "cpu").eval())
    if not models:
        sys.exit(f"{path} không chứa mô hình nào")
    return models


def load_thresholds(path):
    """Thiếu ngưỡng thì dừng: mặc định 0 sẽ tự động phân loại mọi phiếu."""
    try:
        with open(path, encoding="utf-8") as f:
            sel = json.load(f)["hybrid_selective_observed_test"]
        return {k: sel[k] for k in ("tau_cnn", "tau_c45", "leaf_min_n", "tau_stray")}
    except (OSError, ValueError, KeyError, TypeError) as e:
        sys.exit(f"Không đọc được ngưỡng PENDING_REVIEW từ {path} ({e!r}). Chạy main.py trước.")


def explain(label, marked, left_ink, extra_ink):
    """Lý do cụ thể bằng tiếng Việt."""
    n = len(marked)
    names = ", ".join(f"đại biểu {i + 1}" for i in marked)
    if label == "valid":
        return f"Đã chọn {n} đại biểu ({names}), đúng quy định 1–3."
    if label == "blank":
        return "Không phát hiện dấu nào trong 5 ô."
    if label == "multi_mark":
        return f"Chọn {n} đại biểu ({names}), vượt mức tối đa 3."
    if label == "erased":
        return "Phát hiện nét gạch/xóa phủ nhiều ô trong danh sách."
    parts = []
    if left_ink > LEFT_BAND_THR:
        parts.append("có dấu ở dải bên trái ô (không nằm trong ô)")
    if extra_ink > EXTRA_ZONE_THR:
        parts.append("có chữ/nét ghi thêm dưới danh sách")
    return "; ".join(parts).capitalize() + "." if parts else \
        "Dấu nằm ngoài ô hoặc có ghi thêm (mô hình không xác định được vị trí cụ thể)."


def annotate(img_gray, layout, p_marks, label, out_path, conflict=False):
    """Vẽ ô có dấu (xanh), ô trống (xám), vùng ngoài ô (cam) và kết luận (chữ không dấu vì cv2)."""
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for i, b in enumerate(layout["boxes"]):
        on = p_marks[i] > 0.5
        col = (0, 160, 0) if on else (150, 150, 150)
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 3 if on else 1)
        cv2.putText(vis, f"{p_marks[i]:.2f}", (int(b[0]) - 60, int((b[1] + b[3]) / 2) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    for key in ("left_band", "extra_zone"):
        b = layout[key]
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 140, 255), 1)
    ok = label == "valid" and not conflict
    text = "CAN XAC MINH" if conflict else "HOP LE" if ok else "KHONG HOP LE"
    cv2.putText(vis, text + f" [{label}]", (20, IMG_H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 160, 0) if ok else (0, 0, 220), 3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def predict_one(path, c45, cnn, th, layouts):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"file": os.path.basename(path), "error": "không đọc được ảnh"}
    if img.shape != (IMG_H, IMG_W):
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    layout = layouts.get(os.path.basename(path), default_layout())
    binary = binarize_fullres(img)
    F = np.asarray([extract_features(binary, layout)], dtype=np.float32)
    C = crop_boxes(binary, layout["boxes"])[None]
    mark_models, stray_models = cnn
    p_marks = predict_marks_ensemble(mark_models, C)
    p_stray = predict_ensemble(stray_models, maxpool_input(binary)[None])
    H = hybrid_features(F, p_marks)
    proba = c45.predict_proba(H)
    k = int(proba.argmax(1)[0])
    cc, cd = confidences(p_marks, proba)
    pend = bool(is_pending(cc, cd, th["tau_cnn"], th["tau_c45"],
                           leaf_n=c45.leaf_size(H), leaf_min_n=th["leaf_min_n"],
                           extra=p_stray >= th["tau_stray"])[0])
    label = CFG["labels"][k]
    marked = [int(i) for i in np.where(p_marks[0] > 0.5)[0]]
    conflict = label == "valid" and not 1 <= len(marked) <= 3     # C4.5 nói hợp lệ nhưng số ô có dấu trái quy tắc
    pend = pend or conflict
    name = os.path.splitext(os.path.basename(path))[0]
    out_img = abspath(os.path.join(CFG["results_dir"], "annotated", name + "_ketqua.png"))
    annotate(img, layout, p_marks[0], label, out_img, conflict)
    return {"file": os.path.basename(path), "label": label,
            "valid": "CHƯA XÁC ĐỊNH" if conflict else "CÓ" if label == "valid" else "KHÔNG",
            "error": "Cần người xác minh" if conflict else ERROR_NAME[label],
            "reason": (f"Mâu thuẫn: dự đoán hợp lệ nhưng phát hiện {len(marked)} ô có dấu (quy định 1–3)."
                       if conflict else explain(label, marked, float(F[0, 6]), float(F[0, 7]))),
            "confidence": round(float(proba[0, k]), 4),
            "status": "PENDING_REVIEW (nên để người xác minh)" if pend else "AUTO_CLASSIFIED",
            "marked_boxes": marked, "annotated": out_img}


def show(r):
    if "reason" not in r:
        print(f"[{r['file']}] LỖI: {r['error']}")
        return
    print(f"\n=== {r['file']} ===")
    print(f"  Hợp lệ?   : {r['valid']}")
    print(f"  Lỗi       : {r['error']}")
    print(f"  Lý do     : {r['reason']}")
    print(f"  Tin cậy   : {r['confidence']}  |  {r['status']}")
    print(f"  Ảnh đánh dấu: {r['annotated']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="file ảnh hoặc thư mục; bỏ trống để nhập tay")
    ap.add_argument("--layout", help="boxes.json (khóa theo tên file) nếu có")
    ap.add_argument("--csv", help="ghi kết quả ra CSV")
    a = ap.parse_args()

    layouts = {}
    if a.layout:
        with open(a.layout, encoding="utf-8") as f:
            layouts = json.load(f)
    c45, cnn, th = load_models()

    if a.input is None:                                  # chế độ nhập tay
        print("Nhập đường dẫn ảnh (hoặc kéo-thả file vào đây), Enter để kiểm tra; để trống để thoát.")
        while True:
            p = input("\nẢnh > ").strip().strip('"').strip("'")
            if not p:
                break
            if not os.path.isfile(p):
                print("Không tìm thấy file.")
                continue
            show(predict_one(p, c45, cnn, th, layouts))
        return

    if os.path.isdir(a.input):
        paths = sorted(os.path.join(a.input, f) for f in os.listdir(a.input) if f.lower().endswith(EXTS))
    else:
        paths = [a.input]
    if not paths:
        sys.exit(f"Không có ảnh {', '.join(EXTS)} trong {a.input}")
    results = [predict_one(p, c45, cnn, th, layouts) for p in paths]
    for r in results:
        show(r)
    if a.csv:
        ok = [r for r in results if "reason" in r]
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "label", "valid", "error", "reason", "confidence",
                                              "status", "marked_boxes", "annotated"])
            w.writeheader()
            w.writerows(ok)
        print("\nĐã ghi", a.csv)


if __name__ == "__main__":
    main()
