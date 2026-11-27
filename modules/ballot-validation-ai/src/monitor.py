"""
MLOPS — GIÁM SÁT:
  * psi()/monitor(): phát hiện lệch phân phối đặc trưng giữa dữ liệu huấn luyện và dữ liệu sản xuất
      PSI = Σ (a − e)·ln(a / e); < 0.1 ổn định, > 0.25 lệch nặng → cân nhắc huấn luyện lại.
  * log_run(): ghi số liệu mỗi lần chạy vào một file JSON (lịch sử thí nghiệm).
  * check_drift(): so sánh số liệu lần chạy mới nhất với trung bình các lần trước; cảnh báo nếu giảm mạnh.
"""
import json
import os

import numpy as np
import pandas as pd

PSI_THRESHOLD = 0.25


def psi(expected, actual, n_bins=10):
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    if not expected.size or not actual.size:
        raise ValueError("PSI cần dữ liệu không rỗng")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))  # bỏ biên trùng
    if len(edges) < 2:
        return 0.0                       # đặc trưng hằng số: không đo được lệch
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def monitor(train_csv, prod_csv):
    tr = pd.read_csv(train_csv).drop(columns=["label"], errors="ignore")
    pr = pd.read_csv(prod_csv).drop(columns=["label"], errors="ignore")
    alerts = {c: round(psi(tr[c].values, pr[c].values), 3) for c in tr.columns}
    alerts = {c: v for c, v in alerts.items() if v > PSI_THRESHOLD}
    print(f"DRIFT: {alerts} → cân nhắc huấn luyện lại" if alerts else "Không có drift đáng kể")
    return alerts


def log_run(metrics, path):
    runs = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            runs = json.load(f)
    runs.append(dict(metrics, run_index=len(runs)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=1)
    return runs


def check_drift(path, key="hybrid_acc", tol=0.05):
    with open(path, encoding="utf-8") as f:
        runs = json.load(f)
    past = [r[key] for r in runs[:-1] if key in r]   # lần chạy cũ có thể chưa ghi chỉ số này
    if not past or key not in runs[-1]:
        print("Chưa đủ lịch sử để so sánh.")
        return False
    last = runs[-1][key]
    drop = float(np.median(past)) - last
    warn = drop > tol
    print(f"{key}: lần này={last:.4f}, trung vị trước đó={np.median(past):.4f} → "
          + ("⚠ GIẢM BẤT THƯỜNG, cân nhắc kiểm tra dữ liệu/huấn luyện lại" if warn else "bình thường"))
    return warn
