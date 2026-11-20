"""
Sinh N phiếu HỢP LỆ giả lập (nhãn `valid`) bằng pipeline của generate_dataset.py v2.0
(builder _b_valid + oracle kiểm tra nhãn + nhiễu scan), nên ảnh cùng phân phối với bộ gốc:
đủ 4 mức khó easy/medium/hard/boundary, nhiều kiểu dấu, có cả phiếu "gạch sửa rồi vẫn hợp lệ".

Dữ liệu GIẢ LẬP (BR-04). "valid" là nhãn hình thái, KHÔNG phải legal_label LEGAL_VALID (BR-06).

Đặt file này CẠNH generate_dataset.py, rồi chạy:
    python generate_valid.py
    python generate_valid.py --n 500 --seed 7 --out ./out_test
Đầu ra: <out>/data/valid_500/{raw,labels,metadata}/
"""
import argparse
import collections
import csv
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_dataset as gd  # noqa: E402

N_DEFAULT = 500
LABEL = "valid"


def generate_valid(out_root=gd.DEFAULT_OUT, n=N_DEFAULT, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    base_dir = os.path.join(out_root, "data", f"valid_{n}" if n != N_DEFAULT else "valid_500")
    raw_dir = os.path.join(base_dir, "raw")
    lab_dir = os.path.join(base_dir, "labels")
    meta_dir = os.path.join(base_dir, "metadata")
    for d in (raw_dir, lab_dir, meta_dir):
        os.makedirs(d, exist_ok=True)

    # Kế hoạch mức khó theo đúng tỉ lệ DIFF_RATIO của bộ gốc
    plan = [gd.DIFFS[k] for k, q in enumerate(gd._quota(n, gd.DIFF_RATIO)) for _ in range(q)]
    rng.shuffle(plan)

    stats = collections.Counter()
    rows, jsonl, boxes_json = [], [], {}
    for i in range(n):
        voter = gd.random_voter()
        img, boxes = gd.draw_ballot_template(gd.CANDIDATE_NAMES, voter)
        base = gd._add_background(img, rng)
        anchor = int(rng.integers(gd.NC))
        diff = plan[i]

        marks, case, res = gd._make_sample(LABEL, diff, anchor, base, boxes, rng, stats)
        arr = gd._compose(base, marks)
        out_img, rboxes, params = gd._finish(arr, boxes, rng)

        fname = f"valid_{i:04d}.png"
        out_img.save(os.path.join(raw_dir, fname), compress_level=1)
        boxes_json[fname] = {"boxes": rboxes[:gd.NC], "left_band": rboxes[gd.NC],
                             "extra_zone": rboxes[gd.NC + 1]}

        marked = sorted(set(res["votes"]) | set(res["struck"]))
        meta = gd._mark_metadata(marks, res, boxes)
        row = {"file": fname, "label": LABEL, "split": "", "loai_loi": LABEL,
               "so_o_danh_dau": len(marked), "o_co_danh_dau": "|".join(map(str, marked)),
               "group_id": f"v{i:04d}", "difficulty": diff, "case_type": case}
        row.update(meta)
        row.update(params)
        rows.append(row)
        jsonl.append({"file": fname, "label": LABEL, "difficulty": diff, "case_type": case,
                      "anchor": anchor, "template_boxes": boxes,
                      "marks": [m["spec"] for m in marks], "oracle": res,
                      "params": params, "meta": meta})
        if (i + 1) % 50 == 0:
            print(f"  đã sinh {i + 1}/{n}")

    with open(os.path.join(lab_dir, "all.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=gd.CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(lab_dir, "boxes.json"), "w", encoding="utf-8") as f:
        json.dump(boxes_json, f, ensure_ascii=False)
    with open(os.path.join(meta_dir, "samples.jsonl"), "w", encoding="utf-8") as f:
        for rec in jsonl:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Đã sinh {len(rows)} ảnh hợp lệ -> {raw_dir}")
    print("Thống kê sinh mẫu:", dict(stats))
    return base_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sinh phiếu hợp lệ giả lập (nhãn valid).")
    p.add_argument("--n", type=int, default=N_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=gd.DEFAULT_OUT)
    a = p.parse_args()
    generate_valid(a.out, a.n, a.seed)