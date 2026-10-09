"""
PIPELINE END-TO-END
  1. Sinh / đọc dữ liệu phiếu giả lập (dataset/generate_dataset.py) → tiền xử lý
  2. Baseline (majority, Naive Bayes, kNN) trên 8 đặc trưng thủ công
  3. C4.5 chỉ đặc trưng thủ công                       (đối chứng)
  4. CNN toàn ảnh 224×224                                (đối chứng — xem báo cáo)
  5. MÔ HÌNH LAI: CNN phát hiện dấu trên từng ô (out-of-fold) → đặc trưng → C4.5 ra quyết định
  6. Đánh giá trên tập TEST; chọn mô hình KHÔNG dựa trên test
  7. Ghi kết quả từng phiếu (kèm SHA-256 ảnh) lên hash-chain, neo mã băm mô hình
  8. Ghi lịch sử và kiểm tra suy giảm (MLOps)

Chạy trong thư mục ballot-validation-ai/:  python main.py
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from config import CFG, abspath
from preprocess import generate_synthetic_ballots, load_dataset, split_train_val
from baselines import train_baselines
from c45 import C45Classifier
from feature_c45 import FEATURE_NAMES
from train_c45 import train_c45
from train_cnn import train_cnn, predict_cnn, train_mark_cnn, oof_mark_probs, predict_marks
from evaluate import summarize
from hashchain import HashChain, sha256_file
import merkle
from monitor import check_drift, log_run


def hybrid_features(F, p_marks):
    """Đặc trưng cho C4.5 của mô hình lai: 8 đặc trưng thủ công + thống kê xác suất CNN từng ô."""
    thr = 0.5
    cnn_feats = np.stack([(p_marks > thr).sum(1), p_marks.mean(1), p_marks.max(1)], axis=1)
    return np.hstack([F, cnn_feats]).astype(np.float64)


HYBRID_NAMES = FEATURE_NAMES + ["cnn_count_marked", "cnn_mean_prob", "cnn_max_prob"]


def main():
    labels = CFG["labels"]
    results_dir = abspath(CFG["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    # 1. Dữ liệu
    if not os.path.exists(abspath(CFG["data"]["labels_csv"])):
        print("Chưa có dữ liệu → sinh bộ phiếu giả lập")
        generate_synthetic_ballots()
    data = load_dataset()
    idx = split_train_val(data)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    X, F, C, M, y, files = data["X"], data["F"], data["C"], data["M"], data["y"], data["files"]
    print(f"Phiếu: train={len(tr)} val={len(va)} test={len(te)}")

    metrics = {}

    # 2. Baseline
    for name, m in train_baselines(F[tr], y[tr]).items():
        metrics[f"baseline_{name}"] = summarize(f"Baseline {name}", y[te], m.predict(F[te]))

    # 3. C4.5 chỉ đặc trưng thủ công
    print("\n----- C4.5 (đặc trưng thủ công) -----")
    c45_manual = train_c45(F[tr], y[tr], F[va], y[va], save=False)
    metrics["c45_manual"] = summarize("C4.5 (đặc trưng thủ công)", y[te], c45_manual.predict(F[te]))

    # 4. CNN toàn ảnh (đối chứng)
    print("\n----- CNN toàn ảnh 224×224 (đối chứng) -----")
    cnn_full = train_cnn(X[tr], y[tr], X[va], y[va])
    pred_full, _ = predict_cnn(cnn_full, X[te])
    metrics["cnn_full_image"] = summarize("CNN toàn ảnh", y[te], pred_full)

    # 5. Mô hình lai
    print("\n----- CNN phát hiện dấu theo ô (out-of-fold) + C4.5 -----")
    p_marks_tr = oof_mark_probs(C[tr], M[tr])                      # out-of-fold cho train
    mark_model = train_mark_cnn(C[tr], M[tr])
    p_marks_va = predict_marks(mark_model, C[va])
    p_marks_te = predict_marks(mark_model, C[te])
    mark_acc_te = float(((p_marks_te > 0.5).astype(int) == M[te]).mean())
    print(f"CNN phát hiện dấu trên ô (test, từng ô): accuracy={mark_acc_te:.4f}")
    metrics["mark_cnn_box_accuracy"] = round(mark_acc_te, 4)

    H_tr = hybrid_features(F[tr], p_marks_tr)
    H_va = hybrid_features(F[va], p_marks_va)
    H_te = hybrid_features(F[te], p_marks_te)
    hybrid = train_c45(H_tr, y[tr], H_va, y[va], feature_names=HYBRID_NAMES, tag="hybrid_c45")
    pred_h = hybrid.predict(H_te)
    p_h = hybrid.predict_proba(H_te)
    metrics["hybrid_cnn_c45"] = summarize("MÔ HÌNH LAI (CNN ô + C4.5)", y[te], pred_h)

    # 6. Ghi kết quả chi tiết từng phiếu
    with open(os.path.join(results_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "true", "pred", "confidence"])
        for k, i in enumerate(te):
            wr.writerow([files[i], labels[y[i]], labels[pred_h[k]], round(float(p_h[k].max()), 4)])

    # 7. Blockchain: neo mã băm mô hình + manifest, rồi ghi từng phiếu
    model_dir = abspath(CFG["models_dir"])
    chain = HashChain(difficulty=CFG["hashchain"]["difficulty"])
    chain.add_record({"type": "model_anchor",
                      "hybrid_c45_sha256": sha256_file(os.path.join(model_dir, "hybrid_c45.joblib")),
                      "manifest_sha256": sha256_file(abspath(CFG["data"]["labels_csv"]))})
    raw_dir = abspath(CFG["data"]["raw_dir"])
    for k, i in enumerate(te):
        chain.add_ballot(ballot_id=files[i], pred=labels[pred_h[k]], conf=p_h[k].max(),
                         image_sha256=sha256_file(os.path.join(raw_dir, files[i])))
    # Lô phiếu cho sổ cái Fabric: gốc Merkle để chaincode CommitBatch / VerifyBatch
    leaves = [merkle.leaf_hash(files[i], labels[pred_h[k]], p_h[k].max(),
                               sha256_file(os.path.join(raw_dir, files[i])))
              for k, i in enumerate(te)]
    batch = {"batch_id": "test-" + str(len(te)), "root": merkle.root(leaves), "count": len(leaves), "leaves": leaves}
    with open(os.path.join(results_dir, "fabric_batch.json"), "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=1)

    chain_path = os.path.join(results_dir, "hashchain.json")
    chain.save(chain_path)
    print(f"\n[blockchain] {len(chain.blocks)} block | chuỗi hợp lệ: {chain.verify()} → {chain_path}")

    demo = HashChain.load(chain_path)                       # kiểm chứng: sửa một phiếu trong bản sao
    target = next(b.index for b in demo.blocks if "ballot_id" in b.data)
    old = demo.blocks[target].data["pred"]
    demo.tamper(target, "pred", "invalid" if old == "valid" else "valid")
    print(f"[blockchain] Demo sửa kết quả block {target} ({old} → đổi) → chuỗi còn hợp lệ: {demo.verify()}")

    # 8. MLOps
    log_path = os.path.join(results_dir, "metrics_log.json")
    log_run({"c45_manual_acc": metrics["c45_manual"]["accuracy"],
             "cnn_full_acc": metrics["cnn_full_image"]["accuracy"],
             "hybrid_acc": metrics["hybrid_cnn_c45"]["accuracy"]}, log_path)
    check_drift(log_path, key="hybrid_acc")

    with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print("\n✅ PIPELINE HOÀN TẤT. Xem results/ và models/")


if __name__ == "__main__":
    main()
