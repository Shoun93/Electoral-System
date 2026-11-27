"""
PIPELINE AI — Phase 1 safeguards.
Full model/OOF execution is blocked until PM policy approval and separate OOF authorization.
Observed Test is descriptive only, never an acceptance/selection source.
"""
import csv, hashlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np
import torch
from config import CFG, abspath
from preprocess import load_dataset, split_train_val
from cache import compute_cache_key, load_mark_prob_cache, save_mark_prob_cache
from baselines import train_baselines
from train_c45 import train_c45
from train_cnn import train_cnn, predict_cnn, train_mark_cnn, oof_mark_probs, predict_marks
from evaluate import summarize
from decision import PolicyNotApprovedError, confidences, feasible, grid_search, is_pending, screening_report
from hashchain import HashChain, sha256_file
import merkle
from monitor import check_drift, log_run
from feature_c45 import FEATURE_NAMES

HYBRID_NAMES = FEATURE_NAMES + ["cnn_count_marked", "cnn_mean_prob", "cnn_max_prob"]

def hybrid_features(F, p_marks):
    p_marks = np.asarray(p_marks)
    cnn_feats = np.stack([(p_marks > 0.5).sum(1), p_marks.mean(1), p_marks.max(1)], axis=1)
    return np.hstack([F, cnn_feats]).astype(np.float64)

def _require_pipeline_authorization():
    decision = CFG.get("decision", {})
    if decision.get("acceptance_policy_approved") is not True:
        raise PolicyNotApprovedError("Pipeline bị khóa: PM chưa phê duyệt định nghĩa INVALID và các ngưỡng nghiệm thu.")
    if decision.get("oof_selection_authorized") is not True:
        raise PolicyNotApprovedError("Pipeline bị khóa: OOF/model-threshold selection cần PM phê duyệt riêng.")
    if decision.get("nested_oof_selection_implemented") is not True:
        raise PolicyNotApprovedError("Pipeline bị khóa: nested group-aware OOF chưa được triển khai; flat OOF selection bị cấm.")

def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    _require_pipeline_authorization()
    labels = CFG["labels"]
    results_dir = abspath(CFG["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    labels_path = abspath(CFG["data"]["labels_csv"])
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Không tìm thấy manifest dữ liệu: {labels_path}. Phase 1 không tự động sinh/thay đổi dataset.")

    data = load_dataset()
    idx = split_train_val(data, seed=CFG.get("cache", {}).get("seed", 42))
    tr, va, te = idx["train"], idx["val"], idx["test"]
    X, F, C, M, y = data["X"], data["F"], data["C"], data["M"], data["y"]
    files, groups = data["files"], data["groups"]
    print(f"train={len(tr)} val={len(va)} observed_test={len(te)} | leakage={idx['meta']['leakage_check']}")
    metrics = {"validation_split": idx["meta"], "test_partition_role": "OBSERVED_TEST_DESCRIPTIVE_ONLY"}

    for name, model in train_baselines(F[tr], y[tr]).items():
        metrics[f"baseline_{name}"] = summarize(f"Baseline {name}", y[te], model.predict(F[te]))
    c45_manual = train_c45(F[tr], y[tr], F[va], y[va], save=False)
    metrics["c45_manual"] = summarize("C4.5 (đặc trưng thủ công)", y[te], c45_manual.predict(F[te]))
    cnn_full = train_cnn(X[tr], y[tr], X[va], y[va])
    pred_full, _ = predict_cnn(cnn_full, X[te])
    metrics["cnn_full_image"] = summarize("CNN toàn ảnh", y[te], pred_full)

    cache_path = os.path.join(results_dir, "mark_probs_cache.npz")
    mark_model_path = abspath(os.path.join(CFG["models_dir"], "mark_cnn.pt"))
    cache_inputs = {
        "C_train": C[tr], "M_train": M[tr], "groups_train": groups[tr], "labels_train": y[tr],
        "C_val": C[va], "C_test": C[te], "files": files, "split": data["split"],
        # Đổi mã huấn luyện thì cache cũng hết hiệu lực.
        "train_code": np.asarray(_file_sha256(abspath("src/train_cnn.py"))),
    }
    cache_cfg = dict(CFG["mark_cnn"])
    seed = int(CFG.get("cache", {}).get("seed", 42))
    folds = int(CFG["mark_cnn"].get("folds", 5))
    model_hash = _file_sha256(mark_model_path) if os.path.isfile(mark_model_path) else None
    expected_rows = {"tr": len(tr), "va": len(va), "te": len(te)}
    key = compute_cache_key(cache_inputs, cache_cfg, model_hash, seed=seed, folds=folds)
    cached = load_mark_prob_cache(cache_path, key, expected_rows) if model_hash else None

    if cached is not None:
        p_marks_tr, p_marks_va, p_marks_te = cached["tr"], cached["va"], cached["te"]
        print("[cache] hit: data/config/checkpoint checksums match")
    else:
        p_marks_tr = oof_mark_probs(C[tr], M[tr], groups=groups[tr], strat=y[tr], k=folds, seed=seed)
        mark_model = train_mark_cnn(C[tr], M[tr], cfg=cache_cfg)
        os.makedirs(os.path.dirname(mark_model_path), exist_ok=True)
        torch.save(mark_model.state_dict(), mark_model_path)
        p_marks_va = predict_marks(mark_model, C[va])
        p_marks_te = predict_marks(mark_model, C[te])
        model_hash = _file_sha256(mark_model_path)
        key = compute_cache_key(cache_inputs, cache_cfg, model_hash, seed=seed, folds=folds)
        save_mark_prob_cache(cache_path, key, p_marks_tr, p_marks_va, p_marks_te)

    metrics["mark_cnn_box_accuracy_observed_test"] = round(float(((p_marks_te > 0.5).astype(int) == M[te]).mean()), 4)
    def make_hybrid(F_part, p_part):
        feats = np.stack([(p_part > 0.5).sum(1), p_part.mean(1), p_part.max(1)], axis=1)
        return np.hstack([F_part, feats]).astype(np.float64)
    H_tr, H_va, H_te = make_hybrid(F[tr], p_marks_tr), make_hybrid(F[va], p_marks_va), make_hybrid(F[te], p_marks_te)

    best = grid_search(H_tr, y[tr], groups[tr], p_marks_tr)
    hybrid = train_c45(H_tr, y[tr], H_va, y[va], feature_names=HYBRID_NAMES,
                       tag="hybrid_c45", hp={"min_leaf": best["min_leaf"], "cf": best["cf"]})
    pred_h, p_h = hybrid.predict(H_te), hybrid.predict_proba(H_te)
    metrics["hybrid_cnn_c45_observed_test"] = summarize("MÔ HÌNH LAI — Observed Test", y[te], pred_h)
    cc_te, cd_te = confidences(p_marks_te, p_h)
    pend_te = is_pending(cc_te, cd_te, best["tau_cnn"], best["tau_c45"],
                         leaf_n=hybrid.leaf_size(H_te), leaf_min_n=best["leaf_min_n"])
    sel = screening_report(y[te], pred_h, pend_te, invalid_labels=CFG["decision"]["invalid_labels"])
    sel.update(tau_cnn=best["tau_cnn"], tau_c45=best["tau_c45"], leaf_min_n=best["leaf_min_n"],
               min_leaf=best["min_leaf"], cf=best["cf"], threshold_source="OOF_TRAIN_ONLY",
               evaluation_partition="OBSERVED_TEST_DESCRIPTIVE_ONLY", acceptance_status="NOT_AN_ACCEPTANCE_RESULT")
    sel["acceptance_feasible"] = feasible(sel)
    metrics["hybrid_selective_observed_test"] = sel
    print(f"[Observed Test — descriptive only] {sel}")

    with open(os.path.join(results_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "true", "pred", "confidence", "status", "partition_role"])
        for k, index in enumerate(te):
            status = "PENDING_REVIEW" if pend_te[k] else "AUTO_CLASSIFIED"
            writer.writerow([files[index], labels[y[index]], labels[pred_h[k]], round(float(p_h[k].max()), 4), status, "OBSERVED_TEST"])

    model_dir = abspath(CFG["models_dir"])
    chain = HashChain(difficulty=CFG["hashchain"]["difficulty"])
    chain.add_record({"type": "model_anchor", "hybrid_c45_sha256": sha256_file(os.path.join(model_dir, "hybrid_c45.joblib")),
                      "manifest_sha256": sha256_file(labels_path)})
    raw_dir = abspath(CFG["data"]["raw_dir"])
    pred_label = ["PENDING_REVIEW" if pend_te[k] else labels[pred_h[k]] for k in range(len(te))]
    for k, index in enumerate(te):
        chain.add_ballot(ballot_id=files[index], pred=pred_label[k], conf=p_h[k].max(),
                         image_sha256=sha256_file(os.path.join(raw_dir, files[index])))
    leaves = [merkle.leaf_hash(files[index], pred_label[k], p_h[k].max(), sha256_file(os.path.join(raw_dir, files[index])))
              for k, index in enumerate(te)]
    root = merkle.root(leaves)   # batch_id kèm gốc: chaincode ghi một lần, hai lô khác nhau không được trùng khóa
    batch = {"batch_id": f"observed-test-{len(te)}-{root[:16]}", "root": root, "count": len(leaves), "leaves": leaves}
    with open(os.path.join(results_dir, "fabric_batch.json"), "w", encoding="utf-8") as stream:
        json.dump(batch, stream, ensure_ascii=False, indent=1)
    chain_path = os.path.join(results_dir, "hashchain.json")
    chain.save(chain_path)
    print(f"[blockchain] {len(chain.blocks)} block | valid={chain.verify()} → {chain_path}")

    log_path = os.path.join(results_dir, "metrics_log.json")
    log_run({"c45_manual_acc": metrics["c45_manual"]["accuracy"], "cnn_full_acc": metrics["cnn_full_image"]["accuracy"],
             "hybrid_acc_observed_test": metrics["hybrid_cnn_c45_observed_test"]["accuracy"],
             "coverage_observed_test": sel["coverage"]}, log_path)
    check_drift(log_path, key="hybrid_acc_observed_test")
    with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=1)
    print("PIPELINE HOÀN TẤT. Observed Test chỉ có giá trị mô tả; không phải Independent Test.")

if __name__ == "__main__":
    main()
