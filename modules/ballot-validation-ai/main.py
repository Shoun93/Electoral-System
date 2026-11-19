"""
PIPELINE END-TO-END  (Phase 1+2: metric Invalid→Valid, cache có fingerprint, manifest theo run)
  1. Sinh / đọc dữ liệu phiếu → tiền xử lý (cache có fingerprint)
  2. Baseline và C4.5 thủ công, CNN toàn ảnh (đối chứng)
  3. Mô hình lai: CNN phát hiện dấu theo ô + cờ dấu ngoài ô (OOF) → C4.5 ra quyết định
  4. Chọn cấu hình CHỈ trên OOF Train; đánh giá TEST một lần cho cấu hình đã khóa
  5. Ghi artifact vào results/runs/<run_id>/ (không ghi đè), manifest, đăng ký lần đọc Test
Logic phân loại không đổi so với 516ce78.
Chạy trong thư mục ballot-validation-ai/:  python main.py
"""
import csv
import datetime
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

import manifest as mf
import merkle
from baselines import train_baselines
from cache import array_sha256, code_hash, fingerprint, lib_versions
from cache import load as cache_load, save as cache_save
from config import CFG, abspath
from decision import (confidences, feasible, grid_search, is_pending, oof_c45, rank,
                      screening_report)
from evaluate import invalid_to_valid_report, summarize
from feature_c45 import FEATURE_NAMES
from hashchain import HashChain, sha256_file
from monitor import check_drift, log_run
from preprocess import dataset_key, generate_synthetic_ballots, load_dataset, source_hash, split_train_val
from stray_cnn import load_inputs, oof_stray, predict_ensemble, stray_threshold
from train_c45 import train_c45
from train_cnn import oof_mark_probs, predict_cnn, predict_marks_ensemble, train_cnn

SEED = 42                       # seed cho split, OOF, CNN (đồng bộ với giá trị mặc định trong code)
LIBS = ["numpy", "scipy", "scikit-learn", "torch", "opencv-python", "opencv-python-headless", "joblib", "PyYAML"]
HYBRID_NAMES = FEATURE_NAMES + ["cnn_count_marked", "cnn_mean_prob", "cnn_max_prob"]


def hybrid_features(F, p_marks):
    """Đặc trưng cho C4.5 của mô hình lai: 8 đặc trưng thủ công + thống kê xác suất CNN từng ô."""
    thr = 0.5
    cnn_feats = np.stack([(p_marks > thr).sum(1), p_marks.mean(1), p_marks.max(1)], axis=1)
    return np.hstack([F, cnn_feats]).astype(np.float64)


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"không ghi được {type(o)}")


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1, default=_jsonable)


def _csv_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def main():
    results_dir = abspath(CFG["results_dir"])
    cache_dir = os.path.join(results_dir, "cache")
    os.makedirs(results_dir, exist_ok=True)
    run_id = mf.new_run_id()
    run_dir = mf.create_run_dir(results_dir, run_id)          # FileExistsError nếu trùng run_id
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir)
    with mf.tee_log(os.path.join(run_dir, "log.txt")):
        print(f"[run] {run_id} → {run_dir}")
        run_pipeline(run_id, run_dir, models_dir, cache_dir, results_dir)


def run_pipeline(run_id, run_dir, models_dir, cache_dir, results_dir):
    labels = CFG["labels"]
    started = _utc()
    registry_before = mf.read_test_registry(results_dir)     # ghi nhận TRƯỚC khi chạm Test
    cache_info = {}

    def cached(stage, comp, compute):
        """Cache có fingerprint: HIT → nạp; MISS → tính, lưu, ghi key và trạng thái vào manifest."""
        key = fingerprint(stage, comp)[0]
        hit = cache_load(cache_dir, stage, comp, log=print)
        arrays = hit if hit is not None else compute()
        if hit is None:
            cache_save(cache_dir, stage, comp, arrays)
        cache_info[stage] = {"key": key, "status": "HIT" if hit is not None else "MISS"}
        return arrays

    # 1. Dữ liệu
    if not os.path.exists(abspath(CFG["data"]["labels_csv"])):
        print("Chưa có dữ liệu → sinh bộ phiếu giả lập")
        generate_synthetic_ballots()
    ds_key = dataset_key()
    cache_info["dataset"] = {"key": ds_key}
    data = load_dataset()
    idx = split_train_val(data, seed=SEED)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    X, F, C, M, y, files, G = data["X"], data["F"], data["C"], data["M"], data["y"], data["files"], data["groups"]
    split_hash = array_sha256(np.concatenate([tr, va, te]))
    print(f"Phiếu: train={len(tr)} val={len(va)} test={len(te)}")

    metrics = {}

    # 2. Baseline
    for name, m in train_baselines(F[tr], y[tr]).items():
        metrics[f"baseline_{name}"] = summarize(f"Baseline {name}", y[te], m.predict(F[te]))

    # 3. C4.5 chỉ đặc trưng thủ công (đối chứng)
    print("\n----- C4.5 (đặc trưng thủ công) -----")
    c45_manual = train_c45(F[tr], y[tr], F[va], y[va], save=False)
    metrics["c45_manual"] = summarize("C4.5 (đặc trưng thủ công)", y[te], c45_manual.predict(F[te]))

    # 4. CNN toàn ảnh (đối chứng); model ghi vào thư mục run
    print("\n----- CNN toàn ảnh 224×224 (đối chứng) -----")
    cnn_full = train_cnn(X[tr], y[tr], X[va], y[va], cfg={"save_path": os.path.join(models_dir, "cnn_best.pt")})
    pred_full, _ = predict_cnn(cnn_full, X[te])
    metrics["cnn_full_image"] = summarize("CNN toàn ảnh", y[te], pred_full)

    # 5. Mô hình lai
    print("\n----- CNN phát hiện dấu theo ô (out-of-fold) + C4.5 -----")

    def compute_marks():
        p_tr, fold_models = oof_mark_probs(C[tr], M[tr], k=CFG["mark_cnn"]["folds"], seed=SEED,
                                           groups=G[tr], strat=y[tr], return_models=True)
        return {"tr": p_tr, "va": predict_marks_ensemble(fold_models, C[va]),
                "te": predict_marks_ensemble(fold_models, C[te])}

    mk = cached("mark_probs", {"dataset": ds_key, "split": split_hash, "mark_cnn": CFG["mark_cnn"],
                               "seed": SEED, "code": code_hash(["src/train_cnn.py"]),
                               "libs": lib_versions(["torch", "numpy", "scikit-learn"])}, compute_marks)
    p_marks_tr, p_marks_va, p_marks_te = mk["tr"], mk["va"], mk["te"]
    mark_acc_te = float(((p_marks_te > 0.5).astype(int) == M[te]).mean())
    print(f"CNN phát hiện dấu trên ô (test, từng ô): accuracy={mark_acc_te:.4f}")
    metrics["mark_cnn_box_accuracy"] = round(mark_acc_te, 4)

    # 5a. CNN phát hiện dấu ngoài ô (wrong_mark). recall_targets KHÔNG nằm trong khóa cache (chỉ dùng khi chọn ngưỡng)
    from sklearn.metrics import roc_auc_score
    with open(abspath(CFG["data"]["labels_csv"]), encoding="utf-8") as f:
        nstray = {r["file"]: int(r["n_strays"]) for r in csv.DictReader(f)}
    has = np.array([nstray[f] > 0 for f in files])

    def compute_stray():
        XS = load_inputs(files, abspath(CFG["data"]["raw_dir"]))
        s_tr, s_models = oof_stray(XS[tr], has[tr].astype(int), G[tr], seed=SEED)
        return {"tr": s_tr, "va": predict_ensemble(s_models, XS[va]), "te": predict_ensemble(s_models, XS[te])}

    sc = cached("stray_probs", {"dataset": ds_key, "split": split_hash,
                                "stray_cnn": {k: v for k, v in CFG["stray_cnn"].items() if k != "recall_targets"},
                                "seed": SEED, "code": code_hash(["src/stray_cnn.py"]),
                                "libs": lib_versions(["torch", "numpy", "scikit-learn"])}, compute_stray)
    s_tr, s_va, s_te = sc["tr"], sc["va"], sc["te"]

    H_tr = hybrid_features(F[tr], p_marks_tr)
    H_va = hybrid_features(F[va], p_marks_va)
    H_te = hybrid_features(F[te], p_marks_te)

    # 5b. Chọn recall_target + cấu hình C4.5 + τ + leaf_min_n: TOÀN BỘ trên OOF Train (không đọc Test)
    cands = []
    for rt in CFG["stray_cnn"]["recall_targets"]:
        tau_s_rt = stray_threshold(s_tr, has[tr], rt)
        b = grid_search(H_tr, y[tr], G[tr], p_marks_tr, extra=s_tr >= tau_s_rt)
        b.update(recall_target=rt, tau_stray=tau_s_rt)
        cands.append(b)
        o = b["oof"]
        print(f"[OOF rt={rt}] τ_stray={tau_s_rt:.4f} cờ_phiếu_sạch={(s_tr >= tau_s_rt)[~has[tr]].mean():.4f} "
              f"coverage={o['coverage']} risk={o['selective_risk']} cp_upper={o['selective_risk_cp_upper']} "
              f"recall={o['invalid_screening_recall']} | min_leaf={b['min_leaf']} cf={b['cf']} "
              f"leaf_min_n={b['leaf_min_n']} τ=({b['tau_cnn']},{b['tau_c45']}) an_toàn={b['target_met_oof']}")
    best = max(cands, key=lambda t: rank(t["oof"]))            # khóa cấu hình từ đây, trước khi đọc Test
    tau_s = best["tau_stray"]
    flag_tr, flag_te = s_tr >= tau_s, s_te >= tau_s
    print(f"\n[chọn] recall_target={best['recall_target']} min_leaf={best['min_leaf']} cf={best['cf']} "
          f"leaf_min_n={best['leaf_min_n']} τ_CNN={best['tau_cnn']} τ_C4.5={best['tau_c45']} "
          f"| an toàn trên OOF: {best['target_met_oof']} | OOF: {best['oof']}")

    # 5c. OOF theo từng phiếu của cấu hình đã khóa: cùng hàm oof_c45 với lưới; phải khớp báo cáo lưới
    proba_oof, leaf_oof = oof_c45(H_tr, y[tr], G[tr], {"min_leaf": best["min_leaf"], "cf": best["cf"]})
    cc_oof, cd_oof = confidences(p_marks_tr, proba_oof)
    pred_oof = proba_oof.argmax(1)
    pend_oof = is_pending(cc_oof, cd_oof, best["tau_cnn"], best["tau_c45"],
                          leaf_n=leaf_oof, leaf_min_n=best["leaf_min_n"], extra=flag_tr)
    if screening_report(y[tr], pred_oof, pend_oof) != best["oof"]:
        raise RuntimeError("OOF tái tính không khớp báo cáo lưới: dừng, không ghi kết quả")

    hybrid = train_c45(H_tr, y[tr], H_va, y[va], feature_names=HYBRID_NAMES, tag="hybrid_c45",
                       hp={"min_leaf": best["min_leaf"], "cf": best["cf"]}, model_dir=models_dir)
    pred_h = hybrid.predict(H_te)
    p_h = hybrid.predict_proba(H_te)
    metrics["hybrid_cnn_c45"] = summarize("MÔ HÌNH LAI (CNN ô + C4.5)", y[te], pred_h)
    metrics["stray_cnn"] = {"recall_target": best["recall_target"], "tau_stray": round(tau_s, 4),
                            "oof_auc": round(float(roc_auc_score(has[tr], s_tr)), 4),
                            "test_auc": round(float(roc_auc_score(has[te], s_te)), 4),
                            "test_recall": round(float(flag_te[has[te]].mean()), 4),
                            "test_flag_rate_clean": round(float(flag_te[~has[te]].mean()), 4)}

    # 5d. PENDING_REVIEW trên Test: áp đúng τ và leaf_min_n đã chọn ở OOF
    cc_te, cd_te = confidences(p_marks_te, p_h)
    pend_te = is_pending(cc_te, cd_te, best["tau_cnn"], best["tau_c45"],
                         leaf_n=hybrid.leaf_size(H_te), leaf_min_n=best["leaf_min_n"], extra=flag_te)
    sel = screening_report(y[te], pred_h, pend_te)
    sel.update(tau_cnn=best["tau_cnn"], tau_c45=best["tau_c45"], leaf_min_n=best["leaf_min_n"],
               min_leaf=best["min_leaf"], cf=best["cf"], tier=best["tier"],
               target_met_oof=best["target_met_oof"], oof_target_met=best["oof_target_met"])
    sel["recall_target"] = best["recall_target"]
    sel["accepted_test"] = feasible(sel)
    sel["coverage_min_status"] = "đề xuất, chưa phê duyệt"
    metrics["hybrid_selective"] = sel
    metrics["invalid_to_valid_oof"] = invalid_to_valid_report(y[tr], pred_oof, pend_oof, labels=labels)
    metrics["invalid_to_valid_test"] = invalid_to_valid_report(y[te], pred_h, pend_te, labels=labels)
    print(f"[nghiệm thu Test] {'ĐẠT' if sel['accepted_test'] else 'KHÔNG ĐẠT'} "
          f"(Recall≥{CFG['decision']['nfr10_target']}, Coverage≥{CFG['decision']['coverage_min']} [chưa phê duyệt], "
          f"Selective Risk≤{CFG['decision']['selective_risk_max']}) | {sel}")

    # 6. Artifact của run (không ghi đè: thư mục run mới, mode exclusive)
    w = mf.write_exclusive
    p_pred = os.path.join(run_dir, "predictions.csv")
    w(p_pred, _csv_text(["file", "true", "pred", "confidence", "status"],
                        [[files[i], labels[y[i]], labels[pred_h[k]], round(float(p_h[k].max()), 4),
                          "PENDING_REVIEW" if pend_te[k] else "AUTO_CLASSIFIED"] for k, i in enumerate(te)]))
    p_oof = os.path.join(run_dir, "oof_predictions.csv")
    w(p_oof, _csv_text(["file", "true", "pred", "pending", "conf_cnn", "conf_c45", "leaf_n", "stray_flag"],
                       [[files[i], labels[y[i]], labels[pred_oof[k]], bool(pend_oof[k]),
                         round(float(cc_oof[k]), 4), round(float(cd_oof[k]), 4), int(leaf_oof[k]), bool(flag_tr[k])]
                        for k, i in enumerate(tr)]))

    raw_dir = abspath(CFG["data"]["raw_dir"])
    leaves = [merkle.leaf_hash(files[i], ("PENDING_REVIEW" if pend_te[k] else labels[pred_h[k]]), p_h[k].max(),
                               sha256_file(os.path.join(raw_dir, files[i])))
              for k, i in enumerate(te)]
    batch = {"batch_id": "test-" + str(len(te)), "root": merkle.root(leaves), "count": len(leaves), "leaves": leaves}
    p_fabric = os.path.join(run_dir, "fabric_batch.json")
    w(p_fabric, _dump(batch))

    # 7. Blockchain: neo mã băm mô hình, rồi ghi từng phiếu (chuỗi lưu trong thư mục run)
    hybrid_path = os.path.join(models_dir, "hybrid_c45.joblib")
    chain = HashChain(difficulty=CFG["hashchain"]["difficulty"])
    chain.add_record({"type": "model_anchor",
                      "hybrid_c45_sha256": sha256_file(hybrid_path),
                      "manifest_sha256": sha256_file(abspath(CFG["data"]["labels_csv"]))})
    for k, i in enumerate(te):
        chain.add_ballot(ballot_id=files[i], pred=("PENDING_REVIEW" if pend_te[k] else labels[pred_h[k]]),
                         conf=p_h[k].max(), image_sha256=sha256_file(os.path.join(raw_dir, files[i])))
    chain_path = os.path.join(run_dir, "hashchain.json")
    chain.save(chain_path)
    print(f"\n[blockchain] {len(chain.blocks)} block | chuỗi hợp lệ: {chain.verify()} → {chain_path}")

    demo = HashChain.load(chain_path)                       # kiểm chứng trên bản sao trong bộ nhớ
    target = next(b.index for b in demo.blocks if "ballot_id" in b.data)
    old = demo.blocks[target].data["pred"]
    demo.tamper(target, "pred", "invalid" if old == "valid" else "valid")
    print(f"[blockchain] Demo sửa kết quả block {target} ({old} → đổi) → phát hiện sửa: {not demo.verify()}")

    # 8. MLOps (lịch sử toàn cục, chỉ thêm)
    log_path = os.path.join(results_dir, "metrics_log.json")
    log_run({"c45_manual_acc": metrics["c45_manual"]["accuracy"],
             "cnn_full_acc": metrics["cnn_full_image"]["accuracy"],
             "hybrid_acc": metrics["hybrid_cnn_c45"]["accuracy"],
             "hybrid_screening_recall": sel["invalid_screening_recall"],
             "hybrid_coverage": sel["coverage"]}, log_path)
    check_drift(log_path, key="hybrid_acc")

    p_metrics = os.path.join(run_dir, "metrics.json")
    w(p_metrics, _dump(metrics))

    # 9. Đăng ký lần đọc Test, khóa artifact, ghi manifest
    commit = mf.git_info()
    mf.register_test_read(results_dir, run_id, commit["git_commit"])
    locked_paths = [p_pred, p_oof, p_fabric, chain_path, p_metrics, hybrid_path,
                    os.path.join(models_dir, "cnn_best.pt")]
    locked_paths = [p for p in locked_paths if os.path.exists(p)]
    mf.lock_artifacts(locked_paths)
    model_sha = {name: (mf.file_sha256(os.path.join(models_dir, name))
                        if os.path.exists(os.path.join(models_dir, name)) else "không tạo trong run này")
                 for name in ["hybrid_c45.joblib", "cnn_best.pt"]}
    manifest = {
        "run_id": run_id, "started_utc": started, "finished_utc": _utc(),
        "git": commit, "config_sha256": mf.config_sha256(), "source_hash": source_hash(),
        "seed": SEED, "feature_version": CFG["feature_version"],
        "library_versions": lib_versions(LIBS),
        "cache": cache_info, "model_sha256": model_sha,
        "locked": True,
        "test_observed": {"registry_entries_before_run": len(registry_before),
                          "observed_before_run": bool(registry_before),
                          "independent": not registry_before},
        "artifacts": {os.path.relpath(p, run_dir): mf.file_sha256(p) for p in locked_paths},
        "note": "Model hash lịch sử 516ce78 không tái tạo được (model bị ghi đè trước Phase 2); không dùng làm giá trị thay thế.",
    }
    mf.write_manifest(run_dir, manifest)
    print("\n✅ PIPELINE HOÀN TẤT →", run_dir)


if __name__ == "__main__":
    main()
