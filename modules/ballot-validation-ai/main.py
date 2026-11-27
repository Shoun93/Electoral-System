"""
PIPELINE AI — Phase 1 safeguards.
Full model/OOF execution is blocked until PM policy approval and separate OOF authorization.
Observed Test is descriptive only, never an acceptance/selection source.
Artifact của mỗi lần chạy nằm trong results/runs/<run_id>/ (không ghi đè), kèm manifest.json.
"""
import csv, datetime, hashlib, io, json, os, shutil, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np
import torch
from config import CFG, abspath
from preprocess import dataset_key, load_dataset, split_train_val
from cache import compute_cache_key, load_mark_prob_cache, save_mark_prob_cache
from baselines import train_baselines
from train_c45 import train_c45
from train_cnn import train_cnn, predict_cnn, oof_mark_probs, predict_marks_ensemble
from stray_cnn import load_inputs, oof_stray, predict_ensemble, stray_threshold
from evaluate import invalid_to_valid_report, summarize
from decision import (PolicyNotApprovedError, confidences, feasible, grid_search, is_pending, oof_c45,
                      rank, screening_report)
from hashchain import HashChain, sha256_file
import manifest as mf
import merkle
from monitor import check_drift, log_run
from feature_c45 import FEATURE_NAMES

HYBRID_NAMES = FEATURE_NAMES + ["cnn_count_marked", "cnn_mean_prob", "cnn_max_prob"]
LIBS = ["numpy", "scipy", "scikit-learn", "torch", "opencv-python", "opencv-python-headless", "joblib", "PyYAML"]

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

def _cached_ensemble(cache_dir, name, arrays, config, seed, folds, expected_rows, compute):
    """Xác suất + checkpoint các fold; chỉ dùng lại khi khóa dữ liệu, cấu hình, mã và checkpoint khớp."""
    checkpoint = os.path.join(cache_dir, f"{name}_cnn.pt")
    probs_path = os.path.join(cache_dir, f"{name}_probs.npz")
    def key():
        return compute_cache_key(arrays, config, _file_sha256(checkpoint), seed=seed, folds=folds)
    if os.path.isfile(checkpoint):
        cached = load_mark_prob_cache(probs_path, key(), expected_rows)
        if cached is not None:
            print(f"[cache] hit: {name}")
            return cached, checkpoint
    probs, models = compute()
    os.makedirs(cache_dir, exist_ok=True)
    torch.save([model.state_dict() for model in models], checkpoint)
    save_mark_prob_cache(probs_path, key(), probs["tr"], probs["va"], probs["te"])
    return probs, checkpoint

def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def _csv_text(header, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()

def _dump(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: o.item())

def main():
    _require_pipeline_authorization()
    results_dir = abspath(CFG["results_dir"])
    if not os.path.isfile(abspath(CFG["data"]["labels_csv"])):
        raise FileNotFoundError(f"Không tìm thấy manifest dữ liệu: {abspath(CFG['data']['labels_csv'])}. "
                                "Phase 1 không tự động sinh/thay đổi dataset.")
    os.makedirs(results_dir, exist_ok=True)
    run_id = mf.new_run_id()
    run_dir = mf.create_run_dir(results_dir, run_id)
    with mf.tee_log(os.path.join(run_dir, "log.txt")):
        print(f"[run] {run_id} → {run_dir}")
        _run(run_id, run_dir, results_dir)

def _run(run_id, run_dir, results_dir):
    labels = CFG["labels"]
    labels_path = abspath(CFG["data"]["labels_csv"])
    raw_dir = abspath(CFG["data"]["raw_dir"])
    models_dir = os.path.join(run_dir, "models")
    cache_dir = os.path.join(results_dir, "cache")
    os.makedirs(models_dir)
    started = _utc()
    registry_before = mf.read_test_registry(results_dir)

    source_key = dataset_key()
    data = load_dataset()
    seed = int(CFG.get("cache", {}).get("seed", 42))
    idx = split_train_val(data, seed=seed)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    X, F, C, M, y = data["X"], data["F"], data["C"], data["M"], data["y"]
    files, groups = data["files"], data["groups"]
    print(f"train={len(tr)} val={len(va)} observed_test={len(te)} | leakage={idx['meta']['leakage_check']}")
    metrics = {"validation_split": idx["meta"], "test_partition_role": "OBSERVED_TEST_DESCRIPTIVE_ONLY"}
    expected_rows = {"tr": len(tr), "va": len(va), "te": len(te)}

    for name, model in train_baselines(F[tr], y[tr]).items():
        metrics[f"baseline_{name}"] = summarize(f"Baseline {name}", y[te], model.predict(F[te]))
    c45_manual = train_c45(F[tr], y[tr], F[va], y[va], save=False)
    metrics["c45_manual"] = summarize("C4.5 (đặc trưng thủ công)", y[te], c45_manual.predict(F[te]))
    cnn_full = train_cnn(X[tr], y[tr], X[va], y[va], cfg={"save_path": os.path.join(models_dir, "cnn_best.pt")})
    pred_full, _ = predict_cnn(cnn_full, X[te])
    metrics["cnn_full_image"] = summarize("CNN toàn ảnh", y[te], pred_full)

    # CNN theo ô: ensemble các mô hình fold, để val/test cùng phân phối với OOF của train.
    folds = int(CFG["mark_cnn"].get("folds", 5))
    def compute_marks():
        p_tr, models = oof_mark_probs(C[tr], M[tr], groups=groups[tr], strat=y[tr], k=folds, seed=seed,
                                      return_models=True)
        return {"tr": p_tr, "va": predict_marks_ensemble(models, C[va]),
                "te": predict_marks_ensemble(models, C[te])}, models
    mark_inputs = {
        "C_train": C[tr], "M_train": M[tr], "groups_train": groups[tr], "labels_train": y[tr],
        "C_val": C[va], "C_test": C[te], "files": files, "split": data["split"],
        # Đổi mã huấn luyện thì cache cũng hết hiệu lực.
        "train_code": np.asarray(_file_sha256(abspath("src/train_cnn.py"))),
    }
    marks, mark_ckpt = _cached_ensemble(cache_dir, "mark", mark_inputs, dict(CFG["mark_cnn"]), seed, folds,
                                        expected_rows, compute_marks)
    p_marks_tr, p_marks_va, p_marks_te = marks["tr"], marks["va"], marks["te"]
    shutil.copyfile(mark_ckpt, os.path.join(models_dir, "mark_cnn.pt"))

    # CNN phát hiện dấu ngoài ô. n_strays chỉ làm nhãn huấn luyện, không làm đặc trưng.
    with open(labels_path, encoding="utf-8") as stream:
        n_strays = {row["file"]: int(row["n_strays"]) for row in csv.DictReader(stream)}
    has_stray = np.array([n_strays[str(name)] > 0 for name in files])
    stray_folds = int(CFG["stray_cnn"]["folds"])
    def compute_stray():
        XS = load_inputs(files, raw_dir)
        s_tr_, models = oof_stray(XS[tr], has_stray[tr].astype(int), groups[tr], seed=seed, k=stray_folds)
        return {"tr": s_tr_[:, None], "va": predict_ensemble(models, XS[va])[:, None],
                "te": predict_ensemble(models, XS[te])[:, None]}, models
    stray_inputs = {"source": np.asarray(source_key), "split": data["split"], "train": tr, "val": va, "test": te,
                    "has_stray": has_stray, "train_code": np.asarray(_file_sha256(abspath("src/stray_cnn.py")))}
    stray_cfg = {"epochs": CFG["stray_cnn"]["epochs"]}
    stray, stray_ckpt = _cached_ensemble(cache_dir, "stray", stray_inputs, stray_cfg, seed, stray_folds,
                                         expected_rows, compute_stray)
    s_tr, s_te = stray["tr"][:, 0], stray["te"][:, 0]
    shutil.copyfile(stray_ckpt, os.path.join(models_dir, "stray_cnn.pt"))

    metrics["mark_cnn_box_accuracy_observed_test"] = round(float(((p_marks_te > 0.5).astype(int) == M[te]).mean()), 4)
    H_tr = hybrid_features(F[tr], p_marks_tr)
    H_va = hybrid_features(F[va], p_marks_va)
    H_te = hybrid_features(F[te], p_marks_te)

    candidates = []
    for recall_target in CFG["stray_cnn"]["recall_targets"]:
        tau_stray = stray_threshold(s_tr, has_stray[tr], recall_target)
        candidate = grid_search(H_tr, y[tr], groups[tr], p_marks_tr, extra=s_tr >= tau_stray)
        candidate.update(recall_target=recall_target, tau_stray=tau_stray)
        candidates.append(candidate)
    best = max(candidates, key=lambda c: rank(c["oof"]))   # khóa cấu hình trước khi đọc Test
    flag_tr, flag_te = s_tr >= best["tau_stray"], s_te >= best["tau_stray"]
    invalid_labels = CFG["decision"]["invalid_labels"]

    proba_oof, leaf_oof = oof_c45(H_tr, y[tr], groups[tr], {"min_leaf": best["min_leaf"], "cf": best["cf"]})
    cc_oof, cd_oof = confidences(p_marks_tr, proba_oof)
    pred_oof = proba_oof.argmax(1)
    pend_oof = is_pending(cc_oof, cd_oof, best["tau_cnn"], best["tau_c45"],
                          leaf_n=leaf_oof, leaf_min_n=best["leaf_min_n"], extra=flag_tr)
    if screening_report(y[tr], pred_oof, pend_oof, invalid_labels=invalid_labels) != best["oof"]:
        raise RuntimeError("OOF tái tính không khớp báo cáo lưới: dừng, không ghi kết quả")

    hybrid = train_c45(H_tr, y[tr], H_va, y[va], feature_names=HYBRID_NAMES, tag="hybrid_c45",
                       hp={"min_leaf": best["min_leaf"], "cf": best["cf"]}, model_dir=models_dir)
    pred_h, p_h = hybrid.predict(H_te), hybrid.predict_proba(H_te)
    metrics["hybrid_cnn_c45_observed_test"] = summarize("MÔ HÌNH LAI — Observed Test", y[te], pred_h)
    cc_te, cd_te = confidences(p_marks_te, p_h)
    pend_te = is_pending(cc_te, cd_te, best["tau_cnn"], best["tau_c45"],
                         leaf_n=hybrid.leaf_size(H_te), leaf_min_n=best["leaf_min_n"], extra=flag_te)
    sel = screening_report(y[te], pred_h, pend_te, invalid_labels=invalid_labels)
    sel.update(tau_cnn=best["tau_cnn"], tau_c45=best["tau_c45"], leaf_min_n=best["leaf_min_n"],
               tau_stray=best["tau_stray"], recall_target=best["recall_target"],
               min_leaf=best["min_leaf"], cf=best["cf"], threshold_source="OOF_TRAIN_ONLY",
               evaluation_partition="OBSERVED_TEST_DESCRIPTIVE_ONLY", acceptance_status="NOT_AN_ACCEPTANCE_RESULT")
    sel["acceptance_feasible"] = feasible(sel)
    metrics["hybrid_selective_observed_test"] = sel
    metrics["invalid_to_valid_oof"] = invalid_to_valid_report(y[tr], pred_oof, pend_oof, invalid_labels)
    metrics["invalid_to_valid_observed_test"] = invalid_to_valid_report(y[te], pred_h, pend_te, invalid_labels)
    print(f"[Observed Test — descriptive only] {sel}")

    status_te = ["PENDING_REVIEW" if pend_te[k] else "AUTO_CLASSIFIED" for k in range(len(te))]
    mf.write_exclusive(os.path.join(run_dir, "predictions.csv"), _csv_text(
        ["file", "true", "pred", "confidence", "status", "partition_role"],
        [[files[i], labels[y[i]], labels[pred_h[k]], round(float(p_h[k].max()), 4), status_te[k], "OBSERVED_TEST"]
         for k, i in enumerate(te)]))
    mf.write_exclusive(os.path.join(run_dir, "oof_predictions.csv"), _csv_text(
        ["file", "true", "pred", "pending", "conf_cnn", "conf_c45", "leaf_n", "stray_flag"],
        [[files[i], labels[y[i]], labels[pred_oof[k]], bool(pend_oof[k]), round(float(cc_oof[k]), 4),
          round(float(cd_oof[k]), 4), int(leaf_oof[k]), bool(flag_tr[k])] for k, i in enumerate(tr)]))

    chain = HashChain(difficulty=CFG["hashchain"]["difficulty"])
    chain.add_record({"type": "model_anchor",
                      **{f"{n}_sha256": sha256_file(os.path.join(models_dir, f))
                         for n, f in (("hybrid_c45", "hybrid_c45.joblib"), ("mark_cnn", "mark_cnn.pt"),
                                      ("stray_cnn", "stray_cnn.pt"))},
                      "manifest_sha256": sha256_file(labels_path)})
    image_sha = [sha256_file(os.path.join(raw_dir, str(files[i]))) for i in te]
    pred_label = ["PENDING_REVIEW" if pend_te[k] else labels[pred_h[k]] for k in range(len(te))]
    for k, index in enumerate(te):
        chain.add_ballot(ballot_id=str(files[index]), pred=pred_label[k], conf=p_h[k].max(), image_sha256=image_sha[k])
    leaves = [merkle.leaf_hash(files[index], pred_label[k], p_h[k].max(), image_sha[k])
              for k, index in enumerate(te)]
    root = merkle.root(leaves)   # batch_id kèm gốc: chaincode ghi một lần, hai lô khác nhau không được trùng khóa
    batch = {"batch_id": f"observed-test-{len(te)}-{root[:16]}", "root": root, "count": len(leaves), "leaves": leaves}
    mf.write_exclusive(os.path.join(run_dir, "fabric_batch.json"), _dump(batch))
    chain_path = os.path.join(run_dir, "hashchain.json")
    chain.save(chain_path)
    print(f"[blockchain] {len(chain.blocks)} block | valid={chain.verify()} → {chain_path}")

    log_path = os.path.join(results_dir, "metrics_log.json")
    log_run({"run_id": run_id, "c45_manual_acc": metrics["c45_manual"]["accuracy"],
             "cnn_full_acc": metrics["cnn_full_image"]["accuracy"],
             "hybrid_acc_observed_test": metrics["hybrid_cnn_c45_observed_test"]["accuracy"],
             "coverage_observed_test": sel["coverage"]}, log_path)
    check_drift(log_path, key="hybrid_acc_observed_test")
    mf.write_exclusive(os.path.join(run_dir, "metrics.json"), _dump(metrics))

    git = mf.git_info()
    mf.register_test_read(results_dir, run_id, git["git_commit"])
    names = ["predictions.csv", "oof_predictions.csv", "fabric_batch.json", "hashchain.json", "metrics.json",
             "models/hybrid_c45.joblib", "models/mark_cnn.pt", "models/stray_cnn.pt", "models/cnn_best.pt"]
    paths = {name: os.path.join(run_dir, name) for name in names}
    paths = {name: path for name, path in paths.items() if os.path.exists(path)}
    mf.lock_artifacts(paths.values())
    mf.write_manifest(run_dir, {
        "run_id": run_id, "started_utc": started, "finished_utc": _utc(), "git": git,
        "config_sha256": mf.config_sha256(), "source_key": source_key, "seed": seed,
        "library_versions": mf.lib_versions(LIBS), "locked": True,
        "test_observed": {"registry_entries_before_run": len(registry_before), "independent": False},
        "artifacts": {name: mf.file_sha256(path) for name, path in paths.items()},
    })
    mf.write_latest(results_dir, run_id)
    print("PIPELINE HOÀN TẤT. Observed Test chỉ có giá trị mô tả; không phải Independent Test.")

if __name__ == "__main__":
    main()
