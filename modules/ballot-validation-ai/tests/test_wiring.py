"""Nối dây main.py → results/runs/<run_id>/ → predict.py. Mọi bước huấn luyện và OOF đều là hàm giả,
dữ liệu giả; khóa phê duyệt chỉ được mở trong bộ nhớ của test."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [MODULE, os.path.join(MODULE, "src")]


class TestPipelineWiring(unittest.TestCase):
    def test_run_artifacts_feed_predict(self):
        import cv2

        import main
        import predict
        from decision import confidences, is_pending, screening_report
        from stray_cnn import StrayNet
        from train_cnn import MarkCNN

        d = tempfile.mkdtemp()
        n, raw = 40, os.path.join(d, "raw")
        os.makedirs(raw)
        y = np.arange(n) % 5
        files = np.array([f"{i}.png" for i in range(n)])
        for name in files:
            cv2.imwrite(os.path.join(raw, name), np.full((850, 1200), 255, np.uint8))
        with open(os.path.join(d, "all.csv"), "w") as f:
            f.write("file,n_strays\n" + "".join(f"{name},{i % 2}\n" for i, name in enumerate(files)))
        data = {"X": np.zeros((n, 8, 8), np.float32), "F": np.random.default_rng(0).random((n, 8)),
                "C": np.zeros((n, 5, 48, 48), np.float32), "M": np.zeros((n, 5), int), "y": y, "files": files,
                "groups": np.array([f"g{i // 5}" for i in range(n)]),
                "split": np.array(["train"] * 30 + ["test"] * 10)}
        cfg = {"results_dir": os.path.join(d, "results"),
               "data": dict(main.CFG["data"], raw_dir=raw, labels_csv=os.path.join(d, "all.csv")),
               "decision": dict(main.CFG["decision"], acceptance_policy_approved=True,
                                oof_selection_authorized=True, nested_oof_selection_implemented=True,
                                invalid_labels=[1, 2, 3, 4], nfr10_target=0.9, coverage_min=0.1)}
        calls = []

        def fake_marks(C, M, **kwargs):
            calls.append("mark")
            return np.zeros(M.shape), [MarkCNN()]

        def fake_stray(X, labels, groups, **kwargs):
            calls.append("stray")
            return np.linspace(0.1, 1, len(X)), [StrayNet()]

        def fake_c45(H, y_, groups, hp):
            return np.eye(5)[y_], np.full(len(y_), 100)

        def fake_grid(H, y_, groups, p_marks, extra=None):
            proba, leaf = fake_c45(H, y_, groups, None)
            cc, cd = confidences(p_marks, proba)
            pending = is_pending(cc, cd, 0.9, 0.5, leaf_n=leaf, extra=extra)
            return {"min_leaf": 3, "cf": 0.25, "leaf_min_n": 0, "tau_cnn": 0.9, "tau_c45": 0.5,
                    "oof": screening_report(y_, proba.argmax(1), pending, invalid_labels=[1, 2, 3, 4])}

        stubs = {"load_dataset": lambda: data, "dataset_key": lambda: "k", "train_cnn": lambda *a, **k: None,
                 "predict_cnn": lambda m, X: (np.zeros(len(X), int), None), "oof_mark_probs": fake_marks,
                 "predict_marks_ensemble": lambda m, C: np.zeros((len(C), 5)),
                 "load_inputs": lambda f, r: np.zeros((len(f), 8, 8), np.float32), "oof_stray": fake_stray,
                 "predict_ensemble": lambda m, X: np.zeros(len(X)), "oof_c45": fake_c45, "grid_search": fake_grid}
        with mock.patch.dict(main.CFG, cfg):
            with mock.patch.multiple(main, **stubs):
                main.main()
                main.main()
            self.assertEqual(calls, ["mark", "stray"])               # lần 2 dùng cache
            runs = os.listdir(os.path.join(d, "results", "runs"))
            self.assertEqual(len(runs), 2)                           # không ghi đè run cũ
            c45, models, th = predict.load_models()
            self.assertGreater(th["tau_stray"], 0)
            result = predict.predict_one(os.path.join(raw, "0.png"), c45, models, th, {})
            self.assertIn("status", result)
            latest = predict.mf.latest_run_dir(cfg["results_dir"])
        with open(os.path.join(latest, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertFalse(manifest["test_observed"]["independent"])
        self.assertIn("models/stray_cnn.pt", manifest["artifacts"])


if __name__ == "__main__":
    unittest.main()
