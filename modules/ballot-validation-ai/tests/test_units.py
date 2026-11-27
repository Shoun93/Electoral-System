"""Kiểm thử đơn vị (không cần dữ liệu lớn). Chạy:  python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from c45 import C45Classifier, pessimistic_errors            # noqa: E402
from hashchain import HashChain                                # noqa: E402
from monitor import psi                                        # noqa: E402
from merkle import root as merkle_root                         # noqa: E402
from preprocess import check_group_disjoint, split_train_val  # noqa: E402


class TestC45(unittest.TestCase):
    def test_learns_threshold_rule(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 10, size=(300, 2))
        y = (X[:, 0] > 6).astype(int)                       # luật: x0 > 6 → lớp 1
        clf = C45Classifier(max_depth=3, min_leaf=5, cf=0.25).fit(X, y, ["a", "b"], ["no", "yes"])
        self.assertGreaterEqual((clf.predict(X) == y).mean(), 0.97)
        rules = clf.export_rules()
        self.assertTrue(any("a" in cond for cond, *_ in rules))

    def test_predict_proba_rows_sum_to_one(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 3))
        y = rng.integers(0, 3, 200)
        clf = C45Classifier(max_depth=4, min_leaf=5).fit(X, y)
        np.testing.assert_allclose(clf.predict_proba(X).sum(1), 1.0, atol=1e-9)

    def test_pessimistic_error_is_upper_bound(self):
        for N, E in [(10, 0), (50, 5), (200, 40)]:
            self.assertGreaterEqual(pessimistic_errors(N, E, 0.25), E)


class TestHashChain(unittest.TestCase):
    def test_valid_chain_verifies(self):
        c = HashChain(difficulty=2)
        for i in range(5):
            c.add_ballot(f"b{i}", "valid", 0.9, image_sha256="ab" * 32)
        self.assertTrue(c.verify())

    def test_tamper_is_detected(self):
        c = HashChain(difficulty=2)
        for i in range(5):
            c.add_ballot(f"b{i}", "valid", 0.9)
        c.tamper(3, "pred", "invalid")
        ok, bad = c.validate()
        self.assertFalse(ok)
        self.assertEqual(bad, 3)

    def test_save_load_roundtrip(self):
        c = HashChain(difficulty=2)
        c.add_ballot("x", "blank", 0.5)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "chain.json")
            c.save(p)
            self.assertTrue(HashChain.load(p).verify())


class TestPSI(unittest.TestCase):
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=5000)
        self.assertLess(abs(psi(a, a)), 1e-3)

    def test_shifted_distribution_large(self):
        rng = np.random.default_rng(3)
        self.assertGreater(psi(rng.normal(0, 1, 5000), rng.normal(2, 1, 5000)), 0.25)

    def test_constant_feature_does_not_crash(self):
        self.assertEqual(psi(np.zeros(100), np.zeros(50)), 0.0)


class TestMerkle(unittest.TestCase):
    def test_vector_matches_go(self):
        # Cùng vector với chaincode/ballot/merkle/merkle_test.go (đã chạy được trên Go)
        import hashlib
        leaves = [hashlib.sha256(s.encode()).hexdigest() for s in ["a", "b", "c"]]
        self.assertEqual(merkle_root(leaves),
                         "d31a37ef6ac14a2db1470c4316beb5592e6afd4465022339adafda76a18ffabe")

    def test_single_leaf_is_root(self):
        self.assertEqual(merkle_root(["ab" * 32]), "ab" * 32)


def _fixture(n_groups=60, n_test_groups=10):
    """Fixture tổng hợp: mỗi group có đúng 5 mẫu (1/lớp), không dùng dữ liệu thật."""
    groups, y, split = [], [], []
    for g in range(n_groups):
        for label in range(5):
            groups.append(f"g{g:04d}")
            y.append(label)
            split.append("test" if g >= n_groups - n_test_groups else "train")
    return {"groups": np.array(groups), "y": np.array(y), "split": np.array(split)}


class TestGroupSplit(unittest.TestCase):
    def setUp(self):
        self.d = _fixture()

    def test_no_group_overlap_and_mapping(self):
        out = split_train_val(self.d, seed=42)
        tr, va, te = out["train"], out["val"], out["test"]
        self.assertEqual(len(set(self.d["groups"][tr]) & set(self.d["groups"][va])), 0)
        self.assertTrue(set(tr).isdisjoint(va))
        self.assertTrue(all(self.d["split"][i] == "train" for i in tr))
        self.assertTrue(all(self.d["split"][i] == "train" for i in va))
        self.assertTrue(all(self.d["split"][i] == "test" for i in te))
        self.assertEqual(len(tr) + len(va), int((self.d["split"] == "train").sum()))

    def test_same_seed_same_split(self):
        a, b = split_train_val(self.d, seed=42), split_train_val(self.d, seed=42)
        for k in ("train", "val", "test"):
            np.testing.assert_array_equal(a[k], b[k])

    def test_different_seed_can_differ(self):
        base = split_train_val(self.d, seed=1)["val"]
        self.assertTrue(any(not np.array_equal(base, split_train_val(self.d, seed=s)["val"]) for s in range(2, 8)))

    def test_meta_records_method_and_purpose(self):
        m = split_train_val(self.d, seed=42)["meta"]
        self.assertEqual(m["method"], "GroupShuffleSplit")
        self.assertEqual(m["group_overlap_train_val"], 0)
        self.assertEqual(m["leakage_check"], "PASS")
        self.assertIn("model-selection validation", m["validation_purpose"])
        self.assertIn("không phải Independent Test", m["validation_purpose"])

    def test_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            check_group_disjoint(["g0001", "g0002"], ["g0002"])

    def test_empty_group_id_is_rejected(self):
        d = _fixture()
        d["groups"] = d["groups"].copy()
        d["groups"][0] = " "
        with self.assertRaises(ValueError):
            split_train_val(d, seed=42)



class TestDecision(unittest.TestCase):
    def test_pending_is_referral_not_automatic_detection(self):
        from decision import screening_report
        y = np.array([1, 2, 3, 0, 0])
        pred = np.array([0, 2, 0, 0, 0])
        pend = np.array([False, False, True, False, False])
        r = screening_report(y, pred, pend, invalid_labels=[1, 2, 3])
        self.assertAlmostEqual(r["automatic_invalid_recall"], 1 / 3, places=4)
        self.assertAlmostEqual(r["invalid_referral_rate"], 1 / 3, places=4)
        self.assertAlmostEqual(r["invalid_interception_rate"], 2 / 3, places=4)
        self.assertFalse(r["pending_counted_as_automatic_detection"])
        self.assertEqual(r["n_auto"], 4)
        self.assertAlmostEqual(r["referral_rate"], 1 / 5, places=4)
        self.assertAlmostEqual(r["selective_risk"], 1 / 4, places=4)

    def test_or_rule(self):
        from decision import is_pending
        p = is_pending(np.array([0.99, 0.6]), np.array([0.99, 0.99]), tau_cnn=0.9, tau_c45=0.9)
        self.assertEqual(p.tolist(), [False, True])
        p = is_pending(np.array([0.99]), np.array([0.5]), tau_cnn=0.9, tau_c45=0.9)
        self.assertEqual(p.tolist(), [True])

    def test_select_tau_is_blocked_until_policy_and_oof_approval(self):
        from decision import PolicyNotApprovedError, select_tau
        with self.assertRaises(PolicyNotApprovedError):
            select_tau(np.array([0, 1]), np.array([0, 1]),
                       np.array([0.99, 0.99]), np.array([0.99, 0.99]), step=0.01)

    def test_acceptance_is_blocked_without_pm_policy(self):
        from decision import feasible
        self.assertFalse(feasible({"coverage": 1.0, "selective_risk": 0.0}))

    def test_no_auto_gives_undefined_selective_risk(self):
        from decision import screening_report
        r = screening_report(np.array([1, 2]), np.array([1, 2]), np.array([True, True]))
        self.assertIsNone(r["selective_risk"])
        self.assertEqual(r["coverage"], 0.0)


class TestLeafRule(unittest.TestCase):
    def test_small_leaf_forces_pending(self):
        from decision import is_pending
        p = is_pending(np.array([0.99, 0.99]), np.array([0.99, 0.99]), 0.5, 0.5,
                       leaf_n=np.array([5, 60]), leaf_min_n=10)
        self.assertEqual(p.tolist(), [True, False])

    def test_leaf_size_counts_training_samples(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 10, size=(200, 1))
        y = (X[:, 0] > 5).astype(int)
        clf = C45Classifier(max_depth=2, min_leaf=5, cf=0.25).fit(X, y, ["a"], ["no", "yes"])
        n = clf.leaf_size(X)
        self.assertTrue((n >= 5).all())
        self.assertLessEqual(n.max(), 200)


class TestRegressions(unittest.TestCase):
    def test_edge_density_no_uint8_overflow(self):
        from feature_c45 import extract_features
        b = np.zeros((20, 20), np.uint8)
        b[:, :10] = 1                                        # một biên dọc: 20 chỗ chuyển 1→0
        box = [0, 0, 20, 20]
        F = extract_features(b, {"boxes": [box] * 5, "left_band": box, "extra_zone": box})
        self.assertAlmostEqual(F[4], 20 / 400)

    def test_chain_rejects_bad_index_difficulty_and_empty(self):
        c = HashChain(difficulty=2)
        c.add_ballot("x", "valid", 0.9)
        c.blocks[1].index = 7
        c.blocks[1].hash = c.blocks[1].mine()
        self.assertEqual(c.validate(), (False, 1))
        c = HashChain(difficulty=2)
        b = c.add_ballot("x", "valid", 0.9)
        b.difficulty = 0
        self.assertEqual(c.validate(), (False, 1))
        c.blocks = []
        self.assertFalse(c.verify())

    def test_monitor_handles_missing_key_and_empty_input(self):
        from monitor import check_drift, log_run
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "log.json")
            log_run({"old": 1.0}, p)
            log_run({"acc": 0.9}, p)
            self.assertFalse(check_drift(p, key="acc"))
        with self.assertRaises(ValueError):
            psi(np.arange(10), [])

    def test_dataset_cache_invalidates_when_sources_change(self):
        import json
        from unittest import mock

        import cv2
        import preprocess
        with tempfile.TemporaryDirectory() as d:
            data = dict(preprocess.CFG["data"], raw_dir=d, labels_csv=os.path.join(d, "all.csv"),
                        boxes_json=os.path.join(d, "boxes.json"), processed_npz=os.path.join(d, "d.npz"))
            box = [10, 10, 40, 40]
            with open(data["boxes_json"], "w") as f:
                json.dump({"a.png": {"boxes": [box] * 5, "left_band": box, "extra_zone": box}}, f)
            with open(data["labels_csv"], "w") as f:
                f.write("file,label,split,group_id,o_co_danh_dau\na.png,blank,train,g0,\n")
            img = os.path.join(d, "a.png")
            cv2.imwrite(img, np.full((60, 60), 255, np.uint8))
            with mock.patch.dict(preprocess.CFG, {"data": data}):
                first = preprocess.load_dataset()
                cv2.imwrite(img, np.eye(60, dtype=np.uint8) * 255)
                second = preprocess.load_dataset()
                self.assertFalse(np.array_equal(first["X"], second["X"]))


if __name__ == "__main__":
    unittest.main()
