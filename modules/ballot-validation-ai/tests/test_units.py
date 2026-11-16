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


if __name__ == "__main__":
    unittest.main()


class TestDecision(unittest.TestCase):
    def test_screening_counts_pending_as_detected(self):
        from decision import screening_report
        y    = np.array([1, 2, 3, 0, 0])      # 3 không hợp lệ, 2 hợp lệ
        pred = np.array([0, 2, 0, 0, 0])      # thứ 1: dự đoán VALID; thứ 3: dự đoán VALID
        pend = np.array([False, False, True, False, False])
        r = screening_report(y, pred, pend)
        self.assertAlmostEqual(r["invalid_screening_recall"], 2 / 3, places=4)   # idx 1 (auto đúng) + idx 2 (chờ)
        self.assertAlmostEqual(r["automatic_invalid_recall"], 1 / 3, places=4)
        self.assertEqual(r["n_auto"], 4)
        self.assertAlmostEqual(r["referral_rate"], 1 / 5, places=4)
        self.assertAlmostEqual(r["selective_risk"], 1 / 4, places=4)            # idx 0 auto nhưng sai

    def test_or_rule(self):
        from decision import is_pending
        p = is_pending(np.array([0.99, 0.6]), np.array([0.99, 0.99]), tau_cnn=0.9, tau_c45=0.9)
        self.assertEqual(p.tolist(), [False, True])
        p = is_pending(np.array([0.99]), np.array([0.5]), tau_cnn=0.9, tau_c45=0.9)
        self.assertEqual(p.tolist(), [True])

    def test_select_tau_meets_target_when_possible(self):
        from decision import select_tau
        rng = np.random.default_rng(0)
        y = rng.integers(0, 5, 400)
        pred = y.copy()
        conf_c45 = np.where(pred == y, 0.95, 0.3)       # sai → độ tin cậy thấp
        conf_cnn = np.full(400, 0.99)
        t = select_tau(y, pred, conf_cnn, conf_c45, step=0.01)
        self.assertTrue(t["oof_target_met"])
        self.assertGreaterEqual(t["oof"]["invalid_screening_recall"], 0.99)

    def test_rank_order_recall_then_risk_then_coverage(self):
        from decision import rank
        hi_recall_low_cov = {"invalid_screening_recall": 0.995, "selective_risk": 0.02, "coverage": 0.5}
        hi_recall_high_cov = {"invalid_screening_recall": 0.993, "selective_risk": 0.01, "coverage": 0.9}
        floor_only = {"invalid_screening_recall": 0.985, "selective_risk": 0.0, "coverage": 1.0}
        below_floor = {"invalid_screening_recall": 0.97, "selective_risk": 0.0, "coverage": 1.0}
        self.assertGreater(rank(hi_recall_high_cov), rank(hi_recall_low_cov))   # risk thấp hơn thắng
        self.assertGreater(rank(hi_recall_low_cov), rank(floor_only))           # đạt 0.99 thắng 0.98
        self.assertGreater(rank(floor_only), rank(below_floor))                 # trên sàn thắng dưới sàn

    def test_no_auto_gives_risk_one(self):
        from decision import screening_report
        r = screening_report(np.array([1, 2]), np.array([1, 2]), np.array([True, True]))
        self.assertEqual(r["selective_risk"], 1.0)
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
