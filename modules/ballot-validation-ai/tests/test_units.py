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


class TestFeatureEdges(unittest.TestCase):
    def test_edge_density_no_uint8_wraparound(self):
        import feature_c45
        b = np.zeros((4, 4), dtype=np.uint8)
        b[1:3, 1:3] = 1                    # ô 2×2 ở giữa: 8 chuyển 0↔1 theo hàng/cột
        layout = {"boxes": [(0, 0, 2, 2)], "left_band": (0, 0, 2, 2), "extra_zone": (0, 0, 2, 2)}
        f = feature_c45.extract_features(b, layout)
        self.assertAlmostEqual(f[4], 8 / 16, places=6)


class TestDecision(unittest.TestCase):
    def test_blank_is_proposed_invalid(self):
        import decision
        out = decision.propose_legal([0, 1, 2, 3, 4], [False] * 5)
        self.assertEqual(list(out), [decision.PROPOSED_VALID] + [decision.PROPOSED_INVALID] * 4)

    def test_pending_overrides_prediction(self):
        import decision
        out = decision.propose_legal([0, 1], [True, False])
        self.assertEqual(out[0], decision.PENDING)


class TestClopperPearson(unittest.TestCase):
    def test_zero_errors_thresholds(self):
        from evaluate import clopper_pearson_upper as cp
        self.assertLessEqual(cp(0, 299), 0.01)     # BA: 299 mẫu đủ cho 1%
        self.assertGreater(cp(0, 298), 0.00999)
        self.assertLessEqual(cp(0, 149), 0.02)     # BA: 149 mẫu đủ cho 2%


class TestStrayFlag(unittest.TestCase):
    def test_extra_flag_forces_pending(self):
        import decision
        p = decision.is_pending([0.99, 0.99], [0.99, 0.99], 0.5, 0.5, extra=[False, True])
        self.assertEqual(list(p), [False, True])

    def test_stray_threshold_keeps_recall(self):
        import numpy as np
        from stray_cnn import stray_threshold
        p = np.linspace(0, 1, 200)
        has = np.arange(200) >= 100                 # 100 dương, p tăng dần
        t = stray_threshold(p, has, recall=0.99)
        self.assertGreaterEqual((p[has] >= t).mean(), 0.99)

    def test_maxpool_keeps_thin_stroke(self):
        import numpy as np
        from stray_cnn import maxpool_input
        b = np.zeros((850, 1200), np.uint8)
        b[400, 600] = 1                             # nét 1 px
        out = maxpool_input(b)
        self.assertEqual(out.shape, (213, 300))
        self.assertEqual(out.max(), 1.0)

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

    def test_rank_order_safe_then_coverage_then_risk(self):
        """Quy tắc đã duyệt (19/11): an toàn (CP ≤ 0.02) → Coverage cao → Risk thấp khi Coverage gần nhau."""
        from decision import rank, feasible
        a = {"invalid_screening_recall": 0.995, "selective_risk": 0.01, "selective_risk_cp_upper": 0.015, "coverage": 0.85}
        b = {"invalid_screening_recall": 0.990, "selective_risk": 0.005, "selective_risk_cp_upper": 0.012, "coverage": 0.95}
        c = {"invalid_screening_recall": 0.985, "selective_risk": 0.0, "selective_risk_cp_upper": 0.010, "coverage": 1.0}
        bad = {"invalid_screening_recall": 0.999, "selective_risk": 0.0, "selective_risk_cp_upper": 0.0, "coverage": 0.2}  # coverage < 0.34
        unsafe = {"invalid_screening_recall": 0.99, "selective_risk": 0.0, "selective_risk_cp_upper": 0.05, "coverage": 0.99}
        for r in (a, b, c, unsafe):
            self.assertTrue(feasible(r))
        self.assertFalse(feasible(bad))
        self.assertGreater(rank(c), rank(b))      # Coverage cao hơn thắng
        self.assertGreater(rank(b), rank(a))
        self.assertGreater(rank(a), rank(unsafe))  # CP upper vượt 0.02 luôn xếp sau cấu hình an toàn
        self.assertGreater(rank(unsafe), rank(bad))
        near = dict(b, selective_risk=0.002, coverage=0.951)  # gần coverage b → risk thấp hơn thắng
        self.assertGreater(rank(near), rank(b))

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
