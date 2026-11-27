"""Kiểm thử metric Invalid→Valid, cờ dấu ngoài ô, manifest và cách ly run.
Chạy:  python -m unittest discover -s tests -v      (không cần dữ liệu, không train, không đọc Test)"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import manifest as mf                                          # noqa: E402
from decision import clopper_pearson_upper as cp_decision      # noqa: E402
from evaluate import clopper_pearson_upper, invalid_to_valid_report  # noqa: E402

LABELS = ["valid", "blank", "erased", "wrong_mark", "multi_mark"]
INVALID = [1, 2, 3, 4]      # chỉ dùng cho test; chưa phải định nghĩa được PM phê duyệt


def _oof_like():
    """Tái dựng 3/1400 (R1) và 3/132 (R2): 1400 phiếu không hợp lệ, 132 phiếu tự động 'valid', 3 trong số đó sai."""
    y = np.array([1] * 1400 + [0] * 200)
    pred = np.array([0] * 3 + [1] * 1397 + [0] * 129 + [1] * 71)
    pend = np.zeros(len(y), dtype=bool)
    return y, pred, pend


class TestClopperPearson(unittest.TestCase):
    def test_k0_exact(self):
        n = 100
        self.assertAlmostEqual(clopper_pearson_upper(0, n), 1 - 0.05 ** (1 / n), places=10)

    def test_k_ge_n_is_one(self):
        self.assertEqual(clopper_pearson_upper(5, 5), 1.0)

    def test_zero_denominator_none(self):
        self.assertIsNone(clopper_pearson_upper(0, 0))

    def test_single_definition_shared(self):
        self.assertIs(cp_decision, clopper_pearson_upper)


class TestInvalidToValid(unittest.TestCase):
    def test_blocked_without_policy(self):
        y, pred, pend = _oof_like()
        self.assertEqual(invalid_to_valid_report(y, pred, pend), {"status": "BLOCKED_POLICY_NOT_APPROVED"})
        with self.assertRaises(ValueError):
            invalid_to_valid_report(y, pred, pend, [0, 1])

    def test_oof_r1_r2_reproduced(self):
        y, pred, pend = _oof_like()
        r = invalid_to_valid_report(y, pred, pend, INVALID, labels=LABELS)
        self.assertEqual((r["N_inv_to_valid"], r["N_invalid"], r["N_AUTO_ACCEPT"]), (3, 1400, 132))
        self.assertAlmostEqual(r["R1"], 0.002142857, places=8)
        self.assertAlmostEqual(r["R2"], 0.022727273, places=8)
        self.assertAlmostEqual(r["CP_upper_R1"], 0.00553, places=5)
        self.assertAlmostEqual(r["CP_upper_R2"], 0.05769, places=4)

    def test_per_label_and_sum_equals_total(self):
        y, pred, pend = _oof_like()
        pred = pred.copy()
        pred[1400 + 10] = 0                       # một phiếu valid-thật nhưng tự động 'valid' → không tính (nhãn valid bị loại)
        y = y.copy()
        y[0] = 2; pred[0] = 0                     # thêm 1 phiếu 'erased' bị nhận nhầm → tổng 4 trường hợp
        r = invalid_to_valid_report(y, pred, pend, INVALID, labels=LABELS)
        self.assertTrue(r["sum_by_label_equals_total"])
        self.assertEqual(sum(v["auto_accept"] for v in r["by_label"].values()), r["N_inv_to_valid"])
        self.assertAlmostEqual(sum(v["R2"] for v in r["by_label"].values()), r["R2"], places=10)
        self.assertEqual(set(r["by_label"]), {"blank", "erased", "wrong_mark", "multi_mark"})

    def test_zero_denominator_na(self):
        y, pred, _ = _oof_like()
        all_pend = np.ones(len(y), dtype=bool)
        r = invalid_to_valid_report(y, pred, all_pend, INVALID, labels=LABELS)
        self.assertEqual(r["N_AUTO_ACCEPT"], 0)
        self.assertIsNone(r["R2"])
        self.assertIsNone(r["CP_upper_R2"])
        self.assertIsNone(r["by_label"]["blank"]["R2"])

    def test_no_invalid_zero_r1(self):
        y = np.zeros(10, dtype=int)
        r = invalid_to_valid_report(y, np.zeros(10, dtype=int), np.zeros(10, bool), INVALID, labels=LABELS)
        self.assertIsNone(r["R1"])


class TestManifestAndRuns(unittest.TestCase):
    def setUp(self):
        self.res = tempfile.mkdtemp()

    def test_run_dir_refuses_duplicate(self):
        mf.create_run_dir(self.res, "RUN-A")
        with self.assertRaises(FileExistsError):
            mf.create_run_dir(self.res, "RUN-A")

    def test_run_ids_unique(self):
        self.assertNotEqual(mf.new_run_id(), mf.new_run_id())

    def test_exclusive_write_refuses_overwrite(self):
        p = os.path.join(self.res, "x.txt")
        mf.write_exclusive(p, "one")
        with self.assertRaises(FileExistsError):
            mf.write_exclusive(p, "two")
        self.assertEqual(open(p, encoding="utf-8").read(), "one")

    def test_manifest_no_overwrite(self):
        d = mf.create_run_dir(self.res, "RUN-B")
        mf.write_manifest(d, {"run_id": "RUN-B"})
        with self.assertRaises(FileExistsError):
            mf.write_manifest(d, {"run_id": "RUN-B", "seed": 1})

    def test_locked_artifact_is_readonly(self):
        p = os.path.join(self.res, "locked.csv")
        mf.write_exclusive(p, "a,b\n")
        mf.lock_artifacts([p])
        self.assertEqual(os.stat(p).st_mode & 0o777, 0o444)
        os.chmod(p, 0o644)                                     # dọn để tempdir xóa được

    def test_test_registry_append_only(self):
        self.assertEqual(mf.read_test_registry(self.res), [])
        mf.register_test_read(self.res, "RUN-1", "abc")
        mf.register_test_read(self.res, "RUN-2", "abc")
        reg = mf.read_test_registry(self.res)
        self.assertEqual([e["run_id"] for e in reg], ["RUN-1", "RUN-2"])
        self.assertTrue(len(reg[0]) >= 2)                      # mục cũ còn nguyên

    def test_latest_run_requires_manifest(self):
        with self.assertRaises(FileNotFoundError):
            mf.latest_run_dir(self.res)
        d = mf.create_run_dir(self.res, "RUN-C")
        mf.write_latest(self.res, "RUN-C")
        with self.assertRaises(FileNotFoundError):
            mf.latest_run_dir(self.res)
        mf.write_manifest(d, {"run_id": "RUN-C"})
        self.assertEqual(mf.latest_run_dir(self.res), d)


class TestStrayFlag(unittest.TestCase):
    def test_extra_flag_forces_pending(self):
        from decision import is_pending
        p = is_pending([0.99, 0.99], [0.99, 0.99], 0.5, 0.5, extra=[False, True])
        self.assertEqual(p.tolist(), [False, True])
        with self.assertRaises(ValueError):
            is_pending([0.99, 0.99], [0.99, 0.99], 0.5, 0.5, extra=[True])

    def test_stray_threshold_keeps_recall(self):
        from stray_cnn import stray_threshold
        for n in (100, 150, 151):
            p = np.linspace(0, 1, n)
            for recall in (0.95, 0.97, 0.99):
                t = stray_threshold(p, np.ones(n, bool), recall)
                self.assertGreaterEqual((p >= t).mean(), recall, (n, recall))
        with self.assertRaises(ValueError):
            stray_threshold(p, np.zeros(len(p), bool))

    def test_maxpool_keeps_thin_stroke(self):
        from stray_cnn import maxpool_input
        b = np.zeros((850, 1200), np.uint8)
        b[400, 600] = 1                             # nét 1 px
        out = maxpool_input(b)
        self.assertEqual(out.shape, (213, 300))
        self.assertEqual(out.max(), 1.0)


if __name__ == "__main__":
    unittest.main()
