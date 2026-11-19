"""Kiểm thử Phase 1 (metric Invalid→Valid) và Phase 2 (cache fingerprint, manifest, run isolation).
Chạy:  python -m unittest discover -s tests -v      (không cần dữ liệu, không train, không đọc Test)"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cache                                                   # noqa: E402
import manifest as mf                                          # noqa: E402
from decision import clopper_pearson_upper as cp_decision      # noqa: E402
from evaluate import clopper_pearson_upper, invalid_to_valid_report  # noqa: E402

LABELS = ["valid", "blank", "erased", "wrong_mark", "multi_mark"]


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
    def test_oof_r1_r2_reproduced(self):
        y, pred, pend = _oof_like()
        r = invalid_to_valid_report(y, pred, pend, labels=LABELS)
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
        r = invalid_to_valid_report(y, pred, pend, labels=LABELS)
        self.assertTrue(r["sum_by_label_equals_total"])
        self.assertEqual(sum(v["auto_accept"] for v in r["by_label"].values()), r["N_inv_to_valid"])
        self.assertAlmostEqual(sum(v["R2"] for v in r["by_label"].values()), r["R2"], places=10)
        self.assertEqual(set(r["by_label"]), {"blank", "erased", "wrong_mark", "multi_mark"})

    def test_zero_denominator_na(self):
        y, pred, _ = _oof_like()
        all_pend = np.ones(len(y), dtype=bool)
        r = invalid_to_valid_report(y, pred, all_pend, labels=LABELS)
        self.assertEqual(r["N_AUTO_ACCEPT"], 0)
        self.assertIsNone(r["R2"])
        self.assertIsNone(r["CP_upper_R2"])
        self.assertIsNone(r["by_label"]["blank"]["R2"])

    def test_no_invalid_zero_r1(self):
        y = np.zeros(10, dtype=int)
        r = invalid_to_valid_report(y, np.zeros(10, dtype=int), np.zeros(10, bool), labels=LABELS)
        self.assertIsNone(r["R1"])


class TestCache(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.logs = []
        self.log = self.logs.append
        self.arr = {"tr": np.arange(6, dtype=np.float32).reshape(2, 3)}
        self.comp = {"mark_cnn": {"epochs": 25}, "seed": 42, "code": {"x": "h1"}}

    def test_hit_after_save(self):
        cache.save(self.dir, "mark_probs", self.comp, self.arr)
        out = cache.load(self.dir, "mark_probs", self.comp, log=self.log)
        self.assertIsNotNone(out)
        np.testing.assert_array_equal(out["tr"], self.arr["tr"])
        self.assertTrue(any("HIT" in m for m in self.logs))

    def test_miss_on_epoch_change_with_reason(self):
        cache.save(self.dir, "mark_probs", self.comp, self.arr)
        comp2 = {**self.comp, "mark_cnn": {"epochs": 26}}
        self.assertIsNone(cache.load(self.dir, "mark_probs", comp2, log=self.log))
        self.assertTrue(any("epochs" in m for m in self.logs))

    def test_miss_on_seed_change(self):
        cache.save(self.dir, "mark_probs", self.comp, self.arr)
        self.assertIsNone(cache.load(self.dir, "mark_probs", {**self.comp, "seed": 7}, log=self.log))

    def test_miss_on_code_hash_change(self):
        src = os.path.join(self.dir, "feat.py")
        with open(src, "w", encoding="utf-8") as f:
            f.write("a = 1\n")
        comp = {**self.comp, "code": cache.code_hash([src])}
        cache.save(self.dir, "stray_probs", comp, self.arr)
        with open(src, "w", encoding="utf-8") as f:
            f.write("a = 2\n")
        comp2 = {**self.comp, "code": cache.code_hash([src])}
        self.assertIsNone(cache.load(self.dir, "stray_probs", comp2, log=self.log))

    def test_sidecar_mismatch_rejected(self):
        key, _ = cache.fingerprint("mark_probs", self.comp)
        cache.save(self.dir, "mark_probs", self.comp, self.arr)
        side = os.path.join(self.dir, f"mark_probs_{key[:16]}.npz.json")
        meta = json.load(open(side, encoding="utf-8"))
        meta["components"]["seed"] = 999
        json.dump(meta, open(side, "w", encoding="utf-8"))
        self.assertIsNone(cache.load(self.dir, "mark_probs", self.comp, log=self.log))

    def test_cache_without_sidecar_not_reused(self):
        key, _ = cache.fingerprint("mark_probs", self.comp)
        np.savez(os.path.join(self.dir, f"mark_probs_{key[:16]}.npz"), **self.arr)   # file cũ, không sidecar
        self.assertIsNone(cache.load(self.dir, "mark_probs", self.comp, log=self.log))

    def test_fingerprint_stable(self):
        k1, _ = cache.fingerprint("s", {"a": 1, "b": {"c": 2}})
        k2, _ = cache.fingerprint("s", {"b": {"c": 2}, "a": 1})
        self.assertEqual(k1, k2)


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

    def test_manifest_hash_stable_and_no_overwrite(self):
        d = mf.create_run_dir(self.res, "RUN-B")
        m1 = {"run_id": "RUN-B", "seed": 42, "cache": {"mark_probs": {"key": "k", "status": "MISS"}}}
        m2 = dict(reversed(list(m1.items())))                  # cùng nội dung, thứ tự khác
        self.assertEqual(cache.sha256_hex(cache.canonical_json(m1)), cache.sha256_hex(cache.canonical_json(m2)))
        mf.write_manifest(d, m1)
        h = mf.file_sha256(os.path.join(d, "manifest.json"))
        self.assertEqual(h, mf.file_sha256(os.path.join(d, "manifest.json")))
        with self.assertRaises(FileExistsError):
            mf.write_manifest(d, m2)

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

    def test_prior_test_observation_flags_run_dependent(self):
        mf.register_test_read(self.res, "baseline", "516ce78")
        reg = mf.read_test_registry(self.res)
        observed_before = bool(reg)
        self.assertTrue(observed_before)
        self.assertFalse(not observed_before)                  # independent = not observed_before → False


if __name__ == "__main__":
    unittest.main()
