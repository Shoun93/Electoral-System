"""Phase 1 regression tests. No dataset generation, model training, or OOF execution."""
import os, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cache import compute_cache_key, load_mark_prob_cache, save_mark_prob_cache
from decision import PolicyNotApprovedError, feasible, is_pending, screening_report, select_tau
from preprocess import check_group_disjoint, split_train_val

def fixture(n_groups=20, n_test_groups=4):
    groups, labels, splits = [], [], []
    for group in range(n_groups):
        for label in range(5):
            groups.append(f"g{group:03d}")
            labels.append(label)
            splits.append("test" if group >= n_groups - n_test_groups else "train")
    return {"groups": np.asarray(groups), "y": np.asarray(labels), "split": np.asarray(splits)}

class TestGroupValidation(unittest.TestCase):
    def test_all_partitions_group_disjoint(self):
        data = fixture(); result = split_train_val(data, seed=13); groups = data["groups"]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse(set(groups[result[left]]) & set(groups[result[right]]))
        self.assertEqual(result["meta"]["leakage_check"], "PASS")
        self.assertTrue(all(result["meta"]["leakage_checks"].values()))
    def test_test_indices_unchanged(self):
        data = fixture(); result = split_train_val(data)
        np.testing.assert_array_equal(result["test"], np.flatnonzero(data["split"] == "test"))
    def test_missing_key_fails(self):
        with self.assertRaises(ValueError): split_train_val({"split": np.array(["train"]), "y": np.array([0])})
    def test_length_mismatch_fails(self):
        data = fixture(); data["groups"] = data["groups"][:-1]
        with self.assertRaises(ValueError): split_train_val(data)
    def test_empty_group_in_test_fails(self):
        data = fixture(); data["groups"][-1] = " "
        with self.assertRaises(ValueError): split_train_val(data)
    def test_train_test_overlap_fails(self):
        data = fixture(); data["groups"][-1] = data["groups"][0]
        with self.assertRaises(ValueError): split_train_val(data)
    def test_unknown_partition_fails(self):
        data = fixture(); data["split"][0] = "validation"
        with self.assertRaises(ValueError): split_train_val(data)
    def test_empty_test_fails(self):
        data = fixture(); data["split"][:] = "train"
        with self.assertRaises(ValueError): split_train_val(data)
    def test_disjoint_rejects_blank_and_overlap(self):
        with self.assertRaises(ValueError): check_group_disjoint(["g1", ""], ["g2"])
        with self.assertRaises(ValueError): check_group_disjoint(["g1"], ["g1"])

class TestDecisionMetrics(unittest.TestCase):
    def test_pending_is_referral_not_auto_detection(self):
        y=np.array([1,2,3,0,0]); pred=np.array([0,2,0,0,0]); pending=np.array([False,False,True,False,False])
        r=screening_report(y,pred,pending,invalid_labels=[1,2,3])
        self.assertAlmostEqual(r["automatic_invalid_recall"],1/3,places=4)
        self.assertAlmostEqual(r["invalid_referral_rate"],1/3,places=4)
        self.assertAlmostEqual(r["invalid_interception_rate"],2/3,places=4)
        self.assertFalse(r["pending_counted_as_automatic_detection"])
    def test_unresolved_policy_does_not_infer_blank(self):
        r=screening_report(np.array([0,1,2]),np.array([0,1,2]),np.array([False]*3))
        self.assertEqual(r["invalid_metrics_status"],"BLOCKED_POLICY_NOT_APPROVED")
        self.assertNotIn("automatic_invalid_recall",r)
    def test_zero_invalid_denominator(self):
        r=screening_report(np.array([0,0]),np.array([0,0]),np.array([False,True]),invalid_labels=[1,2,3,4])
        self.assertEqual(r["invalid_metrics_status"],"UNDEFINED_NO_INVALID_SAMPLES")
        self.assertIsNone(r["automatic_invalid_recall"])
    def test_zero_coverage_has_undefined_risk(self):
        r=screening_report(np.array([1,2]),np.array([1,2]),np.array([True,True]),invalid_labels=[1,2])
        self.assertEqual(r["coverage"],0.0); self.assertIsNone(r["selective_risk"])
    def test_length_mismatch(self):
        with self.assertRaises(ValueError): screening_report(np.array([0,1]),np.array([0]),np.array([False,False]))
    def test_empty_inputs(self):
        with self.assertRaises(ValueError): screening_report(np.array([]),np.array([]),np.array([]))
    def test_pending_rule(self):
        r=is_pending(np.array([0.99,0.5]),np.array([0.99,0.9]),0.9,0.8)
        self.assertEqual(r.tolist(),[False,True])
    def test_bad_threshold(self):
        with self.assertRaises(ValueError): is_pending(np.array([0.9]),np.array([0.9]),1.2,0.8)
    def test_acceptance_and_oof_gated(self):
        self.assertFalse(feasible({"coverage":1.0,"selective_risk":0.0}))
        with self.assertRaises(PolicyNotApprovedError):
            select_tau(np.array([0,1]),np.array([0,1]),np.array([0.9,0.9]),np.array([0.9,0.9]))

class TestCache(unittest.TestCase):
    def setUp(self):
        self.arrays={"images":np.arange(12).reshape(4,3),"groups":np.array(["a","a","b","b"])}
        self.config={"epochs":2,"lr":0.001}
    def test_key_changes_for_data_config_seed(self):
        key=compute_cache_key(self.arrays,self.config,"hash",seed=1,folds=2)
        changed=dict(self.arrays); changed["images"]=self.arrays["images"].copy(); changed["images"][0,0]+=1
        self.assertNotEqual(key,compute_cache_key(changed,self.config,"hash",1,2))
        self.assertNotEqual(key,compute_cache_key(self.arrays,{"epochs":3},"hash",1,2))
        self.assertNotEqual(key,compute_cache_key(self.arrays,self.config,"hash",2,2))
    def test_cache_roundtrip_and_mismatch(self):
        key=compute_cache_key(self.arrays,self.config,"hash")
        p=np.full((3,5),0.2,dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path=os.path.join(d,"cache.npz"); save_mark_prob_cache(path,key,p,p[:2],p[:1])
            self.assertIsNotNone(load_mark_prob_cache(path,key,{"tr":3,"va":2,"te":1}))
            self.assertIsNone(load_mark_prob_cache(path,"wrong",{"tr":3,"va":2,"te":1}))
    def test_corrupt_cache_is_miss(self):
        with tempfile.TemporaryDirectory() as d:
            path=os.path.join(d,"cache.npz")
            with open(path,"wb") as f: f.write(b"not a zip")
            self.assertIsNone(load_mark_prob_cache(path,"key",{"tr":1,"va":1,"te":1}))

if __name__ == "__main__":
    unittest.main()


class TestOOFSelectionGate(unittest.TestCase):
    def test_flat_oof_grid_is_disabled(self):
        from decision import grid_search
        with self.assertRaises(PolicyNotApprovedError):
            grid_search(None, None, None, None)

if __name__ == "__main__":
    unittest.main()
