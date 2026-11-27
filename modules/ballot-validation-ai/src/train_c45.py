"""
HUẤN LUYỆN C4.5 trên 8 đặc trưng cấu trúc (src/c45.py).
Chỉ huấn luyện trên tập train; validation dùng để báo cáo.
"""
import os

import joblib
import numpy as np
from sklearn.metrics import classification_report

from c45 import C45Classifier
from config import CFG, abspath
from feature_c45 import FEATURE_NAMES


def train_c45(F_tr, y_tr, F_va, y_va, feature_names=None, save=True, tag="c45_model", hp=None, model_dir=None):
    hp = dict(CFG["c45"], **(hp or {}))
    clf = C45Classifier(max_depth=hp["max_depth"], min_leaf=hp["min_leaf"], cf=hp["cf"])
    clf.fit(F_tr, y_tr, feature_names=feature_names or FEATURE_NAMES, class_names=CFG["labels"])

    print("🌳 Luật C4.5 (sau cắt tỉa):")
    print(clf.export_text())
    print("\nValidation:")
    print(classification_report(y_va, clf.predict(F_va), labels=range(len(CFG["labels"])),
                                target_names=CFG["labels"], zero_division=0))
    if save:
        path = os.path.join(model_dir or abspath(CFG["models_dir"]), f"{tag}.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(clf, path)
        print(f"✅ Đã lưu {path}")
    return clf
