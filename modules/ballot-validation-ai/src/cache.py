"""
CACHE CÓ FINGERPRINT.
Mỗi cache = <stage>_<key16>.npz + <stage>_<key16>.npz.json (sidecar lưu từng thành phần fingerprint).
  key = SHA256(canonical_json(components)).  Chỉ dùng cache khi sidecar khớp key VÀ components.
Cache cũ (không có sidecar) KHÔNG được tái sử dụng. Không xóa cache cũ. Miss luôn ghi rõ thành phần đã đổi.
"""
import glob
import hashlib
import importlib.metadata as md
import json
import os
import time

import numpy as np

from config import MODULE_DIR


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def array_sha256(a):
    a = np.ascontiguousarray(a)
    return sha256_hex(a.tobytes() + str(a.dtype).encode() + str(a.shape).encode())


def code_hash(rel_paths):
    """SHA-256 nội dung các file mã liên quan (đường dẫn tương đối với MODULE_DIR)."""
    return {p: sha256_hex(open(os.path.join(MODULE_DIR, p), "rb").read()) for p in sorted(rel_paths)}


def lib_versions(names):
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = "absent"
    return out


def fingerprint(stage, components):
    comp = {"stage": stage, **components}
    return sha256_hex(canonical_json(comp)), comp


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def explain_change(old, new):
    fo, fn = _flatten(old), _flatten(new)
    diffs = [f"{k}: {fo.get(k)!r} → {fn.get(k)!r}" for k in sorted(set(fo) | set(fn)) if fo.get(k) != fn.get(k)]
    return "; ".join(diffs) if diffs else "không rõ (fingerprint khác nhưng thành phần giống nhau)"


def _previous_sidecars(cache_dir, stage):
    paths = glob.glob(os.path.join(cache_dir, f"{stage}_*.npz.json"))
    return sorted(paths, key=os.path.getmtime, reverse=True)


def load(cache_dir, stage, components, log=print):
    """Trả về dict mảng nếu HIT, None nếu MISS (kèm lý do)."""
    key, comp = fingerprint(stage, components)
    base = os.path.join(cache_dir, f"{stage}_{key[:16]}.npz")
    side = base + ".json"
    if os.path.exists(base) and os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("key") == key and meta.get("components") == comp:
            log(f"[cache] HIT {stage} {key[:16]}")
            d = np.load(base, allow_pickle=False)
            return {k: d[k] for k in d.files}
        log(f"[cache] MISS {stage} {key[:16]}: sidecar không khớp")
        return None
    prev = _previous_sidecars(cache_dir, stage)
    if not prev:
        log(f"[cache] MISS {stage} {key[:16]}: chưa có cache cho stage này")
    else:
        with open(prev[0], encoding="utf-8") as f:
            old = json.load(f).get("components", {})
        log(f"[cache] MISS {stage} {key[:16]}: đổi {explain_change(old, comp)}")
    return None


def save(cache_dir, stage, components, arrays):
    """Ghi mảng và sidecar. Ghi qua file tạm rồi os.replace để không để lại file dở dang."""
    key, comp = fingerprint(stage, components)
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.join(cache_dir, f"{stage}_{key[:16]}.npz")
    tmp = base + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, base)
    with open(base + ".json", "w", encoding="utf-8") as f:   # cache theo key: ghi lại cùng nội dung là an toàn
        json.dump({"stage": stage, "key": key, "components": comp, "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  f, ensure_ascii=False, indent=1)
    return key
