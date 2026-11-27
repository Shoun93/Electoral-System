"""Checksum- and configuration-aware cache for per-box CNN probabilities."""
import hashlib
import json
import os
import tempfile

import numpy as np

CACHE_SCHEMA_VERSION = 2
_CACHE_KEYS = ("tr", "va", "te")


def _update_array(digest, name, value):
    arr = np.asarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(json.dumps(arr.shape).encode("ascii"))
    if arr.dtype.hasobject:
        payload = json.dumps(arr.tolist(), ensure_ascii=False, separators=(",", ":"))
        digest.update(payload.encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(arr).tobytes())


def compute_cache_key(arrays, config, model_sha256, seed=42, folds=5):
    """Hash all data, split, training configuration, model identity, and cache schema."""
    digest = hashlib.sha256()
    digest.update(f"mark-prob-cache-v{CACHE_SCHEMA_VERSION}".encode("ascii"))
    digest.update(json.dumps(config, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8"))
    digest.update(str(model_sha256 or "").encode("ascii"))
    digest.update(str(int(seed)).encode("ascii"))
    digest.update(str(int(folds)).encode("ascii"))
    for name in sorted(arrays):
        _update_array(digest, name, arrays[name])
    return digest.hexdigest()


def _valid_probabilities(value, expected_rows):
    arr = np.asarray(value)
    return (arr.ndim == 2 and arr.shape[0] == expected_rows
            and arr.shape[1] > 0 and np.issubdtype(arr.dtype, np.number)
            and np.isfinite(arr).all() and (arr >= 0).all() and (arr <= 1).all())


def load_mark_prob_cache(path, expected_key, expected_rows):
    """Return cached arrays only when key, schema, shapes, and values all validate."""
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"].item()) != CACHE_SCHEMA_VERSION:
                return None
            if str(data["cache_key"].item()) != expected_key:
                return None
            result = {key: np.asarray(data[key]) for key in _CACHE_KEYS}
        if not all(_valid_probabilities(result[k], expected_rows[k]) for k in _CACHE_KEYS):
            return None
        return result
    except (OSError, ValueError, KeyError, EOFError, TypeError):
        return None


def save_mark_prob_cache(path, cache_key, tr, va, te):
    """Atomically replace the cache so an interrupted write cannot leave a valid-looking file."""
    values = {"tr": np.asarray(tr), "va": np.asarray(va), "te": np.asarray(te)}
    if not all(_valid_probabilities(values[k], values[k].shape[0]) for k in _CACHE_KEYS):
        raise ValueError("Cache probabilities must be finite 2-D arrays in [0, 1]")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".mark-cache-", suffix=".npz", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, schema_version=np.asarray(CACHE_SCHEMA_VERSION),
                                cache_key=np.asarray(cache_key), **values)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
