"""
MANIFEST VÀ ARTIFACT THEO RUN.
  results/runs/<run_id>/{manifest.json, metrics.json, predictions.csv, oof_predictions.csv, log.txt, ...}
  - run_id duy nhất; thư mục tạo bằng exist_ok=False (trùng → lỗi, không ghi đè).
  - Mọi file ghi bằng mode "x" (exclusive).
  - Run có Test: đăng ký vào results/TEST_OBSERVED.json (chỉ thêm, không sửa). Test đã quan sát trước đó
    → test_observed_before = True, run này KHÔNG được coi là đánh giá độc lập.
  - Sau khi ghi artifact Test: chmod 0444 và locked = True.
"""
import contextlib
import datetime
import hashlib
import json
import os
import secrets
import subprocess
import sys

from cache import canonical_json, sha256_hex
from config import CFG, MODULE_DIR


def new_run_id():
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{secrets.token_hex(4)}"


def create_run_dir(results_dir, run_id):
    path = os.path.join(results_dir, "runs", run_id)
    os.makedirs(path, exist_ok=False)          # FileExistsError nếu trùng run_id
    return path


def write_exclusive(path, text):
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_sha256():
    return sha256_hex(canonical_json(CFG))


def git_info():
    def run(args):
        return subprocess.run(["git", *args], cwd=MODULE_DIR, capture_output=True, text=True, timeout=30)
    try:
        commit = run(["rev-parse", "HEAD"]).stdout.strip() or None
        dirty = bool(run(["status", "--porcelain"]).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _registry_path(results_dir):
    return os.path.join(results_dir, "TEST_OBSERVED.json")


def read_test_registry(results_dir):
    p = _registry_path(results_dir)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def register_test_read(results_dir, run_id, commit):
    """Chỉ THÊM mục mới. Không sửa mục cũ."""
    entries = read_test_registry(results_dir)
    entries.append({"run_id": run_id, "commit": commit,
                    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    tmp = _registry_path(results_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _registry_path(results_dir))


@contextlib.contextmanager
def tee_log(path):
    """Ghi stdout ra cả màn hình lẫn log.txt (mode exclusive)."""
    f = open(path, "x", encoding="utf-8")
    real = sys.stdout

    class _Tee:
        def write(self, s):
            real.write(s)
            f.write(s)

        def flush(self):
            real.flush()
            f.flush()

    sys.stdout = _Tee()
    try:
        yield
    finally:
        sys.stdout = real
        f.close()


def lock_artifacts(paths):
    for p in paths:
        os.chmod(p, 0o444)


def write_manifest(run_dir, manifest):
    write_exclusive(os.path.join(run_dir, "manifest.json"),
                    json.dumps(manifest, ensure_ascii=False, indent=1))
