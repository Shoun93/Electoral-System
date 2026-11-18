"""
Đọc configs/config.yaml (đường dẫn tính theo vị trí file, không phụ thuộc thư mục làm việc).
Mọi module khác: from config import CFG, MODULE_DIR
"""
import os
import yaml

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # mã nguồn module
MODULE_DIR = os.path.abspath(os.environ.get("BALLOT_ROOT") or SRC_DIR)      # gốc dữ liệu + kết quả (BALLOT_ROOT: tập giữ lại)
CONFIG_PATH = os.path.join(SRC_DIR, "configs", "config.yaml")


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def abspath(rel):
    """Đổi đường dẫn tương đối trong config thành đường dẫn tuyệt đối."""
    return os.path.join(MODULE_DIR, rel)


CFG = load_config()
