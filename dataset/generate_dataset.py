"""
Sinh bộ dữ liệu thẻ cử tri GIẢ LẬP (2500 ảnh mặc định) theo bố cục Luật 85/2015/QH13.

Đặc điểm:
  * 5 nhãn: valid, blank, erased, wrong_mark, multi_mark; mỗi nhãn 500 mẫu (tổng 2500).
  * Mỗi mẫu có 4 mức khó: easy, medium, hard, boundary (boundary = sát ngưỡng định nghĩa).
  * Nhãn SUY RA từ hình học đã render bằng oracle(), không dùng ý định của bộ sinh.
    Mẫu nào oracle không khớp nhãn mục tiêu sẽ bị loại và sinh lại.
  * Một group = một phiếu gốc (cùng cử tri, cùng template); mỗi group có đúng 1 mẫu/lớp.
    Chia train/test theo group, nên không rò rỉ template giữa hai tập.
  * Ghi đầy đủ: ảnh, nhãn CSV (gồm các cột tương thích pipeline), tọa độ ROI đã xoay,
    metadata từng mẫu (JSONL), báo cáo, ảnh preview, và kết quả kiểm tra.

Cách chạy (từ thư mục gốc repo):
  python dataset/generate_dataset.py                       # 2500 ảnh, seed 42
  python dataset/generate_dataset.py --samples-per-class 100 --seed 7 --out /tmp/test

Đầu ra (dưới <out>/data/):
  raw/ballot_vXXXXX.png          ảnh phiếu (grayscale 1200×850)
  labels/all.csv                 nhãn + split + metadata
  labels/boxes.json              ROI đã xoay: boxes (5 ô), left_band, extra_zone
  metadata/samples.jsonl         spec hình học từng mẫu (để tái dựng và kiểm tra nhãn)
  metadata/rules.json            ngưỡng định nghĩa nhãn và thống kê sinh mẫu
  metadata/feature_policy.json   cột CẤM dùng làm đặc trưng cho model
  metadata/report.txt            thống kê
  metadata/validation.json       kết quả kiểm tra
  preview/*.png                  ảnh xem nhanh

TOÀN BỘ DỮ LIỆU CÁ NHÂN TRÊN THẺ LÀ GIẢ LẬP.
"""

import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
import random
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sklearn.model_selection import GroupShuffleSplit

# =============================================================================
# CẤU HÌNH
# =============================================================================
RANDOM_SEED = 42
SAMPLES_PER_CLASS = 500              # 5 nhãn × 500 = 2500 ảnh
CLASSES = ["valid", "blank", "erased", "wrong_mark", "multi_mark"]
MAX_VOTES = 3                        # chọn tối đa 3 trong 5 đại biểu
MIN_VOTES_VALID = 1
IMG_W, IMG_H = 1200, 850
W, H = IMG_W, IMG_H
TEST_SIZE = 0.2

CANDIDATE_NAMES = [
    "Nguyễn Văn A",
    "Trần Văn B",
    "Lê Thị C",
    "Lý Thị D",
    "Nguyễn Thị E",
]
N_CANDIDATES = len(CANDIDATE_NAMES)
NC = N_CANDIDATES

if not (1 <= MIN_VOTES_VALID <= MAX_VOTES < N_CANDIDATES <= 6):
    raise ValueError("Cần 1 <= MIN_VOTES_VALID <= MAX_VOTES < N_CANDIDATES <= 6.")

# Ngưỡng định nghĩa nhãn (đơn vị: pixel mực, tính theo alpha)
VOTE_RATIO = 0.90        # ≥90% mực của dấu nằm trong ô → là một phiếu bầu cho ô đó
VOTE_MIN_W = 60.0        # dấu hợp lệ phải có mực tối thiểu
STRAY_MIN_W = 120.0      # dấu ngoài ô có mực ≥ ngưỡng này → dấu sai (wrong_mark)
SPECK_MAX_W = 30.0       # chấm nhỏ dưới ngưỡng này → bỏ qua
STRIKE_INK = 0.18        # ô bị gạch nếu mực gạch phủ ≥18% diện tích ô
STRIKE_COUNT = 4         # ≥4 ô bị gạch → erased
ORDER_VERSION = "2.0"

DIFFS = ["easy", "medium", "hard", "boundary"]
DIFF_RATIO = [0.20, 0.30, 0.30, 0.20]
SHAPES = ["tick", "check", "x", "slash", "hand", "double"]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "modules", "ballot-validation-ai"))

# =============================================================================
# DỮ LIỆU GIẢ
# =============================================================================
SURNAMES = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Vũ", "Đặng", "Bùi", "Đỗ", "Lý", "Võ", "Phan", "Ngô"]
MIDDLE_MALE = ["Văn", "Hữu", "Minh", "Quốc", "Thanh", "Đức", "Công"]
MIDDLE_FEMALE = ["Thị", "Ngọc", "Thanh", "Kim", "Mai", "Hồng"]
FIRST_MALE = ["An", "Bình", "Cường", "Dũng", "Khoa", "Nam", "Sơn", "Hùng", "Phúc", "Long", "Tuấn", "Hải"]
FIRST_FEMALE = ["Hà", "Hạnh", "Linh", "Phương", "Trang", "Lan", "Thảo", "Ngân", "Yến", "Vy", "Nhi"]
STREETS = ["Lê Lợi", "Nguyễn Huệ", "Trần Hưng Đạo", "Cách Mạng Tháng Tám", "Điện Biên Phủ", "Võ Văn Tần",
           "Lý Thường Kiệt", "Nguyễn Trãi", "Phan Xích Long", "Hoàng Văn Thụ", "Quang Trung", "Nguyễn Thị Minh Khai"]
WARDS = ["Phường Bến Thành", "Phường Sài Gòn", "Phường Tân Định", "Phường Cầu Ông Lãnh", "Phường Thủ Đức",
         "Phường Linh Xuân", "Phường An Khánh", "Phường Gia Định", "Phường Bình Thạnh", "Phường Tân Sơn Hòa",
         "Phường Nhiêu Lộc", "Phường Chợ Lớn", "Phường Bình Tây", "Phường Phú Nhuận", "Phường Hạnh Thông",
         "Xã Bình Chánh", "Xã Củ Chi", "Xã Cần Giờ", "Đặc khu Côn Đảo"]
PROVINCE_CITY = "Thành phố Hồ Chí Minh"
PROVINCE_CODES = ["079", "001", "048", "031", "092", "089", "075", "074", "056", "066", "038", "044"]
BIRTH_MIN = datetime.date(1946, 1, 1)
BIRTH_MAX = datetime.date(2008, 3, 15)


def random_name(gender=None):
    gender = gender or random.choice(["Nam", "Nữ"])
    if gender == "Nam":
        mid, first = random.choice(MIDDLE_MALE), random.choice(FIRST_MALE)
    else:
        mid, first = random.choice(MIDDLE_FEMALE), random.choice(FIRST_FEMALE)
    return f"{random.choice(SURNAMES)} {mid} {first}"


def random_birthdate():
    days = (BIRTH_MAX - BIRTH_MIN).days
    return BIRTH_MIN + datetime.timedelta(days=random.randint(0, days))


def random_cccd(birth, gender):
    century_gender = (birth.year // 100 - 19) * 2 + (0 if gender == "Nam" else 1)
    return f"{random.choice(PROVINCE_CODES)}{century_gender}{birth.year % 100:02d}{random.randint(0, 999999):06d}"


def random_voter():
    gender = random.choice(["Nam", "Nữ"])
    birth = random_birthdate()
    return {
        "ho_ten": random_name(gender),
        "ngay_sinh": birth.strftime("%d/%m/%Y"),
        "gioi_tinh": gender,
        "noi_cu_tru_1": f"Số {random.randint(1, 450)} đường {random.choice(STREETS)}",
        "noi_cu_tru_2": f"Khu phố {random.randint(1, 12)}",
        "so_can_cuoc": random_cccd(birth, gender),
        "so_the_cu_tri": f"{random.randint(0, 99999999):08d}",
        "khu_vuc_bo_phieu": str(random.randint(1, 80)),
        "xa_phuong": random.choice(WARDS),
        "tinh_tp": PROVINCE_CITY,
    }


def random_outside_name():
    name = random_name()
    while name in CANDIDATE_NAMES:
        name = random_name()
    return name


# =============================================================================
# FONT
# =============================================================================
_FONT_FILES = {
    "regular": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/dejavu/DejaVuSerif.ttf",
        "C:/Windows/Fonts/times.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ],
    "bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSerif-Bold.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ],
}


@lru_cache(maxsize=None)
def get_font(size=22, bold=False):
    for path in _FONT_FILES["bold" if bold else "regular"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    raise RuntimeError("Không tìm thấy font hỗ trợ tiếng Việt. Cài DejaVu hoặc thêm đường dẫn vào _FONT_FILES.")


# =============================================================================
# BỐ CỤC THẺ
# =============================================================================
LEFT_X = 12
FIELD_DOTS_END = 520
BOX_X1, BOX_X2, BOX_SIZE = 671, 707, 36
CAND_Y0, CAND_STEP = 370, 43
NAME_X = 722


def _dotted_line(draw, x_start, x_end, y):
    for xx in range(int(x_start), int(x_end), 5):
        draw.rectangle([xx, y + 9, xx + 1, y + 10], fill="black")


def _draw_field(draw, label, value, y):
    font = get_font(21)
    draw.text((LEFT_X, y), label, font=font, fill="black", anchor="lm")
    x_start = LEFT_X + draw.textlength(label, font=font) + 6
    _dotted_line(draw, x_start, FIELD_DOTS_END, y)
    if value:
        draw.text((x_start + 6, y - 2), value, font=font, fill="black", anchor="lm")


def draw_ballot_template(candidates, voter):
    """Vẽ thẻ cử tri. Trả về (ảnh, 5 ô đánh dấu ở hệ tọa độ ảnh gốc)."""
    img = Image.new("RGB", (IMG_W, IMG_H), "white")
    draw = ImageDraw.Draw(img)

    draw.text((304, 14), "ỦY BAN NHÂN DÂN", font=get_font(20, True), fill="black", anchor="mm")
    draw.text((883, 14), "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", font=get_font(20, True), fill="black", anchor="mm")
    draw.text((304, 52), "Tp. Hồ Chí Minh", font=get_font(20, True), fill="black", anchor="mm")
    slogan = "Độc lập - Tự do - Hạnh phúc"
    draw.text((883, 52), slogan, font=get_font(20, True), fill="black", anchor="mm")
    bb = draw.textbbox((883, 52), slogan, font=get_font(20, True), anchor="mm")
    draw.line([bb[0], bb[3] + 2, bb[2], bb[3] + 2], fill="black", width=2)

    draw.text((603, 95), "THẺ CỬ TRI", font=get_font(32, True), fill="black", anchor="mm")
    for text, y in [("BẦU CỬ ĐẠI BIỂU QUỐC HỘI KHÓA XVI", 135),
                    ("VÀ BẦU CỬ ĐẠI BIỂU HỘI ĐỒNG NHÂN DÂN CÁC CẤP", 173),
                    ("NHIỆM KỲ 2026–2031", 212)]:
        draw.text((603, y), text, font=get_font(23, True), fill="black", anchor="mm")

    fields = [
        ("Họ và tên:", voter["ho_ten"], 250),
        ("Ngày, tháng, năm sinh:", voter["ngay_sinh"], 292),
        ("Giới tính:", voter["gioi_tinh"], 334),
        ("Nơi cư trú:", voter["noi_cu_tru_1"], 376),
        ("", voter["noi_cu_tru_2"], 419),
        ("Số Căn cước:", voter["so_can_cuoc"], 460),
        ("SỐ THẺ CỬ TRI:", voter["so_the_cu_tri"], 502),
        ("KHU VỰC BỎ PHIẾU SỐ:", voter["khu_vuc_bo_phieu"], 544),
        ("Xã/phường/đặc khu:", voter["xa_phuong"], 585),
        ("Tỉnh/thành phố:", voter["tinh_tp"], 627),
    ]
    for label, value, y in fields:
        _draw_field(draw, label, value, y)

    draw.text((905, 292), "Đại biểu", font=get_font(22, True), fill="black", anchor="mm")
    bb = draw.textbbox((905, 292), "Đại biểu", font=get_font(22, True), anchor="mm")
    draw.line([bb[0], bb[3] + 2, bb[2], bb[3] + 2], fill="black", width=2)

    boxes = []
    for i, name in enumerate(candidates):
        yc = CAND_Y0 + i * CAND_STEP
        box = (BOX_X1, yc - BOX_SIZE // 2, BOX_X2, yc + BOX_SIZE // 2)
        draw.rectangle(box, fill=(242, 242, 242), outline=(110, 110, 110), width=2)
        draw.text((NAME_X, yc), name, font=get_font(30), fill="black", anchor="lm")
        boxes.append(box)

    draw.text((IMG_W - 30, 690), "Tp. Hồ Chí Minh, ngày ....... tháng ....... năm 2026",
              font=get_font(21), fill="black", anchor="rm")
    draw.text((930, 728), "TM. ỦY BAN NHÂN DÂN", font=get_font(22, True), fill="black", anchor="mm")
    draw.text((930, 760), "CHỦ TỊCH", font=get_font(22, True), fill="black", anchor="mm")
    return img, boxes


def rotate_box(box, angle_deg, w, h):
    """Áp đúng phép xoay của PIL Image.rotate(angle, expand=False) lên một box."""
    t = math.radians(angle_deg)
    cx, cy = w / 2, h / 2
    x1, y1, x2, y2 = box
    pts = []
    for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        dx, dy = x - cx, y - cy
        pts.append((dx * math.cos(t) + dy * math.sin(t) + cx,
                    -dx * math.sin(t) + dy * math.cos(t) + cy))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]


def layout_shapes(boxes):
    """Hai vùng cố định trên thẻ: dải bên trái ô (dấu sai vị trí) và vùng ghi thêm tên."""
    top, bottom = boxes[0][1] - 6, boxes[-1][3] + 6
    left_band = (BOX_X1 - 70, top, BOX_X1 - 24, bottom)
    extra_zone = (NAME_X, boxes[-1][3] + 25, IMG_W - 40, boxes[-1][3] + 60)
    return left_band, extra_zone


# =============================================================================
# RENDER DẤU (spec hình học → mask)
# =============================================================================
class Ambiguous(Exception):
    """Cấu hình hình học không đạt ngưỡng nhãn rõ ràng → sinh lại."""


def _render(spec):
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    fill = int(round(255 * spec["alpha"]))
    if spec["render"] == "line":
        for pl in spec["polylines"]:
            d.line([tuple(p) for p in pl], fill=fill, width=int(spec["width"]), joint="curve")
    elif spec["render"] == "dot":
        cx, cy = spec["center"]
        r = spec["r"]
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif spec["render"] == "text":
        d.text(tuple(spec["pos"]), spec["text"], font=get_font(24), fill=fill, anchor="lm")
    else:
        raise ValueError(spec["render"])
    mask = np.asarray(img, dtype=np.uint8)
    return {"spec": spec, "mask": mask, "weight": float(mask.sum()) / 255.0}


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def _ink_in(mask, b):
    return float(mask[b[1]:b[3], b[0]:b[2]].sum()) / 255.0


def _inside_ratio(m, b):
    return _ink_in(m["mask"], b) / max(m["weight"], 1e-9)


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _tick_pts(b, s=1.0, dx=0.0, dy=0.0, jit=None):
    cx, cy = _center(b)
    bw, bh = (b[2] - b[0]) * s, (b[3] - b[1]) * s
    unit = [(0.12, 0.50), (0.42, 0.82), (0.92, 0.12)]
    out = []
    for k, (u, v) in enumerate(unit):
        jx, jy = jit[k] if jit else (0.0, 0.0)
        out.append((cx + (u - 0.5) * bw + dx + jx, cy + (v - 0.5) * bh + dy + jy))
    return out


def _shape_polys(shape, b, s, dx, dy, rng, hand=0.0):
    cx, cy = _center(b)
    bw, bh = (b[2] - b[0]) * s, (b[3] - b[1]) * s
    if shape in ("tick", "check"):
        return [_tick_pts(b, s, dx, dy)]
    if shape == "x":
        return [[(cx - bw * .35 + dx, cy - bh * .35 + dy), (cx + bw * .35 + dx, cy + bh * .35 + dy)],
                [(cx - bw * .35 + dx, cy + bh * .35 + dy), (cx + bw * .35 + dx, cy - bh * .35 + dy)]]
    if shape == "slash":
        return [[(cx + bw * .4 + dx, cy - bh * .4 + dy), (cx - bw * .4 + dx, cy + bh * .4 + dy)]]
    if shape == "hand":
        jit = [(rng.uniform(-hand, hand), rng.uniform(-hand, hand)) for _ in range(3)]
        return [_tick_pts(b, s, dx, dy, jit)]
    if shape == "double":
        p = _tick_pts(b, s, dx, dy)
        return [p, [(x + 2.0, y + 1.0) for x, y in p]]
    raise ValueError(shape)


def _line_spec(polys, width, alpha, hint, box=None, shape=None, s=1.0, dx=0.0, dy=0.0):
    return {"render": "line", "polylines": [[[float(x), float(y)] for x, y in pl] for pl in polys],
            "width": int(width), "alpha": float(alpha), "hint": hint, "box": box, "shape": shape,
            "s": float(s), "dx": float(dx), "dy": float(dy)}


def _vote_spec(b, idx, shape="tick", s=1.0, dx=0.0, dy=0.0, width=4, alpha=1.0, hand=0.0, rng=None):
    return _line_spec(_shape_polys(shape, b, s, dx, dy, rng, hand), width, alpha, "vote", idx, shape, s, dx, dy)


def _strike_spec(b, idx, width, alpha=1.0, ext=4):
    x1, y1, x2, y2 = b
    polys = [[(x1 - ext, y1 - ext), (x2 + ext, y2 + ext)], [(x1 - ext, y2 + ext), (x2 + ext, y1 - ext)]]
    return _line_spec(polys, width, alpha, "erase", idx, "x")


def _text_spec(text, pos):
    return {"render": "text", "text": text, "pos": [float(pos[0]), float(pos[1])], "alpha": 1.0,
            "hint": "stray", "box": None, "shape": "text"}


def _dot_spec(c, r):
    return {"render": "dot", "center": [float(c[0]), float(c[1])], "r": float(r), "alpha": 1.0,
            "hint": "speck", "box": None, "shape": "dot"}


def _jit(rng, a):
    return rng.uniform(-a, a), rng.uniform(-a, a)


def _search_width(make, b, lo, hi):
    for w in range(2, 15):
        for a in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
            m = _render(make(w, a))
            f = _ink_in(m["mask"], b) / _area(b)
            if lo <= f <= hi:
                return m
    raise Ambiguous("không đạt tỉ lệ gạch")


def _search_edge(make, b, lo, hi, sign):
    for k in range(160):
        m = _render(make(sign * k * 0.5))
        r = _inside_ratio(m, b)
        if lo <= r <= hi:
            return m
    raise Ambiguous("không đạt vị trí biên")


def _pick_boxes(anchor, n, rng):
    others = [i for i in range(NC) if i != anchor]
    extra = list(rng.choice(others, n - 1, replace=False)) if n > 1 else []
    return [anchor] + [int(i) for i in extra]


def _neighbor(anchor):
    return anchor + 1 if anchor < NC - 1 else anchor - 1


# =============================================================================
# BỘ SINH THEO NHÃN VÀ MỨC KHÓ
# =============================================================================
def _b_valid(diff, anchor, boxes, rng):
    if diff == "easy":
        n = int(rng.choice([1, 2, 3], p=[.6, .25, .15]))
        ms = [_render(_vote_spec(boxes[i], i, "tick", 1.0, *_jit(rng, 1), width=4))
              for i in _pick_boxes(anchor, n, rng)]
        return ms, "valid_easy_clear"
    if diff == "medium":
        n = int(rng.integers(1, 4))
        ms = []
        for i in _pick_boxes(anchor, n, rng):
            sh = SHAPES[int(rng.integers(len(SHAPES)))]
            ms.append(_render(_vote_spec(boxes[i], i, sh, rng.uniform(.85, 1.05), *_jit(rng, 3),
                                         width=int(rng.integers(3, 5)), hand=2.0, rng=rng)))
        return ms, "valid_medium_shape"
    if diff == "hard":
        n = int(rng.integers(2, 4))
        idxs = _pick_boxes(anchor, n, rng)
        ms = []
        for i in idxs:
            sh = SHAPES[int(rng.integers(len(SHAPES)))]
            ms.append(_render(_vote_spec(boxes[i], i, sh, rng.uniform(.6, .8), *_jit(rng, 6),
                                         width=int(rng.integers(4, 6)), alpha=rng.uniform(.6, .75),
                                         hand=3.0, rng=rng)))
        if rng.random() < 0.4:
            j = idxs[-1]
            ms.append(_search_width(lambda w, a, j=j: _strike_spec(boxes[j], j, w, alpha=a), boxes[j], .35, .6))
            return ms, "valid_hard_faint_corrected"
        return ms, "valid_hard_faint"
    if diff == "boundary":
        s = float(rng.uniform(1.0, 1.15))
        sign = float(rng.choice([-1, 1]))
        b = boxes[anchor]
        m = _search_edge(lambda dx: _vote_spec(b, anchor, "tick", s, dx, 0.0, width=4), b, .93, 1.0, sign)
        return [m], "valid_boundary_edge"
    raise ValueError(diff)


def _b_wrong(diff, anchor, boxes, rng):
    b = boxes[anchor]
    j = _neighbor(anchor)
    if diff == "easy":
        if rng.random() < 0.5:
            return [_render(_text_spec(f"+ {random_outside_name()}", (NAME_X, boxes[-1][3] + 40)))], \
                "wrong_easy_extra_text"
        cxp = b[0] - 45
        cyp = _center(b)[1]
        pb = (cxp - 18, cyp - 18, cxp + 18, cyp + 18)
        return [_render(_line_spec(_shape_polys("tick", pb, 1.0, 0, 0, rng), 5, 1.0, "stray", shape="tick"))], \
            "wrong_easy_left_band"
    if diff == "medium":
        if rng.random() < 0.5:
            t = float(rng.choice([0.6, 0.4]))
            sign = float(rng.choice([-1, 1]))
            m = _search_edge(lambda dx: _line_spec(_shape_polys("tick", b, 1.0, dx, 0, rng), 5, 1.0, "stray",
                                                   anchor, "tick", 1.0, dx, 0),
                             b, t - 0.02, t + 0.02, sign)
            return [m], "wrong_medium_partial_outside"
        lo_i, hi_i = min(anchor, j), max(anchor, j)
        gy = (boxes[lo_i][3] + boxes[hi_i][1]) / 2.0
        cx = _center(b)[0]
        pb = (cx - 18, gy - 18, cx + 18, gy + 18)
        return [_render(_line_spec(_shape_polys("tick", pb, 1.2, 0, 0, rng), 5, 1.0, "stray", shape="tick"))], \
            "wrong_medium_between_boxes"
    if diff == "hard":
        if rng.random() < 0.5:
            bj = boxes[j]
            cxp = bj[0] - rng.uniform(3, 6) - 14
            cyp = _center(bj)[1]
            pb = (cxp - 18, cyp - 18, cxp + 18, cyp + 18)
            m = _render(_line_spec(_shape_polys("tick", pb, 1.0, 0, 0, rng), 5, 1.0, "stray", shape="tick"))
            case = "wrong_hard_near_other_box"
        else:
            x, y = rng.uniform(740, 1000), rng.uniform(600, 640)
            ang = rng.uniform(-0.6, 0.6)
            p1 = (x + 35 * np.cos(ang), y + 35 * np.sin(ang))
            m = _render(_line_spec([[(x, y), p1]], 4, 1.0, "stray"))
            case = "wrong_hard_small_accidental"
        ms = [m]
        if rng.random() < 0.5:
            ms.append(_render(_vote_spec(boxes[j], j, "tick", 1.0, *_jit(rng, 2), width=4)))
        return ms, case
    if diff == "boundary":
        t = 0.8 if rng.random() < 0.5 else 0.2
        sign = float(rng.choice([-1, 1]))
        m = _search_edge(lambda dx: _line_spec(_shape_polys("tick", b, 1.0, dx, 0, rng), 5, 1.0, "stray",
                                               anchor, "tick", 1.0, dx, 0),
                         b, t - 0.02, t + 0.02, sign)
        ms = [m]
        if rng.random() < 0.5:
            ms.append(_render(_vote_spec(boxes[j], j, "tick", 1.0, *_jit(rng, 2), width=4)))
        return ms, "wrong_boundary_partial_%d" % int(round((1 - t) * 100))
    raise ValueError(diff)


def _b_multi(diff, anchor, boxes, rng):
    if diff == "easy":
        n = int(rng.integers(4, NC + 1))
        ms = [_render(_vote_spec(boxes[i], i, "tick", 1.0, *_jit(rng, 1), width=4))
              for i in _pick_boxes(anchor, n, rng)]
        return ms, "multi_easy_clear"
    if diff == "medium":
        n = int(rng.integers(4, NC + 1))
        ms = []
        for k, i in enumerate(_pick_boxes(anchor, n, rng)):
            sh = "double" if k == 1 else SHAPES[int(rng.integers(len(SHAPES)))]
            ms.append(_render(_vote_spec(boxes[i], i, sh, rng.uniform(.9, 1.05), *_jit(rng, 3),
                                         width=int(rng.integers(3, 5)), hand=2.0, rng=rng)))
        return ms, "multi_medium_mixed"
    if diff == "hard":
        n = int(rng.integers(4, NC + 1))
        idxs = _pick_boxes(anchor, n, rng)
        ms = []
        for k, i in enumerate(idxs):
            if k == 1:
                ms.append(_render(_vote_spec(boxes[i], i, "tick", 1.0, *_jit(rng, 2), width=5,
                                             alpha=rng.uniform(.6, .7))))
            else:
                sh = SHAPES[int(rng.integers(len(SHAPES)))]
                ms.append(_render(_vote_spec(boxes[i], i, sh, rng.uniform(.7, .85), *_jit(rng, 4),
                                             width=4, hand=2.5, rng=rng)))
        return ms, "multi_hard_faint_one"
    if diff == "boundary":
        idxs = _pick_boxes(anchor, 4, rng)
        ms = [_render(_vote_spec(boxes[i], i, "tick", 1.0, *_jit(rng, 1), width=4)) for i in idxs[:3]]
        i = idxs[3]
        s = float(rng.uniform(1.0, 1.1))
        sign = float(rng.choice([-1, 1]))
        b = boxes[i]
        ms.append(_search_edge(lambda dx: _vote_spec(b, i, "tick", s, dx, 0.0, width=4), b, .93, 1.0, sign))
        return ms, "multi_boundary_edge"
    raise ValueError(diff)


def _b_erased(diff, anchor, boxes, rng):
    ms = []
    if diff == "easy":
        for i in range(NC):
            if rng.random() < 0.5:
                ms.append(_render(_vote_spec(boxes[i], i, "tick", 1.0, width=4)))
            ms.append(_search_width(lambda w, a, i=i: _strike_spec(boxes[i], i, w, alpha=a), boxes[i], .35, .6))
        ms.append(_render(_line_spec([[(boxes[0][0] - 20, boxes[0][1] - 10), (boxes[-1][2] + 20, boxes[-1][3] + 10)]],
                                     4, 0.9, "erase")))
        return ms, "erased_easy_full_cross"
    if diff == "medium":
        for i in range(NC):
            if rng.random() < 0.6:
                sh = SHAPES[int(rng.integers(len(SHAPES)))]
                ms.append(_render(_vote_spec(boxes[i], i, sh, 1.0, width=4, hand=2.0, rng=rng)))
            ms.append(_search_width(lambda w, a, i=i: _strike_spec(boxes[i], i, w, alpha=a), boxes[i], .25, .4))
        return ms, "erased_medium_residual"
    if diff == "hard":
        for i in range(NC):
            if rng.random() < 0.6:
                ms.append(_render(_vote_spec(boxes[i], i, "tick", rng.uniform(.8, 1.0), *_jit(rng, 3),
                                             width=4, alpha=rng.uniform(.5, .7))))
            ms.append(_search_width(lambda w, a, i=i: _strike_spec(boxes[i], i, w, alpha=a), boxes[i], .3, .6))
            if i in (1, 3):
                ms.append(_search_width(lambda w, a, i=i: _strike_spec(boxes[i], i, w, alpha=a, ext=6),
                                        boxes[i], .3, .6))
        return ms, "erased_hard_uneven"
    if diff == "boundary":
        for i in range(NC):
            ms.append(_search_width(lambda w, a, i=i: _strike_spec(boxes[i], i, w, alpha=a), boxes[i], .2, .25))
        ms.append(_render(_vote_spec(boxes[anchor], anchor, "tick", 1.0, width=4, alpha=0.6)))
        return ms, "erased_boundary_light"
    raise ValueError(diff)


def _b_blank(diff, anchor, boxes, rng):
    if diff == "easy":
        return [], "blank_easy_clean"
    if diff == "medium":
        ms = [_render(_dot_spec((rng.uniform(40, 1150), rng.uniform(790, 830)), rng.uniform(1, 2)))
              for _ in range(int(rng.integers(2, 5)))]
        return ms, "blank_medium_specks"
    if diff == "hard":
        i = int(rng.integers(NC))
        b = boxes[i]
        x = b[2] + 6 if rng.random() < 0.5 else b[0] - 6
        return [_render(_dot_spec((x, _center(b)[1]), 2.0))], "blank_hard_speck_near_box"
    if diff == "boundary":
        b = boxes[anchor]
        x = b[0] + (b[2] - b[0]) * rng.uniform(.3, .7)
        y = b[1] + (b[3] - b[1]) * rng.uniform(.3, .7)
        return [_render(_dot_spec((x, y), 1.5))], "blank_boundary_dot_in_box"
    raise ValueError(diff)


BUILDERS = {"valid": _b_valid, "blank": _b_blank, "erased": _b_erased,
            "wrong_mark": _b_wrong, "multi_mark": _b_multi}


# =============================================================================
# ORACLE: suy nhãn từ hình học
# =============================================================================
def oracle(marks, boxes):
    """Suy nhãn TỪ HÌNH HỌC (mask đã render). Không nhận nhãn mục tiêu làm đầu vào."""
    erase_union = np.zeros((H, W), np.uint8)
    votes, strays, roles = set(), 0, []
    for m in marks:
        sp = m["spec"]
        if sp["hint"] == "erase":
            erase_union = np.maximum(erase_union, m["mask"])
            roles.append(("erase", None))
            continue
        if m["weight"] <= SPECK_MAX_W:
            roles.append(("speck", None))
            continue
        ratios = [_inside_ratio(m, b) for b in boxes]
        i = int(np.argmax(ratios))
        if ratios[i] >= VOTE_RATIO:
            if m["weight"] < VOTE_MIN_W:
                raise Ambiguous("dấu trong ô quá nhạt")
            votes.add(i)
            roles.append(("vote", i))
        elif m["weight"] >= STRAY_MIN_W:
            strays += 1
            roles.append(("stray", round(1.0 - ratios[i], 4)))
        else:
            raise Ambiguous("dấu trung gian không xác định")
    struck_ink = [_ink_in(erase_union, b) / _area(b) for b in boxes]
    struck = {i for i, f in enumerate(struck_ink) if f >= STRIKE_INK}
    votes -= struck
    if len(struck) >= STRIKE_COUNT:
        label = "erased"
    elif strays:
        label = "wrong_mark"
    elif len(votes) > MAX_VOTES:
        label = "multi_mark"
    elif votes:
        label = "valid"
    else:
        label = "blank"
    return {"label": label, "votes": sorted(votes), "struck": sorted(struck), "strays": strays,
            "struck_ink": [round(f, 4) for f in struck_ink], "roles": roles}


# =============================================================================
# GHÉP ẢNH, NHIỄU SCAN, SINH MẪU
# =============================================================================
def _quota(n, ratios):
    raw = [n * r for r in ratios]
    base = [int(x) for x in raw]
    rem = n - sum(base)
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)[:rem]:
        base[i] += 1
    return base


def _add_background(img_rgb, rng):
    """Vết giấy mờ làm tối nhẹ. Áp cho mọi mẫu nên không là đặc trưng nhãn."""
    arr = np.asarray(img_rgb.convert("L"), dtype=np.float32)
    blob = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(blob)
    for _ in range(int(rng.integers(2, 5))):
        x, y, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(60, 160)
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    blob = np.asarray(blob.filter(ImageFilter.GaussianBlur(25)), dtype=np.float32) / 255.0
    return arr * (1.0 - 0.04 * blob)


def _compose(base_arr, marks):
    alpha = np.zeros((H, W), np.float32)
    for m in marks:
        alpha = np.maximum(alpha, m["mask"].astype(np.float32) / 255.0)
    return np.clip(base_arr * (1.0 - alpha), 0, 255).astype(np.uint8)


def _finish(arr_u8, boxes, rng):
    """Nhiễu scan: xoay ±3°, mờ, đổi sáng, nhiễu hạt. Trả về ảnh, ROI đã xoay, tham số."""
    img = Image.fromarray(arr_u8, mode="L")
    angle = float(rng.uniform(-3, 3))
    img = img.rotate(angle, expand=False, fillcolor=255)
    blur = 0.0
    if rng.random() < 0.5:
        blur = float(rng.uniform(0.3, 1.2))
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    bright = float(rng.uniform(0.75, 1.05))
    sd = float(rng.uniform(2, 8))
    a = np.asarray(img, dtype=np.float32) * bright
    a = np.clip(a + rng.normal(0, sd, a.shape), 0, 255).astype(np.uint8)
    lb, ez = layout_shapes(boxes)
    rs = [rotate_box(b, angle, W, H) for b in list(boxes) + [lb, ez]]
    params = {"rotation": round(angle, 3), "blur_level": round(blur, 3),
              "brightness": round(bright, 3), "noise_level": round(sd, 3)}
    return Image.fromarray(a, mode="L"), rs, params


def _make_sample(label, diff, anchor, base_arr, boxes, rng, stats):
    for _ in range(60):
        try:
            marks, case = BUILDERS[label](diff, anchor, boxes, rng)
            res = oracle(marks, boxes)
        except Ambiguous:
            stats["reject_ambiguous"] += 1
            continue
        if res["label"] == label:
            stats["accepted"] += 1
            return marks, case, res
        stats["reject_label_mismatch"] += 1
    raise RuntimeError(f"Không sinh được mẫu label={label} difficulty={diff} sau 60 lần thử. "
                       "Kiểm tra lại ngưỡng oracle và các bộ sinh.")


def _mark_metadata(marks, res, boxes):
    line_specs = [m["spec"] for m in marks if m["spec"]["render"] in ("line", "text")
                  and m["spec"]["hint"] != "erase"]
    erase_specs = [m["spec"] for m in marks if m["spec"]["hint"] == "erase"]
    dist = None
    for sp in line_specs:
        pts = [p for pl in sp.get("polylines", []) for p in pl] if sp["render"] == "line" else [sp["pos"]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
        for b in boxes:
            dx = max(b[0] - bx2, bx1 - b[2], 0.0)
            dy = max(b[1] - by2, by1 - b[3], 0.0)
            d = float(np.hypot(dx, dy))
            dist = d if dist is None else min(dist, d)
    vote_like = [s for s in line_specs if s["hint"] == "vote"]
    strays = [r[1] for r in res["roles"] if r[0] == "stray"]
    return {
        "mark_type": "|".join(sorted({s["shape"] for s in line_specs if s["shape"]})) or None,
        "mark_size": round(float(np.mean([s["s"] for s in vote_like])), 3) if vote_like else None,
        "mark_thickness": round(float(np.mean([s["width"] for s in line_specs if s["render"] == "line"])), 2)
        if any(s["render"] == "line" for s in line_specs) else None,
        "mark_opacity": round(float(np.mean([s["alpha"] for s in line_specs])), 3) if line_specs else None,
        "distance_to_border": round(dist, 2) if dist is not None else None,
        "outside_ratio": round(max(strays), 3) if strays else None,
        "erasure_strength": round(float(np.mean([s["alpha"] for s in erase_specs])), 3) if erase_specs else 0.0,
        "erasure_ratio": round(len(res["struck"]) / NC, 3),
        "n_votes": len(res["votes"]),
        "n_struck": len(res["struck"]),
        "n_strays": res["strays"],
    }


def _sha_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


CSV_COLS = ["file", "label", "split", "loai_loi", "so_o_danh_dau", "o_co_danh_dau", "group_id", "difficulty",
            "case_type", "mark_type", "mark_size", "mark_thickness", "mark_opacity", "distance_to_border",
            "outside_ratio", "erasure_strength", "erasure_ratio", "n_votes", "n_struck", "n_strays",
            "rotation", "blur_level", "brightness", "noise_level"]
# Cột KHÔNG được dùng làm đặc trưng cho model (suy từ nhãn hoặc từ quá trình sinh).
FORBIDDEN_FEATURES = ["label", "loai_loi", "so_o_danh_dau", "o_co_danh_dau", "difficulty", "case_type",
                      "group_id", "split", "n_votes", "n_struck", "n_strays", "erasure_ratio", "outside_ratio"]


# =============================================================================
# DRIVER
# =============================================================================
def generate(out_root=DEFAULT_OUT, samples_per_class=SAMPLES_PER_CLASS, seed=RANDOM_SEED):
    """Sinh bộ dữ liệu. Mặc định 5 × 500 = 2500 ảnh. Trả về đường dẫn all.csv."""
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    data_dir = os.path.join(out_root, "data")
    raw_dir = os.path.join(data_dir, "raw")
    lab_dir = os.path.join(data_dir, "labels")
    meta_dir = os.path.join(data_dir, "metadata")
    prev_dir = os.path.join(data_dir, "preview")
    for d in (raw_dir, lab_dir, meta_dir, prev_dir):
        os.makedirs(d, exist_ok=True)

    n_groups = samples_per_class
    plan = {c: [DIFFS[k] for k, q in enumerate(_quota(n_groups, DIFF_RATIO)) for _ in range(q)] for c in CLASSES}
    for c in CLASSES:
        rng.shuffle(plan[c])

    stats = collections.Counter()
    rows, jsonl, boxes_json = [], [], {}
    for g in range(n_groups):
        voter = random_voter()
        img, boxes = draw_ballot_template(CANDIDATE_NAMES, voter)
        base = _add_background(img, rng)
        anchor = int(rng.integers(NC))
        order = list(CLASSES)
        rng.shuffle(order)
        for label in order:
            diff = plan[label][g]
            marks, case, res = _make_sample(label, diff, anchor, base, boxes, rng, stats)
            arr = _compose(base, marks)
            out_img, rboxes, params = _finish(arr, boxes, rng)
            fname = f"ballot_{len(rows):05d}.png"
            out_img.save(os.path.join(raw_dir, fname), compress_level=1)
            boxes_json[fname] = {"boxes": rboxes[:NC], "left_band": rboxes[NC], "extra_zone": rboxes[NC + 1]}
            marked = sorted(set(res["votes"]) | set(res["struck"]))
            meta = _mark_metadata(marks, res, boxes)
            row = {"file": fname, "label": label, "split": "", "loai_loi": label,
                   "so_o_danh_dau": len(marked), "o_co_danh_dau": "|".join(map(str, marked)),
                   "group_id": f"g{g:04d}", "difficulty": diff, "case_type": case}
            row.update(meta)
            row.update(params)
            rows.append(row)
            with open(os.path.join(raw_dir, fname), "rb") as fh:
                img_sha = hashlib.sha256(fh.read()).hexdigest()
            jsonl.append({"file": fname, "group_id": f"g{g:04d}", "label": label, "difficulty": diff,
                          "case_type": case, "anchor": anchor, "template_boxes": boxes,
                          "marks": [m["spec"] for m in marks], "oracle": res,
                          "params": params, "meta": meta, "sha256": img_sha})
        if (g + 1) % 50 == 0:
            print(f"  đã sinh {g + 1}/{n_groups} group ({len(rows)} ảnh)")

    groups = [r["group_id"] for r in rows]
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
                          .split(np.zeros(len(rows)), groups=groups))
    split_of = {i: "train" for i in tr_idx}
    split_of.update({i: "test" for i in te_idx})
    for i, r in enumerate(rows):
        r["split"] = split_of[i]
        jsonl[i]["split"] = r["split"]

    _write_outputs(data_dir, rows, jsonl, boxes_json, stats, seed, samples_per_class)
    print(f"Xong: {len(rows)} ảnh → {raw_dir}")
    return os.path.join(lab_dir, "all.csv")


def _write_outputs(data_dir, rows, jsonl, boxes_json, stats, seed, spc):
    lab_dir = os.path.join(data_dir, "labels")
    meta_dir = os.path.join(data_dir, "metadata")
    with open(os.path.join(lab_dir, "all.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(lab_dir, "boxes.json"), "w", encoding="utf-8") as f:
        json.dump(boxes_json, f, ensure_ascii=False)
    with open(os.path.join(meta_dir, "samples.jsonl"), "w", encoding="utf-8") as f:
        for rec in jsonl:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(meta_dir, "feature_policy.json"), "w", encoding="utf-8") as f:
        json.dump({"forbidden_features": FORBIDDEN_FEATURES,
                   "note": "Chỉ dùng metadata để kiểm tra, phân tích và stratify; không đưa vào model."},
                  f, ensure_ascii=False, indent=1)
    rules = {"version": ORDER_VERSION, "seed": seed, "samples_per_class": spc,
             "VOTE_RATIO": VOTE_RATIO, "VOTE_MIN_W": VOTE_MIN_W, "STRAY_MIN_W": STRAY_MIN_W,
             "SPECK_MAX_W": SPECK_MAX_W, "STRIKE_INK": STRIKE_INK, "STRIKE_COUNT": STRIKE_COUNT,
             "generation_stats": dict(stats)}
    with open(os.path.join(meta_dir, "rules.json"), "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=1)
    _write_report(data_dir, rows)
    _write_previews(data_dir, rows)
    _run_validation(data_dir, rows, boxes_json)


def _write_report(data_dir, rows):
    lines = ["Dataset Report", "--------------", f"Total samples: {len(rows)}", ""]
    for c in CLASSES:
        lines.append(f"{c}:")
        for d in DIFFS:
            lines.append(f"    {d}: {sum(1 for r in rows if r['label'] == c and r['difficulty'] == d)}")
    lines += ["", "Split:"]
    for s in ("train", "test"):
        sub = [r for r in rows if r["split"] == s]
        lines.append(f"  {s}: {len(sub)} | " + ", ".join(
            f"{c}={sum(1 for r in sub if r['label'] == c)}" for c in CLASSES))
        lines.append("    difficulty: " + ", ".join(
            f"{d}={sum(1 for r in sub if r['difficulty'] == d)}" for d in DIFFS))
    lines += ["", "Hard Pair Coverage (số group mà cả hai lớp đều ở mức hard/boundary):"]
    groups = collections.defaultdict(dict)
    for r in rows:
        groups[r["group_id"]][r["label"]] = r["difficulty"]
    pairs = [("valid", "wrong_mark"), ("valid", "erased"), ("valid", "multi_mark"),
             ("blank", "wrong_mark"), ("erased", "wrong_mark"), ("erased", "valid"), ("multi_mark", "valid")]
    for a, b in pairs:
        n = sum(1 for g in groups.values() if g.get(a) in ("hard", "boundary") and g.get(b) in ("hard", "boundary"))
        lines.append(f"  {a} vs {b}: {n}")
    with open(os.path.join(data_dir, "metadata", "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _write_previews(data_dir, rows):
    prev = os.path.join(data_dir, "preview")
    raw = os.path.join(data_dir, "raw")
    crop = (BOX_X1 - 170, 330, 1120, 610)
    tw, th = 420, 124
    cols = [(d, k) for d in DIFFS for k in range(2)]
    sheet = Image.new("RGB", (len(cols) * tw, len(CLASSES) * (th + 24)), "white")
    d_draw = ImageDraw.Draw(sheet)
    for r, c in enumerate(CLASSES):
        for k, (d, rep) in enumerate(cols):
            cand = [x for x in rows if x["label"] == c and x["difficulty"] == d]
            if rep >= len(cand):
                continue
            im = Image.open(os.path.join(raw, cand[rep]["file"])).crop(crop).resize((tw - 6, th))
            y = r * (th + 24)
            sheet.paste(im, (k * tw + 3, y + 24))
            d_draw.text((k * tw + 4, y + 4), f"{c} | {d}", fill="red")
    sheet.save(os.path.join(prev, "contact_sheet_class_difficulty.png"))

    groups = collections.defaultdict(dict)
    for x in rows:
        groups[x["group_id"]][x["label"]] = x
    full = [g for g in groups.values() if len(g) == len(CLASSES)]
    full.sort(key=lambda g: -sum(v["difficulty"] in ("hard", "boundary") for v in g.values()))
    pick = full[:3]
    tw2, th2 = 300, 110
    sheet2 = Image.new("RGB", (len(CLASSES) * tw2, len(pick) * (th2 + 24)), "white")
    d2 = ImageDraw.Draw(sheet2)
    for r, g in enumerate(pick):
        for k, c in enumerate(CLASSES):
            x = g[c]
            im = Image.open(os.path.join(raw, x["file"])).crop(crop).resize((tw2 - 6, th2))
            y = r * (th2 + 24)
            sheet2.paste(im, (k * tw2 + 3, y + 24))
            d2.text((k * tw2 + 4, y + 4), f"{x['group_id']} {c} ({x['difficulty']})", fill="red")
    sheet2.save(os.path.join(prev, "group_comparison_hard.png"))


def _recheck_label(rec):
    """Dựng lại mask từ spec đã lưu và suy nhãn lại bằng oracle (không dùng nhãn đã lưu)."""
    marks = [_render(sp) for sp in rec["marks"]]
    return oracle(marks, rec["template_boxes"])["label"]


def _run_validation(data_dir, rows, boxes_json):
    fails, info = [], {}
    raw = os.path.join(data_dir, "raw")

    hashes = collections.Counter()
    for r in rows:
        p = os.path.join(raw, r["file"])
        if not os.path.exists(p):
            fails.append(f"thiếu ảnh {r['file']}")
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                if im.size != (W, H) or im.mode != "L":
                    fails.append(f"kích thước/mode sai {r['file']}: {im.size} {im.mode}")
        except Exception as e:  # noqa: BLE001
            fails.append(f"ảnh hỏng {r['file']}: {e}")
        with open(p, "rb") as f:
            hashes[hashlib.sha256(f.read()).hexdigest()] += 1
    dup_img = sum(v - 1 for v in hashes.values() if v > 1)
    if dup_img:
        fails.append(f"trùng ảnh: {dup_img}")

    labels = {r["label"] for r in rows}
    if labels != set(CLASSES):
        fails.append(f"thiếu lớp: {set(CLASSES) - labels}")
    cnt = collections.Counter(r["label"] for r in rows)
    info["class_counts"] = dict(cnt)
    if len(set(cnt.values())) > 1:
        fails.append(f"lớp không cân bằng: {dict(cnt)}")
    splits = {r["split"] for r in rows}
    if splits != {"train", "test"}:
        fails.append(f"split thiếu: {splits}")
    g_tr = {r["group_id"] for r in rows if r["split"] == "train"}
    g_te = {r["group_id"] for r in rows if r["split"] == "test"}
    if g_tr & g_te:
        fails.append(f"rò rỉ group giữa train và test: {len(g_tr & g_te)}")
    info["group_leakage"] = len(g_tr & g_te)
    gl = collections.defaultdict(list)
    for r in rows:
        gl[r["group_id"]].append(r["label"])
    if any(sorted(v) != sorted(CLASSES) for v in gl.values()):
        fails.append("có group không đủ đúng 1 mẫu mỗi lớp")

    for fname, roi in boxes_json.items():
        for key in ("boxes", "left_band", "extra_zone"):
            items = roi[key] if key == "boxes" else [roi[key]]
            for b in items:
                if not (0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H):
                    fails.append(f"ROI không hợp lệ {fname} {key} {b}")
                    break
        if len(roi["boxes"]) != NC:
            fails.append(f"số ô ROI sai {fname}")

    mismatch = 0
    with open(os.path.join(data_dir, "metadata", "samples.jsonl"), encoding="utf-8") as f:
        jsonl = [json.loads(line) for line in f]
    for rec in jsonl:
        if _recheck_label(rec) != rec["label"]:
            mismatch += 1
    info["label_mismatch_on_recheck"] = mismatch
    if mismatch:
        fails.append(f"nhãn không khớp hình học: {mismatch}")

    meta_hash = collections.Counter(_sha_json({k: v for k, v in rec.items() if k != "file"}) for rec in jsonl)
    info["duplicate_metadata"] = sum(v - 1 for v in meta_hash.values() if v > 1)
    if info["duplicate_metadata"]:
        fails.append("trùng metadata")

    dd = collections.Counter((r["label"], r["difficulty"]) for r in rows)
    info["difficulty_counts"] = {f"{c}/{d}": dd[(c, d)] for c in CLASSES for d in DIFFS}
    info["fail"] = fails
    with open(os.path.join(data_dir, "metadata", "validation.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print("VALIDATION:", "PASS" if not fails else "FAIL")
    if fails:
        for x in fails[:20]:
            print("  -", x)
        raise RuntimeError(f"Validation thất bại: {len(fails)} lỗi (xem data/metadata/validation.json)")


# =============================================================================
# CLI
# =============================================================================
def _cli():
    p = argparse.ArgumentParser(description="Sinh 2500 ảnh thẻ cử tri giả lập (mặc định).")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="thư mục gốc đầu ra; ảnh ghi vào <out>/data (mặc định: modules/ballot-validation-ai)")
    p.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS,
                   help="số mẫu mỗi lớp (mặc định 500 → 2500 ảnh)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = p.parse_args()
    generate(out_root=a.out, samples_per_class=a.samples_per_class, seed=a.seed)


if __name__ == "__main__":
    _cli()