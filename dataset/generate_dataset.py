"""
Sinh bộ dữ liệu thẻ/phiếu bầu GIẢ LẬP theo bố cục "Thẻ cử tri" (Luật 85/2015/QH13).

Đầu ra (mặc định: ../modules/ballot-validation-ai/):
  data/raw/ballot_XXXX.png        ảnh phiếu
  data/labels/all.csv             file, label, split (train/test), loai_loi, so_o_danh_dau,
                                   o_co_danh_dau (chỉ số các ô CÓ dấu bên trong ô, ví dụ "0|2")
  data/labels/boxes.json          theo từng ảnh: "boxes" (5 ô đánh dấu), "left_band" (dải bên
                                   trái ô), "extra_zone" (vùng ghi thêm tên); pixel ảnh GỐC,
                                   đã áp đúng phép xoay nhiễu → dùng để trích đặc trưng C4.5

5 nhãn (khớp với config.yaml của pipeline):
  valid       : chọn 1–3 đại biểu (đúng quy định không quá số lượng)
  blank       : phiếu trắng, không đánh dấu nào
  erased      : gạch xóa toàn bộ danh sách (tẩy xóa)
  wrong_mark  : đánh dấu sai — dấu nằm ngoài ô, hoặc ghi thêm tên ngoài danh sách
  multi_mark  : chọn nhiều hơn số lượng cho phép (4–5 đại biểu)

Mỗi nhãn có cùng số mẫu; chia train/test 80/20 phân tầng theo nhãn.
TOÀN BỘ DỮ LIỆU CÁ NHÂN TRÊN THẺ LÀ GIẢ LẬP.
"""

import os
import csv
import json
import math
import random
import datetime
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.model_selection import train_test_split

# ----------------------------- CẤU HÌNH -----------------------------
RANDOM_SEED = 42
N_PER_CLASS = 400                    # 5 nhãn × 400 = 2000 ảnh
CLASSES = ["valid", "blank", "erased", "wrong_mark", "multi_mark"]
MAX_VOTES = 3                        # chọn tối đa 3 trong 5 đại biểu
MIN_VOTES_VALID = 1                  # phiếu hợp lệ: chọn từ 1 đến MAX_VOTES
IMG_W, IMG_H = 1200, 850
TEST_SIZE = 0.2

CANDIDATE_NAMES = [
    "Nguyễn Văn A",
    "Trần Văn B",
    "Lê Thị C",
    "Lý Thị D",
    "Nguyễn Thị E",
]
N_CANDIDATES = len(CANDIDATE_NAMES)
if not (1 <= MIN_VOTES_VALID <= MAX_VOTES < N_CANDIDATES <= 6):
    raise ValueError("Cần 1 <= MIN_VOTES_VALID <= MAX_VOTES < N_CANDIDATES <= 6.")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "modules", "ballot-validation-ai"))

# ----------------------------- DỮ LIỆU GIẢ -----------------------------
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


# ------------------------------ FONT ------------------------------
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


# --------------------------- BỐ CỤC THẺ ---------------------------
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
    """Vẽ thẻ cử tri. Trả về (ảnh, danh sách 5 box ô đánh dấu ở hệ tọa độ ảnh gốc)."""
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


def mark_box(draw, box, style="tick"):
    x1, y1, x2, y2 = box
    if style == "tick":
        draw.line([x1 + 4, (y1 + y2) // 2, (x1 + x2) // 2, y2 - 4], fill="black", width=3)
        draw.line([(x1 + x2) // 2, y2 - 4, x2 - 3, y1 + 3], fill="black", width=3)
    elif style == "cross":
        draw.line([x1 + 3, y1 + 3, x2 - 3, y2 - 3], fill="black", width=3)
        draw.line([x1 + 3, y2 - 3, x2 - 3, y1 + 3], fill="black", width=3)


def rotate_box(box, angle_deg, w, h):
    """Áp đúng phép xoay của PIL Image.rotate(angle, expand=False) lên một box (đã kiểm chứng)."""
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


def apply_realistic_noise(img, shapes):
    """Mô phỏng ảnh scan: xoay nhẹ, mờ, đổi sáng, nhiễu hạt.
    shapes: danh sách hình chữ nhật (x1,y1,x2,y2) ở hệ tọa độ ảnh gốc.
    Trả về (ảnh, danh sách hình đã áp cùng phép xoay)."""
    img = img.convert("L")                     # ảnh scan xám (giảm khối lượng tính toán)
    angle = random.uniform(-3, 3)
    img = img.rotate(angle, expand=False, fillcolor=255)
    rshapes = [rotate_box(b, angle, IMG_W, IMG_H) for b in shapes]

    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

    arr = np.asarray(img, dtype=np.float32)
    arr = arr * random.uniform(0.75, 1.05)
    noise = np.random.normal(0, random.uniform(2, 8), arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L"), rshapes


def layout_shapes(boxes):
    """Hai vùng cố định trên thẻ (ảnh gốc): dải bên trái ô (dấu sai vị trí) và vùng ghi thêm tên."""
    top, bottom = boxes[0][1] - 6, boxes[-1][3] + 6
    left_band = (BOX_X1 - 70, top, BOX_X1 - 24, bottom)
    extra_zone = (NAME_X, boxes[-1][3] + 25, IMG_W - 40, boxes[-1][3] + 60)
    return left_band, extra_zone


def make_sample(label, rng_candidates):
    """Sinh 1 phiếu với nhãn cho trước. Trả về (ảnh PIL, loai_loi, số ô đánh dấu, box gốc)."""
    candidates = list(rng_candidates)
    voter = random_voter()
    img, boxes = draw_ballot_template(candidates, voter)
    draw = ImageDraw.Draw(img)
    loai_loi = label
    n_marks = 0
    marked = []                                # chỉ số các ô CÓ dấu bên trong ô

    if label == "valid":
        n_marks = random.randint(MIN_VOTES_VALID, MAX_VOTES)
        marked = sorted(random.sample(range(N_CANDIDATES), n_marks))
        for c in marked:
            mark_box(draw, boxes[c], "tick")

    elif label == "blank":
        pass                                   # phiếu trắng

    elif label == "erased":                    # tẩy xóa: gạch chéo toàn bộ danh sách
        n_marks = N_CANDIDATES
        marked = list(range(N_CANDIDATES))
        for c in range(N_CANDIDATES):
            mark_box(draw, boxes[c], "cross")
        draw.line([BOX_X1 - 20, boxes[0][1] - 10, IMG_W - 40, boxes[-1][3] + 10], fill="black", width=4)

    elif label == "wrong_mark":                # chọn sai: dấu ngoài ô, hoặc ghi thêm tên
        if random.random() < 0.5:
            c = random.randrange(N_CANDIDATES)
            x1, y1, x2, y2 = boxes[c]
            off_box = (x1 - 60, y1 - 4, x1 - 24, y2 + 4)   # dấu nằm bên trái ô, sai vị trí
            mark_box(draw, off_box, "tick")
        else:
            draw.text((NAME_X, boxes[-1][3] + 40), f"+ {random_outside_name()}",
                      font=get_font(24), fill="black", anchor="lm")

    elif label == "multi_mark":                # chọn nhiều hơn quy định
        n_marks = random.randint(MAX_VOTES + 1, N_CANDIDATES)
        marked = sorted(random.sample(range(N_CANDIDATES), n_marks))
        for c in marked:
            mark_box(draw, boxes[c], "tick")

    return img, loai_loi, n_marks, boxes, marked


def generate(out_root=DEFAULT_OUT, n_per_class=N_PER_CLASS):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    raw_dir = os.path.join(out_root, "data", "raw")
    lab_dir = os.path.join(out_root, "data", "labels")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    plan = CLASSES * n_per_class
    random.shuffle(plan)
    rows, boxes_json = [], {}
    for i, label in enumerate(plan):
        img, loai_loi, n_marks, boxes, marked = make_sample(label, CANDIDATE_NAMES)
        left_band, extra_zone = layout_shapes(boxes)
        img, rs = apply_realistic_noise(img, boxes + [left_band, extra_zone])
        fname = f"ballot_{i:04d}.png"
        img.save(os.path.join(raw_dir, fname), compress_level=1)
        boxes_json[fname] = {"boxes": rs[:N_CANDIDATES], "left_band": rs[N_CANDIDATES],
                             "extra_zone": rs[N_CANDIDATES + 1]}
        rows.append({"file": fname, "label": label, "loai_loi": loai_loi, "so_o_danh_dau": n_marks,
                     "o_co_danh_dau": "|".join(str(c) for c in marked)})

    files = [r["file"] for r in rows]
    labels = [r["label"] for r in rows]
    train_f, test_f = train_test_split(files, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels)
    train_set = set(train_f)
    for r in rows:
        r["split"] = "train" if r["file"] in train_set else "test"

    with open(os.path.join(lab_dir, "all.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "split", "loai_loi", "so_o_danh_dau", "o_co_danh_dau"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(lab_dir, "boxes.json"), "w", encoding="utf-8") as f:
        json.dump(boxes_json, f)

    print(f"Đã sinh {len(rows)} ảnh -> {raw_dir}")
    print(f"Train={len(train_f)} Test={len(test_f)} | nhãn: " +
          ", ".join(f"{c}={labels.count(c)}" for c in CLASSES))
    return os.path.join(lab_dir, "all.csv")


if __name__ == "__main__":
    generate()
