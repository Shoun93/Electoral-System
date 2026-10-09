# Phân loại phiếu bầu hợp lệ: mô hình lai CNN + C4.5, ghi kết quả lên hash-chain

Pipeline chạy được từ đầu đến cuối trên dữ liệu **giả lập**:
sinh phiếu → tiền xử lý → CNN phát hiện dấu trên từng ô → C4.5 ra quyết định (có luật IF-THEN)
→ đánh giá trên tập test → ghi kết quả từng phiếu lên chuỗi khối.

> Trạng thái: **demo nghiên cứu**. Mã Hyperledger Fabric đã viết nhưng chưa chạy được ở đây; PBFT chưa làm; MLOps tối thiểu (xem mục Giới hạn).

## 1. Kiến trúc

```
Ảnh phiếu (scan)
   │  binarize (CLAHE + Otsu) ở độ phân giải gốc
   ├──► 8 đặc trưng thủ công (đếm ô có dấu, mật độ ô, dải trái ô, vùng ghi thêm, ...)
   └──► 5 crop ô (48×48) ──► CNN phát hiện dấu (out-of-fold cho tập train)
                                   │  3 thống kê: số ô có dấu, xác suất TB, xác suất max
   ┌───────────────────────────────┘
   ▼
C4.5 (cài đặt tại src/c45.py: gain ratio, tách ngưỡng, cắt tỉa pessimistic)
   ▼
Nhãn: valid | blank | erased | wrong_mark | multi_mark   + luật IF-THEN giải thích được
   ▼
Hash-chain: mỗi phiếu = 1 block (nhãn, độ tin cậy, SHA-256 của ảnh gốc)
```

CNN toàn ảnh 224×224 vẫn được huấn luyện như **đối chứng**. Xem kết quả bên dưới.

## 2. Cấu trúc thư mục

```
ballot-validation-ai/
├── main.py                 # pipeline end-to-end
├── configs/config.yaml     # mọi siêu tham số và đường dẫn
├── requirements.txt
├── src/
│   ├── config.py           # đọc config (đường dẫn tính theo vị trí file)
│   ├── preprocess.py       # nhị phân hóa, crop ô, tạo models/dataset.npz
│   ├── feature_c45.py      # 8 đặc trưng thủ công
│   ├── c45.py              # cài đặt C4.5 (numpy)
│   ├── train_c45.py        # huấn luyện C4.5, in luật
│   ├── train_cnn.py        # CNN toàn ảnh (đối chứng) + CNN phát hiện dấu theo ô
│   ├── baselines.py        # majority, Naive Bayes, kNN
│   ├── evaluate.py         # báo cáo 5 lớp, nhị phân hợp lệ/không hợp lệ, ma trận nhầm lẫn
│   ├── hashchain.py        # chuỗi khối giáo dục (SHA-256 + PoW nhẹ), lưu/xác minh
│   └── monitor.py          # PSI, log lịch sử, cảnh báo suy giảm
└── tests/test_units.py     # kiểm thử đơn vị
../../dataset/generate_dataset.py   # sinh phiếu giả lập (nguồn dữ liệu duy nhất)
```

## 3. Cài đặt và chạy

```bash
cd modules/ballot-validation-ai
pip install -r requirements.txt
python main.py                       # tự sinh 600 phiếu nếu chưa có, rồi huấn luyện và đánh giá
python -m unittest discover -s tests -v
```

Thời gian chạy đầy đủ trên CPU 2 nhân khoảng 20–25 phút (phần lớn là CNN toàn ảnh).

## 4. Dữ liệu

`dataset/generate_dataset.py` sinh 600 phiếu giả lập, 120 mỗi lớp:

| Nhãn | Ý nghĩa |
|---|---|
| `valid` | chọn 1–3 đại biểu |
| `blank` | phiếu trắng |
| `erased` | gạch xóa toàn bộ danh sách |
| `wrong_mark` | dấu đặt sai vị trí (ngoài ô), hoặc ghi thêm tên ngoài danh sách |
| `multi_mark` | chọn 4–5 đại biểu (nhiều hơn quy định) |

Chia 80/20 phân tầng theo nhãn; 10% tổng số làm validation (lấy từ phần train). Mỗi ảnh có kèm tọa độ 5 ô đánh dấu đã áp phép xoay nhiễu (`data/labels/boxes.json`) và danh sách ô thực sự có dấu (`o_co_danh_dau`), dùng để kiểm chứng nhãn ô.

Toàn bộ thông tin cá nhân trên thẻ là **giả lập**.

## 5. Kết quả (tập test, 120 phiếu, một lần chạy, seed 42)

| Mô hình | Accuracy | Macro-F1 | Recall "không hợp lệ" | Precision "không hợp lệ" |
|---|---|---|---|---|
| Baseline majority | 0.200 | 0.067 | 0.000 | 0.000 |
| Baseline Naive Bayes | 0.817 | 0.768 | 0.917 | 0.989 |
| Baseline kNN (k=5) | 0.983 | 0.983 | 1.000 | 0.990 |
| C4.5 (đặc trưng thủ công) | 0.992 | 0.992 | 1.000 | 1.000 |
| **Mô hình lai (CNN theo ô + C4.5)** | **0.992** | **0.992** | **1.000** | **1.000** |
| CNN toàn ảnh 224×224 (đối chứng) | 0.400 | 0.280 | 1.000 | 0.800 |

- CNN phát hiện dấu trên từng ô đạt **100%** accuracy trên tập test (600 ô).
- CNN toàn ảnh **không học được**: độ chính xác trên validation không vượt quá 40% sau 15 epoch. Nguyên nhân: nét dấu 3 px bị mờ khi thu về 224×224, và chỉ có 480 mẫu. Đây là lý do nhánh CNN được chuyển sang phát hiện dấu theo ô.

**Cần đọc kết quả này một cách thận trọng:** trên dữ liệu giả lập, 8 đặc trưng thủ công đã gần như tách hoàn toàn các lớp, nên C4.5 không cần CNN và mô hình lai không cho thấy lợi ích rõ ràng. Lợi ích của CNN chỉ có thể kiểm chứng trên ảnh scan thật, với nhiễu, bóng, tẩy xóa và chữ ký không đồng đều. Đây là việc cần làm tiếp.

Luật C4.5 học được (một phần):
```
IF count_ink_boxes ≤ 0.5 AND left_band_ink ≤ 0.011 AND extra_zone_ink ≤ 0.022 THEN blank
IF count_ink_boxes > 0.5 AND count_ink_boxes ≤ 3.5                            THEN valid
IF count_ink_boxes > 3.5 AND max_box_density > 0.401                          THEN erased
IF count_ink_boxes > 3.5 AND max_box_density ≤ 0.401                          THEN multi_mark
```

## 6. Blockchain và MLOps

- `results/hashchain.json`: 1 block neo mã băm file mô hình và manifest dữ liệu, rồi 120 block phiếu. Kiểm chứng: sửa một block trong bản sao thì `verify()` trả về `False`.
- `results/metrics_log.json` + `check_drift()`: so sánh độ chính xác lần chạy mới với lịch sử, cảnh báo nếu giảm quá 5 điểm.
- `monitor.psi()`: đo lệch phân phối đặc trưng giữa train và dữ liệu sản xuất (chưa có dữ liệu sản xuất để chạy thật).

## 7. Giới hạn

- **Dữ liệu giả lập**, dễ tách hơn dữ liệu thật rất nhiều (xem mục 5).
- **Blockchain cục bộ** (`src/hashchain.py`) là chuỗi khối một node để kiểm thử logic toàn vẹn. Mã Hyperledger Fabric nằm ở `modules/blockchain-ledger/`, đã viết nhưng **chưa chạy được** trong môi trường phát triển (xem README của module đó). Gốc Merkle của lô phiếu được xuất ra `results/fabric_batch.json` để chaincode dùng.
- **MLOps** mới ở mức tối thiểu (log lịch sử, PSI). Chưa có DVC, MLflow, CI/CD, Docker.
- **Chưa có web demo** và chưa đo TPS, độ trễ, khả năng chịu lỗi.
- **Dữ liệu chưa có phiếu trắng thật, chọn sai thật**: các nhãn được sinh bằng quy tắc vẽ, không phải từ người gán nhãn.
