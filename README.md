# Electoral System

Hệ thống kiểm phiếu bầu cử có trí tuệ nhân tạo, kết hợp sổ cái blockchain (Hyperledger Fabric) để lưu kết quả có thể kiểm chứng.

> Trạng thái: **demo nghiên cứu**. Module kiểm phiếu bằng AI chạy được end-to-end trên dữ liệu giả lập. Mã blockchain đã viết nhưng chưa được triển khai đầy đủ trong repo này. PBFT chưa làm.

## Các module

| Module | Thư mục | Trạng thái | Mô tả |
|---|---|---|---|
| Kiểm phiếu bằng AI | [`modules/ballot-validation-ai`](modules/ballot-validation-ai) | ✅ Chạy end-to-end (dữ liệu giả lập) | Phân loại phiếu bằng mô hình lai CNN theo ô + C4.5 (có luật IF-THEN), ghi kết quả lên hash-chain |
| Sổ cái blockchain | [`modules/blockchain-ledger`](modules/blockchain-ledger) | ⚠️ Chaincode đã merge vào `main`, chưa hoàn chỉnh cho CCaaS | Hyperledger Fabric 3.1.5, 3 tổ chức, endorsement 2-trong-3, private data, gốc Merkle theo lô |
| Đồng thuận PBFT | — | ❌ Chưa làm | SmartBFT cần 4 orderer và cấu hình `bft-config/` |
| Đăng ký/xác thực cử tri | — | 🔜 Kế hoạch | Xác thực danh tính cử tri trước khi bỏ phiếu |
| Tổng hợp kết quả | — | 🔜 Kế hoạch | Tổng hợp từ nhiều trạm kiểm phiếu, đối chiếu với sổ cái |
| Giao diện web | — | 🔜 Kế hoạch | Tra cứu kết quả công khai |

Dữ liệu giả lập dùng chung: [`dataset/generate_dataset.py`](dataset/generate_dataset.py). Toàn bộ thông tin cá nhân trên thẻ là **giả lập**.

## Cấu trúc thư mục

```
Electoral-System/
├── README.md
├── Dockerfile                    # container cho pipeline Python (build từ gốc repo)
├── .dockerignore
├── dataset/
│   └── generate_dataset.py       # sinh phiếu giả lập (nguồn dữ liệu duy nhất)
├── docs/
│   └── stride.md                 # phân tích STRIDE tầng sổ cái
└── modules/
    ├── ballot-validation-ai/     # pipeline AI + hash-chain cục bộ + MLOps tối thiểu
    └── blockchain-ledger/
        ├── chaincode/ballot/     # chaincode Go: main.go, merkle/, collections.json, Dockerfile
        └── network/deploy.sh     # script triển khai chaincode lên test-network
```

## 1. Module kiểm phiếu bằng AI

Pipeline: sinh phiếu → tiền xử lý → CNN phát hiện dấu trên từng ô → C4.5 ra quyết định → đánh giá → ghi lên hash-chain.

```
Ảnh phiếu (scan)
   │  nhị phân hóa (CLAHE + Otsu) ở độ phân giải gốc
   ├──► 8 đặc trưng thủ công (đếm ô có dấu, mật độ ô, dải trái ô, vùng ghi thêm, ...)
   └──► 5 crop ô (48×48) ──► CNN phát hiện dấu (out-of-fold cho tập train)
                                   │  3 thống kê: số ô có dấu, xác suất TB, xác suất max
   ┌───────────────────────────────┘
   ▼
C4.5 (src/c45.py: gain ratio, tách ngưỡng, cắt tỉa pessimistic)
   ▼
Nhãn: valid | blank | erased | wrong_mark | multi_mark   + luật IF-THEN
   ▼
Hash-chain: mỗi phiếu = 1 block (nhãn, độ tin cậy, SHA-256 ảnh gốc)
```

CNN toàn ảnh 224×224 được huấn luyện như đối chứng.

### Cấu trúc module

```
modules/ballot-validation-ai/
├── main.py                 # pipeline end-to-end
├── configs/config.yaml     # mọi siêu tham số và đường dẫn
├── requirements.txt
├── src/
│   ├── config.py           # đọc config (đường dẫn tính theo vị trí file)
│   ├── preprocess.py       # nhị phân hóa, crop ô, tạo models/dataset.npz
│   ├── feature_c45.py      # 8 đặc trưng thủ công
│   ├── c45.py              # cài đặt C4.5 (numpy)
│   ├── train_c45.py        # huấn luyện C4.5, in luật
│   ├── train_cnn.py        # CNN toàn ảnh (đối chứng) và CNN phát hiện dấu theo ô
│   ├── baselines.py        # majority, Naive Bayes, kNN
│   ├── evaluate.py         # báo cáo 5 lớp, nhị phân hợp lệ/không hợp lệ, ma trận nhầm lẫn
│   ├── hashchain.py        # chuỗi khối cục bộ (SHA-256 + PoW nhẹ), lưu và xác minh
│   ├── merkle.py           # gốc Merkle (cùng vector kiểm thử với Go)
│   └── monitor.py          # PSI, log lịch sử, cảnh báo suy giảm
└── tests/test_units.py
```

### Cài đặt và chạy

```bash
cd modules/ballot-validation-ai
pip install -r requirements.txt
python main.py                              # tự sinh phiếu nếu chưa có, huấn luyện, đánh giá, ghi kết quả
python -m unittest discover -s tests -v     # kiểm thử đơn vị
```

Đầu ra trong `results/`: `metrics.json`, `predictions.csv`, `hashchain.json`, `fabric_batch.json`, `metrics_log.json`. Thư mục `models/` chứa mô hình đã huấn luyện. Hai thư mục này không được commit (xem `.gitignore`).

### Dữ liệu

`dataset/generate_dataset.py` sinh **2000 phiếu** (`N_PER_CLASS = 400`, 5 nhãn):

| Nhãn | Ý nghĩa |
|---|---|
| `valid` | chọn 1–3 đại biểu |
| `blank` | phiếu trắng |
| `erased` | gạch xóa toàn bộ danh sách |
| `wrong_mark` | dấu đặt ngoài ô, hoặc ghi thêm tên ngoài danh sách |
| `multi_mark` | chọn 4–5 đại biểu (nhiều hơn quy định) |

Chia 80/20 phân tầng theo nhãn. Validation lấy 12.5% từ tập train (tương đương 10% tổng). Mỗi ảnh có tọa độ 5 ô đã áp phép xoay nhiễu (`data/labels/boxes.json`) và danh sách ô có dấu (`o_co_danh_dau`) để kiểm chứng nhãn.

### Kết quả

> ⚠️ Bảng dưới đây là kết quả của lần chạy trước với **600 phiếu** (120/lớp), seed 42. Sau khi tăng lên 2000 phiếu cần chạy lại `python main.py` và cập nhật bảng.

| Mô hình | Accuracy | Macro-F1 | Recall "không hợp lệ" | Precision "không hợp lệ" |
|---|---|---|---|---|
| Baseline majority | 0.200 | 0.067 | 0.000 | 0.000 |
| Baseline Naive Bayes | 0.817 | 0.768 | 0.917 | 0.989 |
| Baseline kNN (k=5) | 0.983 | 0.983 | 1.000 | 0.990 |
| C4.5 (đặc trưng thủ công) | 0.992 | 0.992 | 1.000 | 1.000 |
| **Mô hình lai (CNN theo ô + C4.5)** | **0.992** | **0.992** | **1.000** | **1.000** |
| CNN toàn ảnh 224×224 (đối chứng) | 0.400 | 0.280 | 1.000 | 0.800 |

- CNN phát hiện dấu theo ô đạt 100% accuracy trên tập test (600 ô của lần chạy đó).
- CNN toàn ảnh không học được trong 15 epoch: nét dấu 3 px bị mờ khi thu về 224×224 và số mẫu nhỏ. Vì vậy nhánh CNN chuyển sang phát hiện dấu theo ô.

**Đọc kết quả thận trọng:** trên dữ liệu giả lập, 8 đặc trưng thủ công gần như tách hoàn toàn các lớp, nên C4.5 đơn thuần đã đạt kết quả gần tối đa. Lợi ích của CNN chỉ kiểm chứng được trên ảnh scan thật (nhiễu, bóng, tẩy xóa, chữ ký không đều).

Luật C4.5 học được (một phần, từ lần chạy trước):

```
IF count_ink_boxes ≤ 0.5 AND left_band_ink ≤ 0.011 AND extra_zone_ink ≤ 0.022 THEN blank
IF count_ink_boxes > 0.5 AND count_ink_boxes ≤ 3.5                            THEN valid
IF count_ink_boxes > 3.5 AND max_box_density > 0.401                          THEN erased
IF count_ink_boxes > 3.5 AND max_box_density ≤ 0.401                          THEN multi_mark
```

### MLOps tối thiểu

- `results/metrics_log.json` và `check_drift()`: so sánh độ chính xác lần chạy mới với lịch sử, cảnh báo nếu giảm hơn 5 điểm.
- `monitor.psi()`: đo lệch phân phối đặc trưng giữa train và dữ liệu sản xuất. Chưa có dữ liệu sản xuất để chạy thật.
- `hashchain.json`: 1 block neo mã băm mô hình và manifest dữ liệu, rồi một block cho mỗi phiếu. `verify()` trả về `False` khi một block bị sửa.

## 2. Sổ cái blockchain (Hyperledger Fabric)

Chi tiết và trạng thái từng thành phần: [`modules/blockchain-ledger/README.md`](modules/blockchain-ledger/README.md).

Tóm tắt thiết kế:

- **Tổ chức (MSP):** `Org1MSP` (ban tổ chức, đăng ký mô hình), `Org2MSP` (tổ kiểm phiếu, ghi kết quả), `Org3MSP` (giám sát, chỉ đọc). Chaincode kiểm tra MSP người gọi bằng `requireMSP`.
- **Endorsement:** `OutOf(2, Org1MSP.peer, Org2MSP.peer, Org3MSP.peer)`.
- **Ghi một lần:** `putNew` từ chối ghi đè khóa đã có.
- **Private data:** collection `CountingBoardPrivate` (chỉ Org2 và Org3 đọc/ghi). Sổ cái công khai chỉ có mã băm SHA-256.
- **Merkle theo lô:** `CommitBatch` ghi gốc Merkle, `VerifyBatch` tính lại từ danh sách lá và so sánh.
- **Nền tảng:** Hyperledger Fabric 3.1.5, kênh `electionchannel`, đồng thuận Raft hiện tại. PBFT (SmartBFT) chưa làm.

### Trạng thái

| Thành phần | Trạng thái |
|---|---|
| Cây Merkle Go (`chaincode/ballot/merkle/`) | ✅ `go test ./merkle/` |
| Cây Merkle Python (`src/merkle.py`) | ✅ cùng vector với Go |
| Chaincode (`main.go`) | ⚠️ `main` dùng `contractapi` với `cc.Start()` (peer build). Chưa có `shim.ChaincodeServer` cần cho CCaaS |
| Dockerfile chaincode | ⚠️ Build binary `/cc` nhưng vẫn chạy `cc.Start()`. Chưa phục vụ CCaaS. Cần thêm `ChaincodeServer` |
| Triển khai trên test-network | ⚠️ `deploy.sh` còn theo quy trình cũ (`package --lang golang`). Quy trình CCaaS dùng `network.sh deployCCAAS` |
| Kiểm thử trên mạng 3 tổ chức | ⚠️ Các luồng ghi, đọc, endorsement, private data và VerifyBatch đã chạy trên máy phát triển với bản CCaaS chưa commit. Chưa có kiểm thử tự động |
| Phân tích STRIDE | ✅ `docs/stride.md`. Còn 2 kịch bản chưa kiểm thử (sửa private data, DoS) |
| PBFT (SmartBFT) | ❌ Chưa làm |
| Kịch bản nút độc hại, DoS, đo TPS và độ trễ | ❌ Chưa làm |

### Chạy mạng Fabric (test-network)

```bash
# Gốc repo: sinh lô phiếu cho chaincode
cd modules/ballot-validation-ai && python main.py && cd -

# Mạng 2 tổ chức rồi thêm Org3
curl -sSLO https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh
bash install-fabric.sh docker samples binary
cd fabric-samples/test-network
./network.sh up createChannel -c electionchannel -ca
./addOrg3/addOrg3.sh up -c electionchannel -ca

# Triển khai chaincode (script trên main còn cũ; xem mục Trạng thái)
cd /path/to/Electoral-System
FABRIC_SAMPLES=$HOME/fabric-samples modules/blockchain-ledger/network/deploy.sh
```

`deploy.sh` giả định cấu trúc `organizations/` của fabric-samples v3.1.5. Không commit thư mục `organizations/`, khóa hay MSP.

Bản CCaaS đã chạy trên máy phát triển dùng `./network.sh deployCCAAS` với `connection.json`, `metadata.json` (type `ccaas`) và container chaincode trên mạng `fabric_test`. Quy trình này chưa có trong repo.

## 3. Chạy bằng Docker

Container này chỉ chạy pipeline Python:

```bash
docker build -t ballot-ai .
mkdir -p results models
docker run --rm -v "$PWD/results:/app/modules/ballot-validation-ai/results" \
                -v "$PWD/models:/app/modules/ballot-validation-ai/models" ballot-ai
```

## 4. Giới hạn

- **Dữ liệu giả lập**, dễ tách hơn dữ liệu thật nhiều. Nhãn sinh bằng quy tắc vẽ, không do người gán nhãn.
- **Blockchain cục bộ** (`src/hashchain.py`) là chuỗi khối một node, chỉ để kiểm thử tính toàn vẹn logic.
- **Raft** không chịu được nút độc hại. Chỉ PBFT mới có khả năng đó, và PBFT chưa làm.
- **`ballot_id`** là tên file ảnh; `pred` và `conf` lưu công khai trên sổ cái.
- **MLOps** mới ở mức tối thiểu. Chưa có DVC, MLflow, CI/CD.
- Chưa có web demo, chưa đo TPS, độ trễ, khả năng chịu lỗi.
- `torch` cần kiểm tra wheel cho phiên bản Python đang dùng.

## Tài liệu

- `docs/stride.md`: phân tích STRIDE tầng sổ cái.
- `modules/blockchain-ledger/README.md`: chi tiết blockchain.
- `modules/ballot-validation-ai/README.md`: chi tiết pipeline AI.
