# Electoral System

Hệ thống kiểm phiếu bầu cử có trí tuệ nhân tạo, kết hợp sổ cái blockchain (Hyperledger Fabric) để lưu kết quả có thể kiểm chứng.

> Trạng thái: **demo nghiên cứu, chưa đạt nghiệm thu**. Module AI chạy end-to-end trên dữ liệu giả lập. Phần chuyển `PENDING_REVIEW` (nhánh `feat/ba-giai-doan-1-selective`) **chưa đạt** ràng buộc Selective Risk ≤ 0.02. Mã blockchain đã viết nhưng chưa chạy trên Fabric. PBFT chưa làm. Kiểm toán và kế hoạch: [`docs/audit`](docs/audit).

> **Phase 1:** các sửa đổi audit/safeguard đang ở [PR #7](https://github.com/Shoun93/Electoral-System/pull/7), chưa merge vào main. Pipeline AI đang khóa fail-closed cho đến khi các quyết định PM còn treo và nested group-aware OOF được xử lý.

## Các module

| Module | Thư mục | Trạng thái | Mô tả |
|---|---|---|---|
| Kiểm phiếu bằng AI | [`modules/ballot-validation-ai`](modules/ballot-validation-ai) | ✅ Chạy end-to-end; ⚠️ chuyển `PENDING_REVIEW` chưa đạt nghiệm thu (dữ liệu giả lập) | Phân loại phiếu bằng mô hình lai CNN theo ô + C4.5 (có luật IF-THEN), chuyển `PENDING_REVIEW` theo ngưỡng chọn trên OOF Train, ghi kết quả lên hash-chain |
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

## Hướng dẫn chạy từng bước

Làm theo đúng thứ tự. Mỗi bước có lệnh kiểm tra để xác nhận bước trước đã đúng.

### Bước 0 — Yêu cầu

| Công cụ | Dùng cho | Ghi chú |
|---|---|---|
| Python 3.11 | Module AI, sinh dữ liệu, test | Docker image cũng dùng 3.11 |
| Git | Lấy mã | |
| Go ≥ 1.21 | Test Merkle và biên dịch chaincode | Chưa biên dịch được chaincode nếu không có mạng để tải module |
| Docker | Chạy Fabric test-network, build image | Không chạy được trên Google Colab |
| GPU (tùy chọn) | Huấn luyện CNN nhanh hơn | Trên CPU, CNN toàn ảnh mất khoảng 20 phút |

### Bước 1 — Lấy mã và chọn nhánh

```bash
git clone https://github.com/Shoun93/electoral-system
cd electoral-system
git checkout feat/ba-giai-doan-1-selective     # nhánh đang phát triển; main chưa có thay đổi chọn PENDING_REVIEW
```

Kiểm tra: `git log --oneline -1` cho ra commit mới nhất của nhánh.

### Bước 2 — Cài thư viện Python

```bash
python3.11 -m venv .venv
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r modules/ballot-validation-ai/requirements.txt
```

Không có GPU, cài torch bản CPU trước để tránh tải bản CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Kiểm tra: `python -c "import torch, sklearn, cv2; print('ok')"` in ra `ok`.

### Bước 3 — Sinh dữ liệu giả lập

Chạy từ gốc repo:

```bash
python dataset/generate_dataset.py --out modules/ballot-validation-ai
```

Sinh 2500 ảnh (500 mỗi nhãn), seed 42, ghi vào `modules/ballot-validation-ai/data/`. Để thử nhanh, thêm `--samples-per-class 40` (200 ảnh).

Kiểm tra: dòng cuối in `VALIDATION: PASS`.

### Bước 4 — Chạy đơn vị kiểm thử

```bash
cd modules/ballot-validation-ai
python -m unittest discover -s tests -v
```

Kiểm tra: `Ran 18 tests` và `OK`.

### Bước 5 — Chạy pipeline AI

```bash
python main.py
```

Pipeline thực hiện theo thứ tự: tiền xử lý, baseline, C4.5 thủ công, CNN toàn ảnh (đối chứng), CNN theo ô (out-of-fold trên Train), chọn cấu hình C4.5 và ngưỡng `PENDING_REVIEW` trên OOF Train, huấn luyện mô hình lai, đánh giá trên Test, ghi hash-chain và Merkle batch.

Các dòng cần xem:
- `[chọn] min_leaf=... cf=... leaf_min_n=... τ_CNN=... τ_C4.5=...`: cấu hình được chọn trên OOF Train.
- `[nghiệm thu Test] ĐẠT` hoặc `KHÔNG ĐẠT`: kết quả so với ràng buộc Recall ≥ 0.98, Coverage ≥ 0.80 (chưa phê duyệt), Selective Risk ≤ 0.02.

**Cảnh báo:**
- Nếu đổi dữ liệu (chạy lại bước 3 với seed hoặc số ảnh khác), xóa cache trước: `rm -f results/mark_probs_cache.npz`. Cache chỉ kiểm tra danh sách file, không kiểm tra seed hay config.
- Lần chạy hiện tại trên Test **không ĐẠT** ràng buộc Selective Risk. Đây là kết quả thật, không chỉnh ngưỡng để đạt. Xem [`docs/audit/baseline_results.md`](docs/audit/baseline_results.md).
- Tập Test đã được dùng nhiều lần để khám phá. Không dùng số Test để báo cáo nghiệm thu.

Kết quả ghi trong `modules/ballot-validation-ai/results/`:

| File | Nội dung |
|---|---|
| `metrics.json` | Chỉ số tất cả mô hình, kết quả `hybrid_selective` |
| `predictions.csv` | Dự đoán từng phiếu Test, cột `status` (`AUTO_CLASSIFIED` / `PENDING_REVIEW`) |
| `hashchain.json` | Chuỗi khối cục bộ |
| `fabric_batch.json` | Lô lá Merkle để chaincode `CommitBatch` / `VerifyBatch` |
| `metrics_log.json` | Lịch sử các lần chạy |
| `mark_probs_cache.npz` | Cache xác suất CNN theo ô |

Thư mục `models/` chứa mô hình đã huấn luyện. `results/`, `models/` và `data/` không commit (xem `.gitignore`).

Kiểm tra: `python -c "import json; print(json.load(open('results/metrics.json'))['hybrid_selective']['coverage'])"` in ra một số (Coverage).

### Bước 6 — Kiểm tra Merkle trong Go

```bash
cd ../blockchain-ledger/chaincode/ballot
go test ./merkle/ -v
```

Kiểm tra: `PASS` cho 3 test.

Không chạy được `go build ./...` khi không có mạng để tải `fabric-contract-api-go`.

### Bước 7 — Fabric test-network (chỉ trên máy có Docker)

Bước này chưa được kiểm thử đầy đủ. Đối chiếu [`modules/blockchain-ledger/README.md`](modules/blockchain-ledger/README.md) trước khi làm.

```bash
cd "$HOME/fabric-samples/test-network"
./network.sh up createChannel -c electionchannel -ca
./addOrg3/addOrg3.sh up -c electionchannel -ca

cd /path/to/electoral-system                  # thay bằng đường dẫn repo của bạn
FABRIC_SAMPLES="$HOME/fabric-samples" modules/blockchain-ledger/network/deploy.sh
```

`deploy.sh` dùng quy trình `peer lifecycle` cũ. Quy trình CCaaS đã chạy trên máy phát triển nhưng chưa có trong repo.

Không commit thư mục `organizations/`, khóa hay MSP.

### Bước 8 — Chạy bằng Docker (tùy chọn)

Image này chỉ chạy pipeline AI, không chạy Fabric:

```bash
docker build -t ballot-ai .
mkdir -p results models
docker run --rm -v "$PWD/results:/app/modules/ballot-validation-ai/results" \
                -v "$PWD/models:/app/modules/ballot-validation-ai/models" ballot-ai
```

Dockerfile chưa được build kiểm chứng. Nếu container không có `data/` và `main.py` không tự sinh được, chạy bước 3 bên trong container trước.

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

`dataset/generate_dataset.py` sinh **2500 phiếu** (mặc định 500 mỗi nhãn, 5 nhãn):

| Nhãn | Ý nghĩa |
|---|---|
| `valid` | chọn 1–3 đại biểu |
| `blank` | phiếu trắng |
| `erased` | gạch xóa toàn bộ danh sách |
| `wrong_mark` | dấu đặt ngoài ô, hoặc ghi thêm tên ngoài danh sách |
| `multi_mark` | chọn 4–5 đại biểu (nhiều hơn quy định) |

Chia train 1750 / val 250 / test 500 theo `group_id` (một phiếu gốc không nằm ở hai tập). Mỗi ảnh có tọa độ 5 ô đã áp phép xoay nhiễu (`data/labels/boxes.json`) và danh sách ô có dấu (`o_co_danh_dau`) để kiểm chứng nhãn.

### Kết quả

> Lần chạy gần nhất: 2500 phiếu, seed 42, tập test 500. Tập test đã được dùng nhiều lần để khám phá cấu hình, nên các số dưới đây **không phải bằng chứng nghiệm thu**. Chi tiết: [`docs/audit/baseline_results.md`](docs/audit/baseline_results.md).

| Mô hình | Accuracy | Macro-F1 | Recall "không hợp lệ" | Precision "không hợp lệ" |
|---|---|---|---|---|
| Baseline majority | 0.200 | 0.067 | 0.000 | 0.000 |
| Baseline Naive Bayes | 0.802 | 0.781 | 0.910 | 0.995 |
| Baseline kNN (k=5) | 0.916 | 0.914 | 0.948 | 0.977 |
| C4.5 (đặc trưng thủ công) | 0.916 | 0.912 | 0.935 | 0.997 |
| **Mô hình lai (CNN theo ô + C4.5)** | **0.896** | **0.889** | **0.943** | **1.000** |
| CNN toàn ảnh 224×224 (đối chứng, không học) | 0.200 | 0.067 | 0.000 | 0.000 |

- CNN phát hiện dấu theo ô đạt accuracy 1.0 trên tập test (cấp ô).
- CNN toàn ảnh không học: val_acc cố định 0.20. Chưa điều tra nguyên nhân. Đối chứng này chưa có giá trị.
- Ngưỡng chuyển `PENDING_REVIEW` (cấu hình chọn trên OOF Train): Invalid Screening Recall 1.000, Coverage 0.856, Referral Rate 0.144, **Selective Risk 0.068 (vượt ngưỡng 0.02)**. Không đạt nghiệm thu.

**Đọc kết quả thận trọng:** trên dữ liệu giả lập, 8 đặc trưng thủ công gần như tách hoàn toàn các lớp, nên C4.5 đơn thuần đã đạt kết quả gần tối đa. Lợi ích của CNN chỉ kiểm chứng được trên ảnh scan thật (nhiễu, bóng, tẩy xóa, chữ ký không đều). Lớp `wrong_mark` là lớp khó nhất; 19 phiếu `wrong_mark` trên tập test rơi vào một lá C4.5 không tách được (xem audit).

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
