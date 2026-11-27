# Audit trạng thái hiện tại (Phase 0)

Ngày: 16/11/2026. Nhánh: `feat/ba-giai-doan-1-selective` (HEAD `a31d9ad`). Repo chính `main` tại `9fcdaa9`.
Phạm vi: đọc code, chạy lại pipeline AI trên dữ liệu 2500 ảnh, phân tích chỉ đọc. Không đổi logic mã.

## 1. Chức năng đã kiểm chứng

| Thành phần | File | Trạng thái | Bằng chứng |
|---|---|---|---|
| Sinh dữ liệu giả lập | `dataset/generate_dataset.py` | Chạy được, 2500 ảnh, group_id | `VALIDATION: PASS`, `labels/all.csv` |
| Tiền xử lý, 8 đặc trưng | `src/preprocess.py`, `src/feature_c45.py` | Chạy được; npz có `groups` | pipeline chạy |
| C4.5 tự cài | `src/c45.py` | Chạy được; có `leaf_size` | unit test |
| CNN toàn ảnh (đối chứng) | `src/train_cnn.py` | **Không học**: val_acc = 0.20 | log pipeline |
| CNN dấu theo ô + OOF | `src/train_cnn.py` | Chạy được; OOF chia theo group | accuracy ô test = 1.0 |
| Mô hình lai | `main.py` | Chạy được | `metrics.json` |
| Quyết định PENDING (OR rule, τ, leaf rule) | `src/decision.py` | Chạy được; có ràng buộc nghiệm thu | unit test, log |
| Lưới C4.5 chọn trên OOF | `src/decision.py` | Chạy được; 192 cấu hình | log `[grid]` |
| Hash-chain + demo sửa | `src/hashchain.py` | Chạy được; phát hiện sửa đổi | `validate()` trả `False` sau sửa |
| Merkle (Python) | `src/merkle.py` | Có test vector khớp Go | unit test |
| PSI, drift | `src/monitor.py` | Có hàm; chưa có dữ liệu sản xuất | unit test |
| Chaincode Go | `modules/blockchain-ledger/chaincode/ballot/main.go` | Biên dịch và `go vet` sạch trong bản sao có replace GitHub; repo gốc cần mạng Go proxy; chưa chạy trên Fabric | `main_test.go` (stub giả, thêm 27/11); xem `chaincode_build_verify.md` |
| Merkle (Go) | `chaincode/ballot/merkle/` | ✅ `go test ./merkle/`: 3/3 PASS | `baseline_results.md` mục 8 |
| Triển khai Fabric | `modules/blockchain-ledger/network/deploy.sh` | **Chưa kiểm thử** (script ghi rõ) | — |
| Dockerfile | `Dockerfile` (gốc) | **Chưa build** | — |
| STRIDE | `docs/stride.md` | Chỉ phạm vi chaincode | — |

Unit test: 18/18 OK (`python3 -m unittest discover -s tests`).

## 2. Chức năng thiếu

- CI (`.github/` không tồn tại). Docker Compose. MLflow. DVC.
- Backend API, Web App, JSON Schema.
- Chaincode: không có khái niệm `PENDING_REVIEW` (không có hàm pending/review; FR-10 chưa làm).
- Mã R01–R10, rule engine, `legal_label` (chờ sổ tay pháp lý — BLOCKED_BY_BUSINESS).
- Document Validation (R01, R02): chưa có.
- Go unit test cho chaincode (chưa viết; chaincode chưa biên dịch được trên máy này).
- Pipeline không ghi `PENDING_REVIEW` vào Fabric thật; chỉ vào hash-chain cục bộ và Merkle leaf.

## 3. Đang chạy được

- Pipeline AI end-to-end (CPU máy này; Colab GPU trước đó).
- Unit test Python.
- Hash-chain cục bộ và Merkle batch JSON.

## 4. Chưa chạy được / chưa kiểm chứng

- Fabric test-network: chưa.
- Go build chaincode: chưa (thiếu module, không có mạng). Go test merkle: đã chạy, PASS.
- Docker build: chưa.
- CI: không có.

## 5. Vấn đề phát hiện trong phiên audit này

Xem `baseline_results.md` mục 4 (kiểm tra leakage và phân tích lỗi). Tóm tắt:
- Validation tách từ Train theo nhãn, **không theo group**: 188/188 group của validation nằm trong Train. Train∩Test = 0, Val∩Test = 0.
- Các phiếu `wrong_mark` bị dự đoán `blank` (19 phiếu Test) rơi vào một lá duy nhất: `n=404`, counts `[blank 341, wrong_mark 63]`. Đặc trưng không phân biệt được hai lớp ở lá này vì không có phiếu nào có dấu trong ô (CNN count = 0). Đặc trưng ngoài ô (`extra_zone_ink`) không đủ để tách.
- Lịch sử: nhiều lần chạy đã đánh giá trên Test trong phiên này, gồm cả các lần thay đổi tiêu chí xếp hạng và mở rộng lưới sau khi xem Test. Kết quả Test từ các lần này là **khám phá**, không phải nghiệm thu.

## 6. Module hoàn thành / chưa hoàn thành

| Module | Trạng thái |
|---|---|
| AI pipeline (phần cơ bản) | Hoàn thành, có test |
| Selective classification | Đang làm; **không đạt** ràng buộc nghiệm thu |
| Legal label & R01–R10 | Chưa làm; BLOCKED_BY_BUSINESS |
| Chaincode nghiệp vụ (pending, review) | Chưa làm |
| Fabric deployment | Chưa chạy |
| Backend API | Chưa làm |
| MLOps (CI, MLflow) | Chưa làm |
| Web App | Chưa làm |
| STRIDE đầy đủ | Chỉ chaincode |

## 7. Kế hoạch thay đổi file (Phase 1 — chờ duyệt riêng)

- `src/preprocess.py`: tách validation theo group.
- `src/train_cnn.py`: khắc phục CNN toàn ảnh không học (cần điều tra).
- `src/feature_c45.py`: xem xét đặc trưng phân biệt `wrong_mark` với `blank` khi không có dấu trong ô.
- `main.py`: khóa cache theo checksum dữ liệu và config.
- `tests/`: thêm test cho group split, cache, grid.
- `docs/`: cập nhật README khớp code.
