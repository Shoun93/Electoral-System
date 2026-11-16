# Baseline và phân tích lỗi (Phase 0)

Dữ liệu: 2500 ảnh giả lập, seed 42. Train 1750 (val 250 tách từ Train), Test 500. Nguồn số liệu: `results/metrics.json` của lần chạy gần nhất (cấu hình chọn bởi lưới mở rộng).

**Cảnh báo về tập Test:** trong phiên này, Test đã được xem nhiều lần để so sánh cấu hình và tiêu chí. Các số dưới đây **không** là bằng chứng nghiệm thu. Cần tập giữ lại mới để đánh giá cuối.

## 1. Baseline (Test)

| Mô hình | Accuracy | Macro-F1 | Invalid Recall | Invalid Precision |
|---|---|---|---|---|
| Majority | 0.200 | 0.067 | 0.000 | 0.000 |
| Naive Bayes | 0.802 | 0.781 | 0.910 | 0.995 |
| kNN (k=5) | 0.916 | 0.914 | 0.948 | 0.977 |
| C4.5 (đặc trưng thủ công) | 0.916 | 0.912 | 0.935 | 0.997 |
| CNN toàn ảnh 224×224 | 0.200 | 0.067 | 0.000 | 0.000 |
| Lai CNN ô + C4.5 (lần chạy gần nhất) | 0.896 | 0.889 | 0.943 | 1.000 |

CNN toàn ảnh **không học** (loss đứng ở ln5 ≈ 1.6096, val_acc 0.20). Đối chứng này chưa có giá trị.

CNN dấu theo ô: accuracy 1.0 trên Test (cấp ô).

## 2. Chỉ số selective (Test, lần chạy gần nhất)

| Chỉ số | Giá trị | Ràng buộc |
|---|---|---|
| Invalid Screening Recall | 1.000 | ≥ 0.98 ✓ |
| Automatic Invalid Recall | 0.943 | — |
| Coverage | 0.856 | ≥ 0.80 ✓ (chưa phê duyệt) |
| Referral Rate | 0.144 | — |
| Selective Risk | 0.068 | ≤ 0.02 ✗ |

Trên OOF Train: không cấu hình nào trong 192 cấu hình đạt đồng thời ba ràng buộc. Cấu hình chọn có Selective Risk OOF 0.120.

## 3. Lịch sử lần chạy trên Test (khám phá)

Các lần chạy ghi trong `results/metrics_log.json` (hai lần đầu) và log phiên:
- Lần 1 (lai, grid cũ): accuracy 0.950, Invalid Screening Recall 1.000, Coverage 0.854.
- Lần 2 (lai, grid cũ): accuracy 0.896.
- Các lần sau: thay đổi xếp hạng và mở rộng lưới sau khi xem Test.

Kết luận: đã dùng Test để điều chỉnh tiêu chí và lưới → vi phạm nguyên tắc "không dùng Test để chọn". Cần tập giữ lại mới.

## 4. Kiểm tra leakage

| Kiểm tra | Kết quả |
|---|---|
| Group overlap Train ∩ Val | **188 / 188 group của Val thuộc Train** |
| Group overlap Train ∩ Test | 0 |
| Group overlap Val ∩ Test | 0 |
| Số group Train / Val / Test | 400 / 188 / 100 |
| OOF CNN theo ô | Đã chia theo group (`StratifiedGroupKFold`) |
| OOF C4.5 | Đã chia theo group |

Hệ quả: Val không độc lập với Train. Val dùng để chọn checkpoint CNN toàn ảnh và báo cáo C4.5. Không ảnh hưởng Test (Test sạch).

Sửa: tách Val theo group trước khi tách theo nhãn (Phase 1).

## 5. Phân tích lỗi wrong_mark (Test)

**Phân bố dự đoán của 100 phiếu `wrong_mark` trên Test (lần chạy gần nhất):**
- 19 → `blank`, trạng thái `AUTO_CLASSIFIED`.
- 23 → `valid`, trạng thái `PENDING_REVIEW` (đã được chuyển kiểm tra).
- Còn lại: 58 phiếu khác đúng hoặc đã chuyển PENDING.

**19 phiếu `wrong_mark` → `blank`:**
- Đều rơi vào một lá C4.5: `cnn_count_marked ≤ 0.5` (tức CNN không thấy dấu trong ô), `max_box_density ≤ 0.06037`, `left_band_ink ≤ 0.00065`.
- Lá: `n = 404`, counts `[valid 0, blank 341, erased 0, wrong_mark 63, multi 0]` (theo thứ tự nhãn `[valid, blank, erased, wrong_mark, multi_mark]` của `counts` đã in: `[0, 341, 0, 63, 0]`).
- Độ tin cậy lá ≈ 0.84 cho `blank`. Lá này không đủ tách hai lớp.
- Nguyên nhân khả dĩ: `wrong_mark` là dấu nằm ngoài ô. Nếu không có dấu trong ô và đặc trưng ngoài ô (`extra_zone_ink`) không được chọn đúng ngưỡng, lá gộp chung hai lớp. Cần kiểm tra thêm đặc trưng `extra_zone_ink` và ngưỡng trong nhánh này (Phase 1).

**Lưu ý:** lá này là đặc trưng của mô hình chọn bằng lưới mở rộng. Các cấu hình khác có thể có lá khác.

## 6. Pipeline và đầu ra

| Hạng mục | Kết quả |
|---|---|
| Hash-chain demo sửa 1 block | `validate()` = False (đúng: phát hiện sửa). Thông báo gốc "chuỗi còn hợp lệ: False" dễ nhầm. |
| `metrics_log.json` | Có 2 lần trước khi làm mới; `results/` bị ghi đè mỗi lần chạy (P2) |
| Cache CNN theo ô | Khóa theo danh sách file, không theo seed/config/checksum (P0-P1) |

## 7. Ổn định qua seed dữ liệu (Phase 0, chỉ báo cáo)

Sinh lại dữ liệu với `--seed 1, 2, 3` (mỗi seed 2500 ảnh, chia theo group của generator). Chỉ chạy baseline và C4.5 thủ công trên 8 đặc trưng (không có CNN, để tiết kiệm thời gian). Tập test không dùng để chọn gì; số liệu chỉ để đo độ biến thiên.

| Mô hình | seed 1 acc | seed 2 acc | seed 3 acc | seed 42 acc (lần chạy chính) |
|---|---|---|---|---|
| Majority | 0.200 | 0.200 | 0.200 | 0.200 |
| Naive Bayes | 0.800 | 0.848 | 0.800 | 0.802 |
| kNN (k=5) | 0.948 | 0.904 | 0.918 | 0.916 |
| C4.5 (thủ công) | 0.942 | 0.924 | 0.936 | 0.916 |

Invalid Recall của C4.5 thủ công: 0.960 / 0.928 / 0.950 (seed 1/2/3), 0.935 (seed 42).

Kết luận: C4.5 thủ công có accuracy 0.916–0.942 (biên độ 0.026, 4 seed) và invalid recall 0.928–0.960. Biên độ của kNN là 0.904–0.948 và của Naive Bayes là 0.800–0.848. C4.5 ổn định hơn cả hai trên các seed này. Không có số liệu hybrid qua nhiều seed (chưa chạy do chi phí CNN).

Dữ liệu seed 1–3 nằm ngoài repo (`scratchpad`). Script: `stab.py`.

## 8. Kiểm tra chaincode và Merkle (Go)

| Kiểm tra | Kết quả |
|---|---|
| `go test ./merkle/ -v` (Go đã cài tại `/usr/local/go`) | ✅ PASS: `TestVector`, `TestSingleLeafIsItself`, `TestRejectsBadLeaf` |
| `go build ./...` trong `chaincode/ballot` | ❌ Không tải được `github.com/hyperledger/fabric-contract-api-go/v2 v2.2.0` (không truy cập `proxy.golang.org` từ máy này). Chưa biên dịch được `main.go`. |
| `go vet ./...` | ❌ Cùng nguyên nhân (không có module). |
| Đọc `main.go` (177 dòng) | Có 11 hàm: `requireMSP`, `putNew`, `RecordModelAnchor`, `RecordBallot`, `GetBallot`, `CommitBatch`, `VerifyBatch`, `PutTallyPrivate`, `VerifyTallyPrivate`, `main`. Không có hàm pending/review. |
| Triển khai Fabric | Chưa chạy. `deploy.sh` không kiểm thử. |

Hạn chế kiểm tra: chaincode chưa biên dịch trên máy này. Cần mạng để tải module, hoặc vendor module trước.

## 9. Việc đã làm và còn lại của Phase 0

Đã có: `current_state.md`, `baseline_results.md`, `known_failures_wrong_mark.csv` (19 phiếu, cùng một lá), số liệu ổn định qua seed, kiểm tra Go Merkle.

Chưa xong: chaincode chưa biên dịch được (cần mạng hoặc vendor). Phần này nằm ngoài khả năng hoàn thành trên máy này.
