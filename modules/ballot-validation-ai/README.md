# Ballot Validation AI — Phase 1

Module nghiên cứu phân loại phiếu bầu bằng CNN phát hiện dấu theo ô + C4.5,
kèm đánh giá selective classification và hash-chain cục bộ.

> **Trạng thái:** đã có Phase 1 safeguards; pipeline huấn luyện đang khóa (mục 4),
> nên `predict.py` chỉ chạy được khi đã có một run hoàn tất trong `results/runs/`. Đây là demo nghiên cứu trên dữ liệu giả lập; không phải hệ thống kiểm phiếu
> đã được nghiệm thu. Observed Test đã được sử dụng trong quá trình phát triển,
> chỉ được báo cáo mô tả, không dùng để chọn mô hình/ngưỡng hoặc nghiệm thu.
> Independent Test acceptance đang BLOCKED.

## 1. Thành phần

- src/preprocess.py: tiền xử lý ảnh, group-aware Train/Validation split và
  kiểm tra group leakage giữa Train/Validation/Test.
- src/feature_c45.py, src/c45.py: đặc trưng cấu trúc và bộ phân loại C4.5.
- src/train_cnn.py: CNN phát hiện dấu theo ô và CNN toàn ảnh đối chứng.
- src/stray_cnn.py: CNN phát hiện dấu ngoài ô, chỉ làm cờ PENDING_REVIEW.
- src/decision.py: confidence, PENDING_REVIEW và các chỉ số selective.
- src/evaluate.py: báo cáo phân lớp, cận trên Clopper–Pearson, Invalid → Valid Auto.
- src/manifest.py: thư mục run bất biến, manifest, sổ đăng ký Test.
- src/cache.py: cache xác suất CNN theo checksum dữ liệu, cấu hình, checkpoint,
  seed, folds và schema; ghi cache atomic.
- src/hashchain.py, src/merkle.py: hash-chain cục bộ và Merkle batch.
- tests/: unit tests không cần sinh dataset hoặc huấn luyện mô hình.

## 2. Quy ước đánh giá

- Automatic Invalid Recall: chỉ tính phiếu INVALID được tự động kết luận không hợp lệ.
- Invalid Referral Rate: phần INVALID được chuyển PENDING_REVIEW.
- Invalid Interception Rate: gồm tự động phát hiện và chuyển xác minh; không được
  gọi là automatic recall.
- Danh sách nhãn INVALID phải được cấu hình sau khi PM phê duyệt. Không suy diễn
  blank là INVALID hoặc PENDING_REVIEW.
- Coverage 0.80 và 0.34 chưa được phê duyệt. Vai trò nghiệm thu của mục tiêu NFR-10
  0.98 cần PM xác nhận. Selective Risk tối đa 0.02 là mục tiêu đã ghi nhận.
- Mọi mẫu số bằng 0 phải được báo cáo là undefined, không thay bằng 1 hoặc 0.
- CI theo group chưa được chốt đầy đủ; không tự tạo khoảng tin cậy khi chưa có
  phê duyệt phương pháp.

## 3. Unit tests

Từ thư mục module:

    python -m unittest discover -s tests -v

Bộ test kiểm tra logic split, group leakage, metric, policy gate, cache, manifest và nối dây
main.py → predict.py (huấn luyện/OOF là hàm giả).
Workflow GitHub Actions cài các dependency kiểm thử tối thiểu và chạy lệnh này
trên pull request.

## 4. Điều kiện để chạy pipeline huấn luyện

Lệnh python main.py hiện **fail-closed có chủ đích**. Nó chỉ chạy khi đồng thời:

1. PM phê duyệt định nghĩa INVALID, bao gồm cách xử lý BLANK.
2. PM phê duyệt rõ các ngưỡng nghiệm thu; không tự dùng Coverage 0.80/0.34.
3. PM cấp phê duyệt riêng cho OOF.
4. Nested group-aware OOF selection được triển khai và kiểm tra. Flat OOF grid
   search bị vô hiệu vì dùng cùng OOF predictions để chọn và báo cáo sẽ gây
   selection optimism.

Chưa đáp ứng các điều kiện trên thì không chạy training, OOF, inference hay
đánh giá trên Observed Test. Module không tự động sinh hoặc thay đổi dataset.

Invalid → Valid Auto trong metrics.json là `BLOCKED_POLICY_NOT_APPROVED` khi chưa có nhãn INVALID;
khi có, nhãn đối chiếu vẫn là nhãn hình thái, không phải `legal_label`.

## 5. Ngoài phạm vi Phase 1

Triển khai Fabric, PBFT/SmartBFT, backend, web app, dữ liệu scan thật, MLOps đầy đủ
và Independent Test governance không được tuyên bố hoàn thành bởi các thay đổi
Phase 1 này.

Chi tiết: docs/audit/phase1_status.md.
