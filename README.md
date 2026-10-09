# Electoral System

Hệ thống bầu cử điện tử, gồm nhiều module. Hiện chỉ có module kiểm phiếu bằng AI đã chạy được (dạng demo).

## Các module

| Module | Thư mục | Trạng thái | Mô tả |
|---|---|---|---|
| Kiểm phiếu bằng AI | [`modules/ballot-validation-ai`](modules/ballot-validation-ai) | ✅ Chạy được end-to-end (demo, dữ liệu giả lập) | Phân loại phiếu bằng mô hình lai CNN theo ô + C4.5 (có luật giải thích được), ghi kết quả lên hash-chain |
| Sổ cái blockchain | `modules/blockchain-ledger` | ⚠️ Mã Fabric đã viết, chưa chạy | Hyperledger Fabric 3 tổ chức, chaincode (MSP, ghi một lần), private data, cây Merkle; PBFT chưa làm |
| Đăng ký/xác thực cử tri | `modules/voter-registration` | 🔜 Kế hoạch | Xác thực danh tính cử tri trước khi bỏ phiếu |
| Tổng hợp kết quả | `modules/results-tally-service` | 🔜 Kế hoạch | Tổng hợp kết quả từ nhiều trạm kiểm phiếu, đối chiếu với blockchain |
| Giao diện web | `modules/web-portal` | 🔜 Kế hoạch | Tra cứu kết quả công khai |

Sinh dữ liệu giả lập dùng chung: [`dataset/generate_dataset.py`](dataset/generate_dataset.py).

## Bắt đầu

```bash
cd modules/ballot-validation-ai
pip install -r requirements.txt
python main.py
```

Chi tiết xem `modules/ballot-validation-ai/README.md`.
