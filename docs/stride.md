# Phân tích STRIDE: tầng sổ cái Hyperledger Fabric

Phạm vi: chaincode `ballot`, mạng Fabric 3.1.5 (test-network, 3 tổ chức, Raft), kênh `electionchannel`.
Trạng thái: kiểm thử trên mạng cục bộ. Chưa có PBFT, chưa có kiểm thử nút độc hại.

| Mối đe dọa | Mô tả | Biện pháp trong hệ thống | Kịch bản kiểm thử | Kết quả |
|---|---|---|---|---|
| Spoofing (giả mạo danh tính) | Tổ chức không được phép ghi bản ghi | `requireMSP`: chỉ Org2MSP ghi phiếu | Org1 và Org3 gọi `RecordBallot` | Đạt: bị từ chối `MSP ... không được phép` |
| Tampering (sửa dữ liệu) | Ghi đè bản ghi đã commit | `putNew`: khóa đã tồn tại thì từ chối | Ghi lại `ballot_0001.png` | Đạt: `khóa ... đã tồn tại, không ghi đè` |
| Tampering (sửa lô Merkle) | Sửa một lá trong lô | `VerifyBatch` tính lại gốc Merkle | Sửa lá đầu, gọi `VerifyBatch` | Đạt: `false` |
| Tampering (sửa private data) | Sửa giá trị serial đã ghi | `VerifyTallyPrivate` so băm công khai | Chưa thực hiện | Chưa kiểm thử |
| Repudiation (chối bỏ) | Không truy được ai ghi | Trường `recorder` (MSP) lưu trong bản ghi | Đọc `GetBallot` sau khi Org2 ghi | Đạt một phần: có `recorder`, chưa có nhật ký riêng |
| Information disclosure (lộ thông tin) | Org không thuộc collection đọc private data | `member_only_read` trên `CountingBoardPrivate` | Org1 gọi `VerifyTallyPrivate` | Đạt: `không đọc được private data` |
| Information disclosure (lộ thông tin) | Dữ liệu phiếu công khai trên sổ cái | Không có. `ballot_id`, `pred`, `conf` lưu dạng rõ | Quan sát `GetBallot` | Rủi ro còn lại. `ballot_id` là tên file ảnh, cần thay bằng mã giả danh |
| Denial of service | Nút sập hoặc nghẽn orderer | Raft chịu được sập một số nút, không chịu được nút độc hại | Chưa thực hiện | Chưa kiểm thử. PBFT chưa làm |
| Elevation of privilege | Ghi phiếu bằng chữ ký không đủ tổ chức | Endorsement policy `OutOf(2, Org1, Org2, Org3)` | Invoke chỉ endorse qua Org2 | Đạt: giao dịch bị vô hiệu (không có bản ghi `ballot_0004.png`) |

## Giới hạn đã biết
- Blockchain chỉ chứng minh bản ghi không bị sửa sau khi commit. Không chứng minh nhãn AI đúng với phiếu giấy.
- Nút độc hại trong orderer và nút độc hại trong peer chưa được kiểm thử. Cần PBFT (SmartBFT, Fabric 3.x) cho phần này.
- `ballot_id` nên thay bằng mã giả danh trước khi dùng dữ liệu thật.
