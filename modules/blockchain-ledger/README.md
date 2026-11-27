# Sổ cái blockchain: Hyperledger Fabric

Lưu kết quả phân loại phiếu lên sổ cái Hyperledger Fabric với 3 tổ chức, endorsement 2-trong-3,
collection riêng cho dữ liệu nhạy cảm và gốc Merkle cho từng lô.

## Trạng thái kiểm thử

| Thành phần | File | Trạng thái |
|---|---|---|
| Cây Merkle (Go, thư viện chuẩn) | `chaincode/ballot/merkle/` | ✅ Đã kiểm thử: `go test ./merkle/` |
| Cây Merkle (Python) | `modules/ballot-validation-ai/src/merkle.py` | ✅ Cùng vector với Go |
| Chaincode (MSP, ghi một lần, private data, verify) | `chaincode/ballot/main.go` | ⚠️ `go vet`, `go test` với stub giả trong CI. Chưa có kiểm thử tự động trên mạng Fabric |
| Collection riêng | `chaincode/ballot/collections.json` | ⚠️ Chưa triển khai |
| Triển khai mạng | `network/deploy.sh` | ⚠️ Chưa chạy |
| Đồng thuận PBFT (SmartBFT) | chưa có | ❌ Chưa làm. Cần cấu hình orderer BFT theo phiên bản Fabric đang dùng |

Các mục ⚠️ chưa được kiểm chứng trên mạng Fabric bằng mã hiện tại.

## Thiết kế

- **Tổ chức (MSP):** `Org1MSP` (ban tổ chức, đăng ký mô hình), `Org2MSP` (tổ kiểm phiếu, ghi kết quả), `Org3MSP` (giám sát, chỉ đọc).
  Chaincode kiểm tra MSP của người gọi qua `requireMSP`.
- **Endorsement:** `OutOf(2, ...)`, cần 2 trong 3 tổ chức. Tên MSP khớp với test-network của fabric-samples.
- **Ghi một lần:** `putNew` không ghi đè khóa đã có, nên bản ghi đã commit không sửa được.
- **Private data:** số serial phiếu (`serial`) và 32 byte ngẫu nhiên do client sinh (`salt`) đi qua transient map,
  chỉ nằm trong collection `CountingBoardPrivate`. Sổ cái công khai chỉ có SHA-256 của `salt || serial`,
  nên không dò ngược được serial dễ đoán.
- **Endorse `PutTallyPrivate`:** client chỉ gửi giao dịch tới peer Org2 và Org3. Peer endorse nhận transient map,
  nên gửi tới Org1 sẽ làm lộ `serial`. `requiredPeerCount: 1` buộc phát private data tới ít nhất một peer
  thành viên khác trước khi endorse (chưa kiểm thử trên mạng thật).
- **Merkle:** pipeline tính gốc Merkle cho lô phiếu (`results/runs/<run_id>/fabric_batch.json`), `CommitBatch` ghi gốc lên sổ cái,
  `VerifyBatch` tính lại từ danh sách lá và so sánh.

## Chạy (khi có môi trường Docker)

```bash
cd modules/ballot-validation-ai && python main.py         # đang khóa; khi mở khóa tạo results/runs/<run_id>/fabric_batch.json
cd ../blockchain-ledger/chaincode/ballot && go test ./...    # kiểm thử chaincode và Merkle
FABRIC_SAMPLES=~/fabric-samples ../../network/deploy.sh    # cần mạng 3 tổ chức đã lên
```

Trước khi dùng thật, hãy kiểm tra lại đường dẫn trong `deploy.sh` theo phiên bản fabric-samples.

## Chạy trên Docker

Pipeline (Python) chạy trong container:

```bash
# từ thư mục gốc repo electoral-system
docker build -t ballot-ai .
mkdir -p results models
docker run --rm -v "$PWD/results:/app/modules/ballot-validation-ai/results" \
                -v "$PWD/models:/app/modules/ballot-validation-ai/models" ballot-ai
```

Mạng Fabric dùng `test-network` của fabric-samples (cần Docker, Go và Fabric binary):

```bash
curl -sSLO https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh
bash install-fabric.sh docker samples binary          # tải fabric-samples + image + binary
cd fabric-samples/test-network
./network.sh up createChannel -c electionchannel -ca   # mạng 2 tổ chức (Org1, Org2)
./addOrg3/addOrg3.sh up -c electionchannel -ca        # thêm Org3 (cần để đủ 3 tổ chức)
cd /path/to/electoral-system/modules/blockchain-ledger
FABRIC_SAMPLES=$HOME/fabric-samples ./network/deploy.sh   # triển khai chaincode, chạy một lần
```

Lưu ý: `deploy.sh` giả định `electionchannel` và thư mục `organizations/` của test-network; chưa kiểm thử trên máy thật.
