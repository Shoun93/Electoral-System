// Chaincode ghi kết quả phân loại phiếu lên sổ cái Hyperledger Fabric.
// CHƯA KIỂM THỬ: cần chạy trên mạng Fabric thật (xem network/deploy.sh). Chỉ đã kiểm tra cú pháp (gofmt).
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strconv"

	"ballot/merkle"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

const (
	mspCommission    = "Org1MSP"              // Org1: ban tổ chức bầu cử, đăng ký mô hình
	mspCountingBoard = "Org2MSP"              // Org2: tổ kiểm phiếu, ghi kết quả
	mspSupervisor    = "Org3MSP"              // Org3: giám sát, chỉ đọc + kiểm chứng
	privateColl      = "CountingBoardPrivate" // collection riêng, xem collections.json
)

type BallotRecord struct {
	BallotID    string `json:"ballotId"`
	Pred        string `json:"pred"`
	Conf        string `json:"conf"`
	ImageSHA256 string `json:"imageSha256"`
	Recorder    string `json:"recorder"`
}

type BatchRecord struct {
	BatchID  string `json:"batchId"`
	Root     string `json:"root"`
	Count    int    `json:"count"`
	Recorder string `json:"recorder"`
}

type BallotChaincode struct {
	contractapi.Contract
}

// requireMSP kiểm tra danh tính người gọi (MSP) thuộc nhóm được phép.
func requireMSP(ctx contractapi.TransactionContextInterface, allowed ...string) (string, error) {
	id, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return "", err
	}
	for _, a := range allowed {
		if id == a {
			return id, nil
		}
	}
	return "", fmt.Errorf("MSP %s không được phép thực hiện thao tác này", id)
}

// validPred: nhãn pipeline được phép ghi (khớp configs/config.yaml, thêm PENDING_REVIEW).
var validPred = map[string]bool{"valid": true, "blank": true, "erased": true, "wrong_mark": true,
	"multi_mark": true, "PENDING_REVIEW": true}

// requireSHA256 kiểm tra chuỗi là mã băm SHA-256 dạng hex.
func requireSHA256(name, v string) error {
	if b, err := hex.DecodeString(v); err != nil || len(b) != sha256.Size {
		return fmt.Errorf("%s phải là SHA-256 hex", name)
	}
	return nil
}

// putNew ghi một khóa MỘT LẦN: không ghi đè, để bản ghi không thể bị sửa sau khi đã commit.
func putNew(ctx contractapi.TransactionContextInterface, key string, v interface{}) error {
	existing, err := ctx.GetStub().GetState(key)
	if err != nil {
		return err
	}
	if existing != nil {
		return fmt.Errorf("khóa %s đã tồn tại, không ghi đè", key)
	}
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(key, b)
}

func (c *BallotChaincode) RecordModelAnchor(ctx contractapi.TransactionContextInterface, modelSHA, manifestSHA string) error {
	msp, err := requireMSP(ctx, mspCommission)
	if err != nil {
		return err
	}
	if err := requireSHA256("modelSHA", modelSHA); err != nil {
		return err
	}
	if err := requireSHA256("manifestSHA", manifestSHA); err != nil {
		return err
	}
	return putNew(ctx, "model:"+modelSHA, map[string]string{"modelSha256": modelSHA, "manifestSha256": manifestSHA, "recorder": msp})
}

func (c *BallotChaincode) RecordBallot(ctx contractapi.TransactionContextInterface, ballotID, pred, conf, imageSHA string) error {
	msp, err := requireMSP(ctx, mspCountingBoard)
	if err != nil {
		return err
	}
	if ballotID == "" || !validPred[pred] {
		return fmt.Errorf("ballotID rỗng hoặc nhãn %q không hợp lệ", pred)
	}
	// So sánh dạng này loại cả NaN.
	if v, err := strconv.ParseFloat(conf, 64); err != nil || !(v >= 0 && v <= 1) {
		return fmt.Errorf("conf %q phải là số trong [0,1]", conf)
	}
	if err := requireSHA256("imageSHA", imageSHA); err != nil {
		return err
	}
	return putNew(ctx, "ballot:"+ballotID, BallotRecord{ballotID, pred, conf, imageSHA, msp})
}

func (c *BallotChaincode) GetBallot(ctx contractapi.TransactionContextInterface, ballotID string) (*BallotRecord, error) {
	b, err := ctx.GetStub().GetState("ballot:" + ballotID)
	if err != nil {
		return nil, err
	}
	if b == nil {
		return nil, fmt.Errorf("không có phiếu %s", ballotID)
	}
	var r BallotRecord
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

// CommitBatch ghi gốc Merkle của một lô phiếu.
func (c *BallotChaincode) CommitBatch(ctx contractapi.TransactionContextInterface, batchID, root string, count int) error {
	msp, err := requireMSP(ctx, mspCountingBoard)
	if err != nil {
		return err
	}
	if batchID == "" || count <= 0 {
		return fmt.Errorf("batchID rỗng hoặc count %d không dương", count)
	}
	if err := requireSHA256("root", root); err != nil {
		return err
	}
	return putNew(ctx, "batch:"+batchID, BatchRecord{batchID, root, count, msp})
}

// VerifyBatch tính lại gốc Merkle từ danh sách lá (JSON mảng hex) và so với gốc đã commit.
func (c *BallotChaincode) VerifyBatch(ctx contractapi.TransactionContextInterface, batchID, leavesJSON string) (bool, error) {
	b, err := ctx.GetStub().GetState("batch:" + batchID)
	if err != nil {
		return false, err
	}
	if b == nil {
		return false, fmt.Errorf("không có lô %s", batchID)
	}
	var rec BatchRecord
	if err := json.Unmarshal(b, &rec); err != nil {
		return false, err
	}
	var leaves []string
	if err := json.Unmarshal([]byte(leavesJSON), &leaves); err != nil {
		return false, err
	}
	root, err := merkle.Root(leaves)
	if err != nil {
		return false, err
	}
	return root == rec.Root && len(leaves) == rec.Count, nil
}

// PutTallyPrivate lưu số serial phiếu vào private data. serial đi qua transient map
// để KHÔNG xuất hiện trong đối số giao dịch hay sổ cái công khai.
func (c *BallotChaincode) PutTallyPrivate(ctx contractapi.TransactionContextInterface, ballotID string) error {
	if _, err := requireMSP(ctx, mspCountingBoard); err != nil {
		return err
	}
	transient, err := ctx.GetStub().GetTransient()
	if err != nil {
		return err
	}
	serial, ok := transient["serial"]
	if !ok || len(serial) == 0 {
		return fmt.Errorf("thiếu transient 'serial'")
	}
	if ballotID == "" {
		return fmt.Errorf("ballotID rỗng")
	}
	// Ghi một lần: kiểm tra qua băm công khai (mọi peer đọc được). Giữ nguyên định dạng hex thô.
	key := "tallyhash:" + ballotID
	old, err := ctx.GetStub().GetState(key)
	if err != nil {
		return err
	}
	if old != nil {
		return fmt.Errorf("khóa %s đã tồn tại, không ghi đè", key)
	}
	if err := ctx.GetStub().PutPrivateData(privateColl, "tally:"+ballotID, serial); err != nil {
		return err
	}
	h := sha256.Sum256(serial)
	// Chỉ băm được công khai: người không thuộc collection kiểm chứng được qua hàm dưới.
	return ctx.GetStub().PutState(key, []byte(hex.EncodeToString(h[:])))
}

// VerifyTallyPrivate: thành viên collection đọc private data và so với băm công khai.
func (c *BallotChaincode) VerifyTallyPrivate(ctx contractapi.TransactionContextInterface, ballotID string) (bool, error) {
	priv, err := ctx.GetStub().GetPrivateData(privateColl, "tally:"+ballotID)
	if err != nil || priv == nil {
		return false, fmt.Errorf("không đọc được private data của %s", ballotID)
	}
	pub, err := ctx.GetStub().GetState("tallyhash:" + ballotID)
	if err != nil {
		return false, err
	}
	h := sha256.Sum256(priv)
	return hex.EncodeToString(h[:]) == string(pub), nil
}

func main() {
	cc, err := contractapi.NewChaincode(new(BallotChaincode))
	if err != nil {
		panic(err)
	}
	if err := cc.Start(); err != nil {
		panic(err)
	}
}
