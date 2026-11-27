package main

import (
	"strings"
	"testing"

	"github.com/hyperledger/fabric-chaincode-go/v2/pkg/cid"
	"github.com/hyperledger/fabric-chaincode-go/v2/shim"
	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

type fakeStub struct {
	shim.ChaincodeStubInterface
	state, private map[string][]byte
}

func (s fakeStub) GetState(k string) ([]byte, error)          { return s.state[k], nil }
func (s fakeStub) PutState(k string, v []byte) error          { s.state[k] = v; return nil }
func (s fakeStub) PutPrivateData(_, k string, v []byte) error { s.private[k] = v; return nil }
func (s fakeStub) GetPrivateData(_, k string) ([]byte, error) { return s.private[k], nil }
func (s fakeStub) GetTransient() (map[string][]byte, error) {
	return map[string][]byte{"serial": s.state["_serial"]}, nil
}

type fakeID struct{ cid.ClientIdentity }

func (fakeID) GetMSPID() (string, error) { return mspCountingBoard, nil }

func newCtx() (*contractapi.TransactionContext, fakeStub) {
	s := fakeStub{state: map[string][]byte{"_serial": []byte("s-1")}, private: map[string][]byte{}}
	ctx := &contractapi.TransactionContext{}
	ctx.SetStub(s)
	ctx.SetClientIdentity(fakeID{})
	return ctx, s
}

func TestTallyPrivateIsWriteOnce(t *testing.T) {
	ctx, s := newCtx()
	c := &BallotChaincode{}
	if err := c.PutTallyPrivate(ctx, "b1"); err != nil {
		t.Fatal(err)
	}
	s.state["_serial"] = []byte("s-2")
	if err := c.PutTallyPrivate(ctx, "b1"); err == nil {
		t.Fatal("ghi đè private data phải bị từ chối")
	}
	if string(s.private["tally:b1"]) != "s-1" {
		t.Fatal("private data đã bị đổi")
	}
	if ok, err := c.VerifyTallyPrivate(ctx, "b1"); err != nil || !ok {
		t.Fatal(ok, err)
	}
}

func TestRecordBallotValidatesInput(t *testing.T) {
	ctx, _ := newCtx()
	c := &BallotChaincode{}
	sha := strings.Repeat("ab", 32)
	for _, a := range [][4]string{
		{"", "valid", "0.9", sha}, {"b", "abc", "0.9", sha}, {"b", "valid", "NaN", sha},
		{"b", "valid", "1.5", sha}, {"b", "valid", "0.9", "xyz"},
	} {
		if c.RecordBallot(ctx, a[0], a[1], a[2], a[3]) == nil {
			t.Fatalf("phải từ chối %v", a)
		}
	}
	if err := c.RecordBallot(ctx, "b", "PENDING_REVIEW", "0.9", sha); err != nil {
		t.Fatal(err)
	}
	if c.CommitBatch(ctx, "x", sha, 0) == nil || c.CommitBatch(ctx, "x", "zz", 1) == nil {
		t.Fatal("lô sai phải bị từ chối")
	}
}
