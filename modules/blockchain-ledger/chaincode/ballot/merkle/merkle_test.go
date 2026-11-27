package merkle

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

func leaf(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

// Vector này cũng được kiểm trong tests/test_units.py của pipeline Python.
func TestVector(t *testing.T) {
	leaves := []string{leaf("a"), leaf("b"), leaf("c")}
	got, err := Root(leaves)
	if err != nil {
		t.Fatal(err)
	}
	const want = "d31a37ef6ac14a2db1470c4316beb5592e6afd4465022339adafda76a18ffabe"
	if got != want {
		t.Fatalf("root=%s want=%s", got, want)
	}
}

func TestSingleLeafIsItself(t *testing.T) {
	l := leaf("x")
	got, _ := Root([]string{l})
	if got != l {
		t.Fatalf("got %s want %s", got, l)
	}
}

func TestRejectsBadLeaf(t *testing.T) {
	if _, err := Root([]string{"zz"}); err == nil {
		t.Fatal("expected error")
	}
}
