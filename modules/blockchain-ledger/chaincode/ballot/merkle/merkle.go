// Package merkle tính gốc Merkle của một lô phiếu. Thư viện chuẩn, không phụ thuộc Fabric.
// Quy tắc (phải khớp với src/merkle.py của pipeline):
//   - lá là mã băm SHA-256 (hex) của bản ghi phiếu
//   - số lá lẻ thì nhân đôi lá cuối; cặp (a,b) → SHA-256(a || b)
//   - không có lá nào → SHA-256 của chuỗi rỗng
package merkle

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
)

// Root trả về gốc Merkle (hex) của danh sách lá dạng hex.
func Root(leavesHex []string) (string, error) {
	if len(leavesHex) == 0 {
		e := sha256.Sum256(nil)
		return hex.EncodeToString(e[:]), nil
	}
	level := make([][]byte, 0, len(leavesHex))
	for _, h := range leavesHex {
		b, err := hex.DecodeString(h)
		if err != nil || len(b) != sha256.Size {
			return "", errors.New("lá không phải SHA-256 hex: " + h)
		}
		level = append(level, b)
	}
	for len(level) > 1 {
		if len(level)%2 == 1 {
			level = append(level, level[len(level)-1])
		}
		next := make([][]byte, 0, len(level)/2)
		for i := 0; i < len(level); i += 2 {
			s := sha256.Sum256(append(append([]byte{}, level[i]...), level[i+1]...))
			next = append(next, s[:])
		}
		level = next
	}
	return hex.EncodeToString(level[0]), nil
}
