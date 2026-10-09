"""
Cây Merkle cho một lô phiếu. Cùng quy tắc với modules/blockchain-ledger/chaincode/ballot/merkle/merkle.go:
lá = SHA-256 (hex); số lá lẻ thì nhân đôi lá cuối; cặp (a,b) → SHA-256(a||b); rỗng → SHA-256("").
"""
import hashlib


def leaf_hash(ballot_id, pred, conf, image_sha256):
    """Lá của một phiếu: băm bản ghi có thứ tự cố định."""
    return hashlib.sha256(f"{ballot_id}|{pred}|{conf:.4f}|{image_sha256}".encode("utf-8")).hexdigest()


def root(leaves_hex):
    if not leaves_hex:
        return hashlib.sha256(b"").hexdigest()
    level = [bytes.fromhex(h) for h in leaves_hex]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()
