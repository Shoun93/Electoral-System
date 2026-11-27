"""
HASH-CHAIN (chuỗi khối giáo dục, MỘT node) để ghi kết quả phân loại:
  * Mỗi block: index, timestamp, data, prev_hash, nonce, hash (SHA-256, Proof-of-Work nhẹ)
  * Mỗi phiếu là một block: mã phiếu, nhãn dự đoán, độ tin cậy, SHA-256 của ảnh gốc
  * verify(): băm lại toàn chuỗi; phát hiện nếu nội dung bị sửa hoặc liên kết bị đứt
LƯU Ý: đây KHÔNG phải Hyperledger Fabric/PBFT. Chỉ là mô hình băm để kiểm thử logic toàn vẹn.
"""
import hashlib
import json
import time


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class Block:
    def __init__(self, index, data, prev_hash, difficulty, timestamp=None, nonce=0, hash_=None):
        self.index = index
        self.timestamp = time.time() if timestamp is None else timestamp
        self.data = data
        self.prev_hash = prev_hash
        self.difficulty = difficulty
        self.nonce = nonce
        self.hash = hash_ if hash_ is not None else self.mine()

    def compute_hash(self):
        payload = json.dumps({"index": self.index, "timestamp": self.timestamp, "data": self.data,
                              "prev_hash": self.prev_hash, "nonce": self.nonce},
                             sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(payload).hexdigest()

    def mine(self):
        while True:
            h = self.compute_hash()
            if h.startswith("0" * self.difficulty):
                return h
            self.nonce += 1

    def to_dict(self):
        return {"index": self.index, "timestamp": self.timestamp, "data": self.data,
                "prev_hash": self.prev_hash, "nonce": self.nonce, "difficulty": self.difficulty,
                "hash": self.hash}


class HashChain:
    def __init__(self, difficulty=3):
        self.difficulty = difficulty
        self.blocks = [Block(0, {"genesis": True}, "0", difficulty)]

    def add_record(self, data):
        blk = Block(len(self.blocks), data, self.blocks[-1].hash, self.difficulty)
        self.blocks.append(blk)
        return blk

    def add_ballot(self, ballot_id, pred, conf, image_sha256=None):
        return self.add_record({"ballot_id": ballot_id, "pred": pred,
                                "conf": round(float(conf), 4), "image_sha256": image_sha256})

    def validate(self):
        """Trả về (True, None) nếu hợp lệ; (False, chỉ số block lỗi) nếu có sửa đổi/đứt liên kết."""
        # ponytail: không phát hiện được cắt đuôi chuỗi hay đào lại toàn bộ; cần neo hash cuối ra ngoài (Fabric).
        if not self.blocks:
            return False, 0
        prev_hash = "0"
        for i, b in enumerate(self.blocks):
            if (b.index != i or b.prev_hash != prev_hash or b.difficulty != self.difficulty
                    or not b.hash.startswith("0" * self.difficulty) or b.compute_hash() != b.hash):
                return False, i
            prev_hash = b.hash
        return True, None

    def verify(self):
        return self.validate()[0]

    def tamper(self, idx, key, new_value):
        """Dùng để DEMO: giả lập kẻ gian sửa dữ liệu của block idx."""
        self.blocks[idx].data[key] = new_value

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([b.to_dict() for b in self.blocks], f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        chain = cls(difficulty=items[0]["difficulty"] if items else 3)
        chain.blocks = [Block(d["index"], d["data"], d["prev_hash"], d["difficulty"],
                              timestamp=d["timestamp"], nonce=d["nonce"], hash_=d["hash"])
                        for d in items]
        return chain
