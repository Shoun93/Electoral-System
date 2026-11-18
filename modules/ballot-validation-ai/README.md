# Phân loại phiếu bầu: mô hình lai CNN theo ô + C4.5, ghi kết quả lên hash-chain

Pipeline chạy end-to-end trên dữ liệu **giả lập** (2500 ảnh thẻ cử tri):
sinh ảnh → tiền xử lý → CNN phát hiện dấu trên từng ô → C4.5 ra quyết định (có luật IF-THEN)
→ đánh giá trên tập test → ghi kết quả từng phiếu lên chuỗi khối cục bộ.

> Trạng thái: **demo nghiên cứu**. Chưa kiểm chứng trên ảnh scan thật. Chaincode Hyperledger Fabric đã viết nhưng chưa chạy trên mạng thật. PBFT chưa làm.

## 1. Kiến trúc

```
Ảnh phiếu (grayscale 1200×850)
   │  CLAHE + Otsu ở độ phân giải gốc
   ├──► 8 đặc trưng thủ công (số ô có dấu, mật độ ô, dải trái ô, vùng ghi thêm, ...)
   └──► 5 crop ô (48×48) ──► CNN phát hiện dấu (out-of-fold cho tập train)
                                   │  3 thống kê: số ô có dấu, xác suất TB, xác suất max
   ┌───────────────────────────────┘
   ▼
C4.5 (src/c45.py: gain ratio, tách ngưỡng nhị phân, cắt tỉa pessimistic)
   ▼
Nhãn: valid | blank | erased | wrong_mark | multi_mark  + luật IF-THEN
   ▼
Hash-chain: mỗi phiếu là một block (nhãn, độ tin cậy, SHA-256 ảnh gốc)
```

Ngoài ra có CNN toàn ảnh 224×224 làm **đối chứng**; mô hình này không học được (xem mục 5).

## 2. Cấu trúc thư mục

```
ballot-validation-ai/
├── main.py                 # pipeline end-to-end
├── configs/config.yaml     # siêu tham số và đường dẫn
├── requirements.txt
├── src/
│   ├── config.py           # đọc config, đường dẫn tính theo vị trí file
│   ├── preprocess.py       # binarize, crop ô, tạo data/processed/dataset.npz
│   ├── feature_c45.py      # 8 đặc trưng thủ công
│   ├── c45.py              # cài đặt C4.5 bằng numpy
│   ├── train_c45.py        # huấn luyện C4.5, in luật
│   ├── train_cnn.py        # CNN toàn ảnh (đối chứng) + CNN phát hiện dấu theo ô
│   ├── baselines.py        # majority, Naive Bayes, kNN
│   ├── evaluate.py         # báo cáo 5 lớp, nhị phân hợp lệ/không hợp lệ, ma trận nhầm lẫn
│   ├── hashchain.py        # chuỗi khối (SHA-256 + PoW nhẹ), lưu và xác minh
│   ├── merkle.py           # gốc Merkle cho lô phiếu (cùng quy tắc với chaincode Go)
│   └── monitor.py          # PSI, log lịch sử, cảnh báo suy giảm
├── tests/test_units.py
└── data/ models/ results/  # sinh ra khi chạy, không đưa lên git
```

Bộ sinh dữ liệu nằm ngoài module: `dataset/generate_dataset.py` (dùng chung cho repo).

## 3. Cài đặt và chạy

Trên máy local, từ thư mục gốc repo:

```bash
pip install -r modules/ballot-validation-ai/requirements.txt
python dataset/generate_dataset.py --out modules/ballot-validation-ai   # 2500 ảnh
cd modules/ballot-validation-ai
python main.py
python -m unittest discover -s tests -v
```

Thử nhanh với ít mẫu: `--samples-per-class 20`.

Trên Google Colab: bật GPU (Runtime → Change runtime type → T4 GPU) trước khi clone. Không cài lại `torch`, Colab đã có sẵn.

Sinh dữ liệu không cần GPU. Huấn luyện CNN nên chạy trên GPU.

## 4. Dữ liệu

`dataset/generate_dataset.py` sinh 2500 ảnh, 500 mỗi lớp:

| Nhãn | Ý nghĩa |
|---|---|
| `valid` | chọn 1–3 đại biểu |
| `blank` | phiếu trắng hoặc chỉ có nhiễu nhỏ |
| `erased` | gạch xóa toàn bộ danh sách |
| `wrong_mark` | dấu đặt ngoài ô, hoặc ghi thêm tên ngoài danh sách |
| `multi_mark` | chọn 4–5 đại biểu (nhiều hơn quy định) |

Đặc điểm:
- **Nhãn suy ra từ hình học.** Mỗi dấu được render từ spec, rồi hàm `oracle()` tính nhãn từ mask. Mẫu nào oracle không khớp nhãn mục tiêu sẽ bị loại và sinh lại.
- **4 mức khó** (easy / medium / hard / boundary) với tỉ lệ 20/30/30/20%. Ngưỡng định nghĩa nhãn (ví dụ dấu được tính vào ô khi ≥90% mực nằm trong ô) là quy ước của nhóm, không phải chuẩn bầu cử.
- **Một group = một phiếu gốc**, có đúng một mẫu mỗi lớp. Chia train/test theo group, nên không rò rỉ template giữa hai tập.
- **Phân chia:** train 1750 / val 250 / test 500.
- Mỗi ảnh có tọa độ ROI đã xoay (`data/labels/boxes.json`), metadata từng mẫu (`data/metadata/samples.jsonl`), và kết quả kiểm tra (`data/metadata/validation.json`).

Toàn bộ thông tin cá nhân trên thẻ là **giả lập**.

## 5. Kết quả (tập test, 500 phiếu, seed 42, một lần chạy)

Tập test đã được dùng nhiều lần để khám phá. Các số dưới đây **không** là bằng chứng nghiệm thu. Nghiệm thu cần tập giữ lại mới (mục 5.1).

| Mô hình | Accuracy | Macro-F1 | Recall "không hợp lệ" | Precision "không hợp lệ" |
|---|---|---|---|---|
| Baseline majority | 0.200 | 0.067 | 0.000 | 0.000 |
| Baseline Naive Bayes | 0.804 | 0.784 | 0.903 | 0.995 |
| Baseline kNN (k=5) | 0.920 | 0.917 | 0.943 | 0.987 |
| C4.5 (đặc trưng thủ công) | 0.906 | 0.901 | 0.928 | 0.992 |
| CNN toàn ảnh 224×224 (đối chứng, không học) | 0.200 | 0.067 | 1.000* | 0.800* |
| **Mô hình lai (CNN theo ô + C4.5)** | **0.926** | **0.923** | **0.958** | **1.000** |

\* CNN toàn ảnh chỉ dự đoán `blank` cho mọi mẫu. `blank` thuộc nhóm không hợp lệ, nên recall 1.0 và precision 0.8 không có ý nghĩa phân loại.

Selective (mô hình lai, `results/metrics.json` → `hybrid_selective`):
- Cấu hình chọn trên OOF Train: `min_leaf=10`, `cf=0.25`, `leaf_min_n=0`, τ_CNN=0.58, τ_C4.5=0.92. Không cấu hình nào đạt đủ ba ràng buộc trên OOF.
- Temperature T = 23.32 (`std_fallback`). OOF Train phân tách hoàn hảo (0/8750 ô sai), nên NLL không xác định và T được đặt theo độ lệch chuẩn logit.
- Trên Test: Invalid Screening Recall 1.000 (Clopper–Pearson 95% một phía, cận dưới 0.9925), Coverage 0.726, Referral Rate 0.274, Selective Risk 0.044 (cận trên 0.066). **Không đạt** (Coverage < 0.80 đề xuất, Selective Risk > 0.02).

CNN phát hiện dấu theo ô đạt accuracy 0.9952 trên tập test (cấp ô).

**Đọc kết quả thận trọng:**
- Mô hình lai cao hơn C4.5 thuần khoảng 2 điểm phần trăm trên 500 mẫu. Chưa có nhiều seed hay kiểm định thống kê.
- Lớp khó nhất là `wrong_mark`. Lỗi chủ yếu nhầm sang `valid` và `blank`.
- CNN toàn ảnh không học được: loss đứng ở ln5 ≈ 1.6096, val_acc 0.20. Nguyên nhân chưa được kiểm chứng.
- Dữ liệu là giả lập, nhãn sinh bằng quy tắc. Kết quả không chứng minh hiệu năng trên phiếu thật.

### 5.1 Tập giữ lại (chưa chạy)

Tập giữ lại dùng seed mới, không dùng để chọn cấu hình. Chạy đúng một lần, sau khi ghi sha256 cấu hình và manifest.

```bash
cd /path/to/electoral-system
python dataset/generate_dataset.py --seed 101 --out $HOME/holdout_101     # gốc dữ liệu riêng
cd modules/ballot-validation-ai
BALLOT_ROOT=$HOME/holdout_101 python main.py                            # đọc data/ và ghi results/ trong gốc riêng
```

`BALLOT_ROOT` đổi gốc cho `data/`, `models/`, `results/`. Mã nguồn và `configs/config.yaml` vẫn lấy từ repo.

## 6. Blockchain (mức mô phỏng)

- `results/hashchain.json`: 502 block (1 block neo mã băm mô hình và manifest, 1 block genesis, 500 block phiếu). Kiểm chứng ban đầu: chuỗi hợp lệ.
- Demo: sửa một block trong bản sao thì `validate()` trả về `False` và chỉ ra block bị lỗi.
- `results/fabric_batch.json`: gốc Merkle của lô phiếu, để chaincode `CommitBatch` / `VerifyBatch`.
- Chaincode Go (`modules/blockchain-ledger/`): kiểm tra MSP, ghi một lần, private data qua transient map, verify Merkle. Cây Merkle Go và Python cùng vector kiểm thử. Chaincode **chưa biên dịch với thư viện Fabric và chưa chạy trên mạng**.

## 7. MLOps (mức tối thiểu)

- `results/metrics_log.json`: lịch sử các lần chạy. Hiện mới có một lần, nên chưa so sánh được suy giảm.
- `check_drift()`: cảnh báo nếu độ chính xác lần mới thấp hơn trung vị các lần trước quá 5 điểm.
- `monitor.psi()`: đo lệch phân phối đặc trưng giữa train và dữ liệu sản xuất. Chưa có dữ liệu sản xuất để chạy thật.

## 8. Giới hạn

- **Dữ liệu giả lập**, nhãn sinh bằng quy tắc, chưa có người gán nhãn độc lập.
- **Chưa có ảnh scan thật** (bóng, tẩy xóa, nét không đều, chữ ký).
- **Một lần chạy, một seed.** Chưa có khoảng tin cậy.
- **Blockchain cục bộ** là chuỗi khối một node để kiểm thử logic toàn vẹn, không phải Fabric.
- **Fabric và PBFT** (SmartBFT) chưa triển khai và chưa chạy thử.
- **MLOps** chưa có DVC, MLflow, CI/CD, Docker Compose.
- **Chưa có web demo**; chưa đo TPS và độ trễ.
- **Bảo mật:** phân tích mối đe dọa (STRIDE) chưa viết.
- Lịch sử git cũ vẫn chứa `models/` và `results/` đã từng được commit.