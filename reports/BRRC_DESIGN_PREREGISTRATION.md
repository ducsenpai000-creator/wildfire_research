# BRRC-Loss — đăng ký thiết kế và lưới thí nghiệm (trước khi train)

Ngày: 2026-10-18. Căn cứ: `reports/research_gap.md` (G-narrow), `reports/gap_evidence_train_sam_vs_bbox.json` (chỉ train). **Không** dùng bất kỳ số test nào (S1*, S2*) cho các lựa chọn dưới đây.

## 1. Bản chất đóng góp (trung thực)

BRRC = **tổ hợp hai kỹ thuật đã có tiền lệ**: (i) giảm độ tin cậy nhãn quanh biên (SVLS, ignore-band VOC, Simple Does It) và (ii) loss cân bằng FN/FP (Tversky, Focal-Tversky). Điểm khác được kiểm định: (i) được áp **một phía** (chỉ pixel nền ngay ngoài mask SAM, vì pseudo-label SAM bảo thủ: P 0.912 ≫ R 0.693 công bố) và với **bán kính tỉ lệ theo kích thước vùng khói** (dải cố định 8 px chiếm 65 % pixel khói AIFM trên train). Không tuyên bố là thuật toán mới.

## 2. Định nghĩa

Áp lên PIDNet-S, cấu hình **E0-IN-m3** giữ nguyên (ImageNet, AdamW 1e-4, wd 1e-2, 50 epoch, batch 16 = micro 3, pos_weight 2.5, augmentation của code). Nhãn gốc **không** bị sửa; chỉ thay trọng số loss.

**Thành phần 1 — Boundary reliability weighting (BR).** Với mỗi ảnh train (sau augmentation, ở 1080×1920), gọi M là mask SAM, A = |M| số pixel khói.
- Bán kính theo ảnh: r = clip(ρ·√A, r_min, r_max), r_min = 3 px, r_max = 40 px; ảnh không có khói → không áp.
- Khoảng cách d(p) của pixel nền p tới vùng khói: khoảng cách chessboard, tính trên GPU bằng giãn nở (max-pool 3×3) lặp lại.
- Trọng số: w(p) = w_min + (1 − w_min)·d(p)/r nếu p là nền và d(p) ≤ r; w(p) = 1 cho mọi pixel khác (khói, nền xa). w_min = 0.
- Áp lên hai số hạng BCE phân đoạn của baseline (l0 head P và l1 head chính): BCE có trọng số chuẩn hóa bằng tổng trọng số. l2 (≡ 0) và l3 (BAS) giữ nguyên.

**Thành phần 2 — Recall-calibrated Tversky (RC).** Trên xác suất head chính p = σ(logit), theo từng ảnh:
L_T = 1 − (TP + 1)/(TP + α·FP + β·FN + 1), TP = Σp·y, FP = Σp·(1−y), FN = Σ(1−p)·y (mềm), trung bình trên các ảnh của micro-batch. α = 0.3, β = 0.7 (Salehi et al. 2017; β > α ưu tiên recall). Tổng: L = L_baseline(BR?) + λ_T·L_T.

## 3. Lưới thí nghiệm (cố định trước; mỗi run 50 epoch, seed 42)

| ID | BR | RC | Ghi chú |
|---|---|---|---|
| E0-IN-m3 | – | – | baseline (đang chạy trong hàng đợi) |
| E1a | ρ = 0.10 | – | |
| E1b | ρ = 0.25 | – | |
| E2a | – | λ_T = 1 | |
| E2b | – | λ_T = 5 | |
| E3 | ρ* | λ_T* | ρ*, λ_T* = biến thể có val batch-mIoU cao nhất trong E1a/E1b và E2a/E2b |
| E0-IN-m3 seed 7, seed 2026 | – | – | ước lượng nhiễu |
| BRRC-best seed 7, seed 2026 | | | BRRC-best = run có val batch-mIoU cao nhất trong {E1*, E2*, E3} |

Tổng: 9 run mới × ~11 giờ ≈ 100 giờ GPU (≤ 5 GB VRAM).

## 4. Quy tắc chọn và báo cáo (cố định trước)

- Checkpoint của mỗi run: best val batch-mIoU (chính) và best val loss (phụ) — như baseline.
- Chọn biến thể chỉ bằng validation (val batch-mIoU). **Giới hạn đã biết**: val là nhãn SAM, nên quy tắc này thiên vị chống lại biến thể dự đoán nhiều khói hơn SAM; vẫn giữ quy tắc vì không có val sạch trong protocol.
- Test: **một lần duy nhất** ở `reports/FINAL_EVALUATION_LOCK.md` cho tất cả run BRRC + baseline seeds, sau khi mọi run xong. Báo cáo mọi run (kể cả kém), trung bình ± độ lệch chuẩn theo seed, và từng nguồn (BFF/AIFM).
- Tiêu chí "cải thiện": trung bình test mIoU của BRRC-best (3 seed) cao hơn trung bình E0-IN-m3 (3 seed) và chênh lệch lớn hơn độ lệch chuẩn giữa seed; recall tăng đồng thời. Nếu không đạt → báo kết quả âm.
