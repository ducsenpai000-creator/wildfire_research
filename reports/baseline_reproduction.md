# Baseline reproduction — WACV 2025 (Pesonen et al.), Protocol P

Ngày chốt: 2026-10-18. Kết quả test lấy từ các khóa `S1a_official_v2`, `S1b_E0_v2`, `S1c_E0IN` (`reports/test_results/`). Bảng đầy đủ: `reports/baseline_comparison.csv`. Phân tích sai khác: `reports/reproduction_gaps.md`.

## 1. Kết luận

**Đã tái thực nghiệm thành công** paper WACV 2025 trên Protocol P (AI For Mankind + Boreal Forest Fire Subset C, split của tác giả, 3252 / 701 / 80):

1. **Đánh giá lại checkpoint của tác giả**: khớp tới 3 chữ số mọi số liệu đã công bố (Bảng 2: SAM*, PIDNet-S/M/L; Bảng 4; Bảng 8 test) → dữ liệu, split, tiền xử lý, evaluator đúng.
2. **Train lại PIDNet-S theo đúng quy trình tác giả thực sự dùng (E0-IN)**: test mIoU **0.608** so với 0.594 của paper (+0.014), Boreal 0.697 vs 0.689, AIFM 0.518 vs 0.498. Tất cả chênh lệch nằm trong mức dao động giữa lần chạy (~0.04, ước lượng từ chính Bảng 8 của paper).
3. **Train lại theo cấu hình mặc định của code phát hành (E0)**: **không** tái lập (0.474) — vì code phát hành tắt khởi tạo ImageNet trong khi tác giả có dùng (xác minh bằng pháp y trọng số).

## 2. Bảng kết quả test (80 mask vẽ tay; mIoU theo ảnh, ngưỡng 0.5)

| Dòng | Nguồn | P_all mIoU | Acc | Prec | Rec | F1 | Boreal mIoU | AIFM mIoU |
|---|---|---|---|---|---|---|---|---|
| Paper — SAM→PIDNet-S (Bảng 2 & 4) | paper công bố | 0.594 | 0.960 | 0.891 | 0.645 | 0.707 | 0.689 | 0.498 |
| Checkpoint tác giả PIDNet-S | chạy lại (S1a) | 0.594 | 0.960 | 0.891 | 0.645 | 0.707 | 0.689 | 0.498 |
| **E0-IN** (khởi tạo ImageNet; ckpt best val mIoU, chính) | train lại (S1c) | **0.608** | 0.960 | 0.877 | 0.656 | 0.719 | 0.697 | 0.518 |
| E0-IN (ckpt best val loss, phụ) | train lại (S1c) | 0.593 | 0.963 | 0.859 | 0.626 | 0.699 | 0.719 | 0.466 |
| E0 (code phát hành, không ImageNet; ckpt chính) | train lại (S1b) | 0.474 | 0.966 | 0.678 | 0.524 | 0.561 | 0.738 | 0.210 |
| E0 (ckpt phụ) | train lại (S1b) | 0.436 | 0.964 | 0.651 | 0.477 | 0.512 | 0.730 | 0.141 |
| Pseudo-label SAM so với mask thủ công | chạy lại (S1a) | 0.636 | 0.958 | 0.912 | 0.693 | 0.754 | 0.646 | 0.626 |

Validation (nhãn SAM, cùng routine): checkpoint tác giả 0.540 (paper Bảng 8: 0.540); E0-IN 0.525 (epoch 39); E0 0.376.

## 3. Cấu hình E0-IN và sai khác còn lại

Cấu hình = cấu hình tác giả thực sự dùng (xác minh bằng pháp y checkpoint, `PROTOCOL_LOCK.md` v2): PIDNet-S, khởi tạo ImageNet nhánh P/I, AdamW lr 1e-4, wd 1e-2, 50 epoch, batch 16, input 1080×1920, loss của code (boundary loss ≡ 0, pos_weight 2.5), augmentation của code, fp32.

Sai khác không tránh được:
- Batch 16 = 4 micro-batch × 4 ảnh (VRAM 6 GB) → BatchNorm thống kê trên 4 ảnh.
- Trọng số ImageNet từ bản chuyển đổi OpenMMLab của repo PIDNet (link gốc chết); đã xác minh tương ứng với khởi tạo của tác giả (cosine 0.964).
- Phiên bản thư viện không biết bản tác giả dùng (torch 2.4.1 / CUDA 11.8 ở đây).
- Run bị gián đoạn 2 lần (GPU treo + máy khởi động lại ở epoch 27; tạm dừng theo yêu cầu ở epoch 42) và được resume từ checkpoint trạng thái đầy đủ; chỉ chuỗi ngẫu nhiên augmentation khác với một lần chạy liền mạch.
- Một seed (42). Hai checkpoint được đăng ký trước; cả hai được báo cáo.

## 4. Phát hiện quan trọng (chi tiết trong `reproduction_gaps.md`)

- Văn bản paper ghi lr 1e-3; tác giả thực tế dùng 1e-4 (khớp đúng bit từ weight decay của tham số không nhận gradient).
- Boundary loss l2 luôn bằng 0 trong code **và** trong checkpoint của tác giả → hàng "bỏ l2" và "đầy đủ" ở Bảng 8 là cùng một phép tính; chênh lệch test 0.594 vs 0.633 là nhiễu giữa lần chạy.
- Code phát hành tắt khởi tạo ImageNet; tác giả có dùng.
- Scientific Data báo chất lượng SAM "trên 40 ảnh" = 0.636 → thực ra là số trên 80 ảnh; trên 40 ảnh Boreal là 0.646.

## 5. Phần không tái lập

Teacher BoxSnake (không có code/checkpoint), FPS Jetson Orin NX (không có phần cứng), Bảng 6 (FESB Croatia) và Bảng 7 (AIFM không khói) (thiếu dữ liệu), train lại PIDNet-M/L (thời gian GPU).
