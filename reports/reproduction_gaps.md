# Reproduction gaps — WACV 2025 (Pesonen et al.)

Cập nhật: 2026-10-12. Số liệu: `reports/baseline_comparison.csv`, `reports/test_results/S1a_official_v2.json`, `S1b_E0_v2.json`, `reports/val_diagnostic_official_vs_E0.json`, các file `reports/forensic_*.json`.

## 1. Những gì đã tái lập chính xác

| Hạng mục | Kết quả |
|---|---|
| Split, dữ liệu, tiền xử lý, evaluator | 4 checkpoint chính thức + pseudo-label SAM trên test cho **đúng tới 3 chữ số** mọi chỉ số của Bảng 2 (hàng SAM*, S, M, L), Bảng 4 (S: UAV/AIFM) và Bảng 8 (test hàng 3) |
| Validation | Checkpoint chính thức S / loss-ablation cho val batch-mIoU 0.5397 / 0.5365 vs Bảng 8 0.540 / 0.535 |

→ Mọi sai lệch còn lại **không** đến từ dữ liệu, split hay cách tính metric.

## 2. Sai lệch giữa văn bản paper và code/checkpoint phát hành

| # | Mục | Paper | Code phát hành | Checkpoint tác giả (bằng chứng) | Hệ quả |
|---|---|---|---|---|---|
| G1 | Khởi tạo | ImageNet | **tắt** (`if False:`) | **ImageNet** — conv1 tương quan không gian 0.45; cosine 0.964 với trọng số ImageNet PIDNet-S (OpenMMLab) trên 50 lớp conv; E0 train từ đầu: 0.0006 | **Nguyên nhân chính khiến E0 không đạt**: val 0.376 vs 0.540; test P_all 0.474 vs 0.594 |
| G2 | Learning rate | 1e-3 | 1e-4 | **1e-4** — khớp đúng bit với weight decay đơn thuần trên tham số đầu biên: S = 37 epoch, M = 50, L = 21 (không có nghiệm với 1e-3) | Văn bản paper sai; code đúng |
| G3 | Boundary loss l2 | Gated-SCNN weighted BCE, λ2 = 1 | **≡ 0** (nhãn biên bị chia 255) | **≡ 0** — BN weight đầu biên bằng nhau tuyệt đối, chỉ co theo weight decay; checkpoint "bỏ l2" có giá trị đúng 1.0 | Đầu dự đoán biên không được huấn luyện; BAS-loss (l3) dùng ngưỡng trên đầu ra ngẫu nhiên; hàng "bỏ l2" và "đầy đủ" Bảng 8 cùng phép tính → khác biệt test 0.633 vs 0.594 là **nhiễu giữa lần chạy** |
| G4 | Chọn checkpoint | best val mIoU | `eval.py` nạp best val loss | không xác định trực tiếp; S chọn ở epoch 37/50 | Chúng tôi báo cả hai quy tắc |
| G5 | pos_weight BCE | không nêu | 2.5 | không kiểm chứng được | giữ như code |
| G6 | Augmentation | crop là một lựa chọn | crop không bao giờ chạy; chọn đều 12 lựa chọn | không kiểm chứng được | giữ như code |
| G7 | Kích thước ảnh Boreal (Sci. Data) | 1080×1920 | – | 906 ảnh train + 210 ảnh val là 2048×1080 | loader resize méo nhẹ (như tác giả) |
| G8 | Chất lượng SAM "trên 40 ảnh" (Sci. Data) | 0.636 / 0.958 / 0.912 / 0.693 / 0.754 | – | trên 40 ảnh Boreal: **0.646** / 0.921 / 0.981 / 0.659 / 0.764; số 0.636… là của 80 ảnh | Sci. Data báo nhầm tập |

## 3. Sai lệch do môi trường của chúng tôi

| # | Mục | Ảnh hưởng ước lượng |
|---|---|---|
| H1 | Batch 16 = 4×4 tích lũy gradient (VRAM 6 GB) → BatchNorm trên 4 ảnh | chưa định lượng; cần so E0-IN với checkpoint chính thức |
| H2 | Phiên bản thư viện (torch 2.4.1, torchvision 0.19.1, …) không biết bản tác giả dùng | chưa định lượng |
| H3 | Seed: tác giả không seed torch; nhiễu giữa lần chạy ≥ 0.039 mIoU test (G3) | một lần chạy không đủ để kết luận khác biệt < ~0.04 |
| H4 | E0 bị gián đoạn (khởi động lại máy) và resume từ trạng thái cuối epoch 2 | chỉ đổi luồng ngẫu nhiên augmentation |
| H5 | Trọng số ImageNet lấy từ bản chuyển đổi của OpenMMLab (link gốc chết) | bản này được xác minh tương ứng với khởi tạo của tác giả (cosine 0.964 sau fine-tune); khả năng khác biệt nhỏ về BN stats không loại trừ |

## 4. Chưa tái lập / ngoài phạm vi

- Teacher BoxSnake (Bảng 2 các hàng khác): không có code/checkpoint.
- Bảng 3 (FPS Jetson Orin NX + TensorRT): không có phần cứng.
- Bảng 6 (FESB Croatia) và Bảng 7 (AIFM không khói): chưa có dữ liệu / danh sách ảnh.
- Train lại PIDNet-M/L: không thực hiện (thời gian GPU).

## 5. Trạng thái

- E0 (code-faithful): **không tái lập** được Bảng 2 (0.474 vs 0.594) — nguyên nhân đã xác định (G1).
- E0-IN (khởi tạo ImageNet, khớp quy trình tác giả theo pháp y): **tái lập thành công** — test mIoU 0.608 (paper 0.594), Boreal 0.697 (0.689), AIFM 0.518 (0.498); checkpoint phụ 0.593. Chênh lệch nằm trong nhiễu giữa lần chạy (khóa `S1c_E0IN`, 2026-10-17). Tác động của H1 (BatchNorm trên 4 ảnh) không thấy rõ ở mức chính xác này.
