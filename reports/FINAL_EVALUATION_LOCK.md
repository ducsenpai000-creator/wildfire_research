# FINAL EVALUATION LOCK — Giai đoạn 2 (BRRC)

Viết ngày 2026-10-18, **trước** khi có bất kỳ run BRRC nào hoàn tất. Khóa được tạo tự động bởi hàng đợi (`configs/queue.json`, việc `eval_FINAL`) sau khi mọi run BRRC và seed baseline kết thúc; khi đó trạng thái thực tế (commit, hash) được ghi vào `reports/locks/FINAL_BRRC.json`.

## Những gì được đóng băng trước test

| Mục | Giá trị |
|---|---|
| Code | `research_extension/methods/brrc.py`, `research_extension/repro/train_repro.py`, evaluator (`research_extension/eval/*`) — hash SHA-256 ghi trong file khóa; commit dự án ghi trong file khóa |
| Thiết kế & lưới | `reports/BRRC_DESIGN_PREREGISTRATION.md` (E1a ρ=0.10, E1b ρ=0.25, E2a λ=1, E2b λ=5, α=0.3, β=0.7, E3 = tổ hợp chọn theo val, BRRC-best seeds 7/2026) |
| Quy tắc chọn biến thể | chỉ theo val batch-mIoU (tự động trong `tools/scripts/queue_runner.py`, ghi sự kiện chọn vào `reports/queue_status.json`) |
| Checkpoint mỗi run | best val batch-mIoU (chính) + best val loss (phụ) |
| Seed | 42 cho lưới; 7 và 2026 cho baseline E0-IN-m3 và BRRC-best |
| Ngưỡng | 0.5 (như paper) — không tuning ngưỡng |
| Dữ liệu test | 80 mask thủ công (P_all), tách BFF/AIFM; manifest hash ghi trong khóa |
| Báo cáo | mọi run (kể cả kém), trung bình ± độ lệch chuẩn theo seed; tiêu chí cải thiện như trong bản đăng ký |

## Mô hình được chấm trong khóa

E0-IN-m3 seed 7, seed 2026 (seed 42 đã chấm ở `S2_E0IN_m3` sau khi train — số liệu được gộp vào bảng cuối), BRRC E1a, E1b, E2a, E2b, E3, BRRC-best seed 7, seed 2026 — mỗi run 2 checkpoint đã đăng ký. Run nào FAILED (thiếu checkpoint) được ghi vào `missing_models_not_evaluated` của khóa, không bị loại khỏi báo cáo.

## Sau khi khóa

Kết quả `reports/test_results/FINAL_BRRC.json` là kết quả cuối của vòng nghiên cứu này. Không quay lại thay đổi BRRC rồi báo cáo lại trên cùng test. Mọi nghiên cứu tiếp theo dựa trên kết quả này là vòng mới và cần tập đánh giá độc lập.
