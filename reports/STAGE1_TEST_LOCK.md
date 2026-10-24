# STAGE 1 TEST LOCK

Mục đích: đánh giá test **chỉ cho baseline Giai đoạn 1** (đối chiếu với số công bố của paper). Không có quyết định thiết kế nào của Giai đoạn 2 được phép dựa trên các kết quả này.

## Quy tắc chung (áp dụng cho mọi khóa S1*)

- Evaluator: `research_extension/eval/final_test_eval.py` + `evaluator_core.py` (gọi nguyên `eval()`/`eval_sam()` của repo gốc; tính lại độc lập để đối chiếu). Hash code, checkpoint và manifest được ghi trong file JSON khóa; evaluator từ chối chạy nếu sai hash hoặc nếu khóa đã có kết quả.
- Ngưỡng 0.5 (cố định theo code gốc), không TTA, batch 1, input 1080×1920.
- Tập test: `P_all` (80 ảnh = test Protocol P, đúng Bảng 2), `BFF` (40 ảnh Boreal, dùng cho cột UAV Bảng 4 và kiểm tra mâu thuẫn số SAM* của Scientific Data), `AIFM` (40 ảnh, cột AIFM Bảng 4).
- Chỉ ghi/hiển thị số tổng hợp. Không xuất ảnh, mask, dự đoán hay số liệu theo từng ảnh.
- Kết quả được ghi vào `reports/test_results/<lock_id>.json` và chỉ được chạy **một lần**.

## Khóa S1a_official — `reports/locks/S1a_official.json`

- Mô hình: 4 checkpoint chính thức tải từ link README (`sam_sup_pidnet_s/m/l`, `loss_ablation_pidnet_s`) + đánh giá pseudo-label SAM so với mask thủ công.
- Không có tham số nào do chúng tôi chọn → không thể "tuning trên test".
- Chạy trên CPU (giới hạn luồng) trong khi E0 dùng GPU.

## Khóa S1b_E0 — tạo sau khi E0 kết thúc

- Mô hình: E0 checkpoint best-val-mIoU (chính) và best-val-loss (phụ) — quy tắc đã đăng ký trong `PROTOCOL_LOCK.md` §11 trước khi train.
- Không được train lại/chọn lại epoch sau khi xem kết quả S1b.

## Sự cố 2026-10-12: khóa S1a_official / S1b_E0 bị hủy (VOID)

Lần chạy đầu của hai khóa dừng với `FileNotFoundError` cho chính file khóa (đường dẫn tương đối bị mất sau `chdir` vào repo gốc). Lỗi xảy ra sau bước kiểm tra hash và **trước khi tạo dataset test** → không ảnh/mask test nào được đọc, không có số liệu nào được tính hay hiển thị. Sửa `final_test_eval.py` (dùng đường dẫn tuyệt đối) làm đổi hash code, nên hai khóa cũ được đổi tên `*.VOID.json` và thay bằng `S1a_official_v2`, `S1b_E0_v2` với nội dung mô hình/tập test giống hệt.
