# Data integrity

Ngày kiểm tra: 2026-10-11. Script: `research_extension/data_tools/build_manifest.py`, `tools/scripts/verify_extract_bff.py`, `tools/scripts/build_protocol_dirs.py`.

## 1. Nguồn và xác thực tải xuống

| Nguồn | Cách tải | Xác thực |
|---|---|---|
| Boreal Forest Fire Subset C (Fairdata, Etsin `1dce1023-…`, DOI 10.23729/fd-72c6cf74-…) | Package dựng sẵn `…_1wvdi0u1.zip` qua `etsin.fairdata.fi/api/v3/download/authorize` (luồng giống web UI) | Zip 479 026 065 B, SHA-256 `5a1e936fea8bc4b8a4ac0e1cc329fd0c321bfb5bd0de567392883676de7fd3ab` = checksum server. **4 392 / 4 392 file khớp SHA-256 từng file trong Metax** (`references/fairdata/metax_v2_files_full.json`); 0 sai, 0 thiếu, 0 thừa. |
| Remixed AI For Mankind (link Google Drive trong README) | `gdown` id `1mbthNoeld-DnfrBxH8Ok9V1w5G52qC3P` | `ai_for_mankind_data.zip` 245 639 861 B, SHA-256 `1f4ff59ad4c0f663f68e7ec6db648ae8145811773ffde0408012c9cf218ab57c`, CRC zip OK. **Không có checksum chính thức** → chỉ xác nhận được số lượng/cấu trúc khớp paper. Bản Kaggle chưa đối chiếu. |

Dữ liệu gốc được giữ nguyên ở `data/raw/bff/Boreal-Forest-Fire-Subset-C/` và `data/raw/aifm/data/`. Thư mục `data/protocol_P/` là **hardlink** của các file gốc (không sao chép nội dung, không sửa, không lọc); xung đột tên → script dừng (không xảy ra).

## 2. Số lượng theo split

### Protocol P (manifest `data/manifests/protocol_P_manifest.csv`, SHA-256 `1e7a04238e4653aa93b0bcc618320a0c3f4c277d5315090d6773cd7f690e9708`)

| | images | labels (bbox) | sam_masks | manual_masks | Paper Bảng 1 (images) |
|---|---|---|---|---|---|
| train | 3252 (AIFM 2068 + BFF 1184) | 3243 | 3218 | – | 3252 (2068 + 1184) ✅ |
| valid | 701 (453 + 248) | 701 | 680 | – | 701 (453 + 248) ✅ |
| test | 80 (40 + 40) | 80 | 80 | 80 | 80 (40 + 40) ✅ |

### Protocol B — Boreal Subset C chính thức (manifest `protocol_B_manifest.csv`, SHA-256 `b3dacb00c77712c35b646a264c3d6c4c966bab0f6ee5c7beb58ac1f7a4d55d9c`)

| | images | labels | sam_masks | manual_masks | Scientific Data |
|---|---|---|---|---|---|
| train | 1184 | 1175 | 1150 | – | 1184 ✅ |
| valid | 248 | 248 | 227 | – | 248 ✅ |
| test | 40 | 40 | 40 | 40 | 40 ✅ |

## 3. Kiểm tra nội dung (CHỈ train/valid)

| Kiểm tra | train | valid |
|---|---|---|
| Kích thước ảnh | AIFM 640×480 (2068); BFF **2048×1080** (906), 1920×1080 (278) | AIFM 640×480 (453); BFF 2048×1080 (210), 1920×1080 (38) |
| Chế độ ảnh | RGB (3252) | RGB (701) |
| SAM mask: kích thước = ảnh | 3218/3218 khớp | 680/680 khớp |
| SAM mask: chế độ / giá trị | `L`, tập giá trị đúng {0,255} cho mọi mask | như train |
| SAM mask rỗng | 0 | 0 |
| Ảnh không có SAM mask (→ loader gán mask rỗng = mẫu âm) | 34: 25 có bbox rỗng (heinola_DJI_0028_*), 9 không có file bbox (evoDJI_0001_frame216–220, ruokolahti_DJI_0087_frame62–65) | 21 (heinola_DJI_0024_*), đều bbox rỗng |
| Bất thường | `evoDJI_0001_frame23`: bbox rỗng nhưng có SAM mask khác rỗng | – |
| Tỉ lệ pixel khói trung bình (median) | BFF 0.224 (0.166); AIFM 0.0052 (0.0012) | BFF 0.304 (0.219); AIFM 0.0094 (0.0014) |

Quy ước lớp: nhị phân, 1 = khói (255 trong PNG), 0 = nền; không có nhãn ignore.

Tiền xử lý (đối chiếu code gốc `datasets/segmentation_data.py`): ảnh → `resize(1080,1920, antialias)` → float/255 → chuẩn hóa ImageNet; mask → `resize(1080,1920, antialias)` trên uint8 → /255 → `round` (nhị phân lại). Ảnh 2048×1080 và 640×480 bị resize méo tỉ lệ về 16:9 — đúng như code gốc.

Kiểm tra batch thật (`reports/figures/d4_train_batch.png`, `d4_valid_batch.png`, `d4_batch_report.json`): ảnh `[B,3,1080,1920]` float32 trong [−2.118, 2.640]; mask ∈ {0,1}; ảnh/mask căn đúng sau augmentation hình học; **nhãn biên ∈ {0, 1/255}** (xem PROTOCOL_LOCK D-2).

## 4. Test-set lock

- Test chỉ được **băm SHA-256 theo byte** và đối chiếu tên file (80/80 ảnh có mask thủ công; 40/40 BFF). Không giải mã, không hiển thị, không thống kê pixel ảnh/mask test.
- Không có thao tác nào: tạo ảnh tổng hợp, nhân bản mẫu, bỏ mẫu khó, lọc theo dự đoán, đổi danh sách mẫu, chia lại split. Augmentation chỉ chạy lúc train (không lưu ra đĩa).
