# PROTOCOL LOCK — Giai đoạn 0/1

- Ngày khóa: 2026-10-11 (giờ máy UTC+7)
- Người thực hiện: AI research engineer (Claude) trên máy của người dùng
- Trạng thái: **KHÓA cho baseline E0**. Mọi thay đổi sau ngày này phải ghi thành phiên bản mới ở cuối file, không sửa đè.

## 1. Nguồn đã đọc

| Nguồn | Cách truy cập | Ghi chú |
|---|---|---|
| WACV 2025, Pesonen et al., *Detecting Wildfires on UAVs with Real-Time Segmentation Trained by Larger Teacher Models*, pp. 5166–5176 | PDF open access CVF, `references/papers/wacv2025_pesonen.pdf` (+ `.txt`) | Bản đọc gồm 11 trang (thân bài + tài liệu tham khảo). **Supplementary (Bảng 9–17, Eq. 3–6) chưa đọc được** — xem §9. |
| Scientific Data 12 (2025), DOI 10.1038/s41597-025-05634-0 (Boreal Forest Fire dataset) | Full text JATS từ Europe PMC (PMC12350734), `references/papers/scidata2025_bff.xml/.txt` | nature.com yêu cầu chuyển hướng đăng nhập; dùng bản PMC. |
| Repo chính thức `gitlab.com/fgi_nls/public/wildfire-real-time-segmentation` | `git clone` 2026-10-11T23:08+07:00, branch `main`, HEAD `2382c295ff3e6185d9b3ae82cfb200f8d23d8fa6` (2025-08-14), không có tag, working tree sạch | Đã đọc toàn bộ: `README.md`, `train.py`, `eval.py`, `infer.py`, `generate_pseudo_labels.py`, `datasets/segmentation_data.py`, `models/pidnet.py`, `pidnet_utils/{utils,criterion,configs/default}.py`, `utils/*`. `pidnet_utils/function.py` được import nhưng hàm `train/validate` của nó bị hàm cục bộ trong `train.py` che khuất (không dùng). |
| Fairdata dataset `fd-72c6cf74-...` (Etsin id `1dce1023-493a-4d63-a906-f2a44f831898`) | Metax v2 file list (14 592 file, SHA-256 từng file) lưu tại `references/fairdata/metax_v2_files_full.json` | License CC BY 4.0. |

**Thay đổi repo sau khi paper công bố:** từ commit `fca70e4` (2024-08-21, thời điểm arXiv) đến HEAD chỉ đổi `README.md`, `utils/plot_outputs.py` (+2 dòng), `utils/vis_data.py` (mới), `data_vis/*`. **Không có thay đổi nào ở code train/eval/model/loss/dataset.**

## 2. Mục tiêu tái thực nghiệm chính

- Paper mục tiêu: **WACV 2025 (Pesonen et al.)**.
- Hàng mục tiêu chính: **Bảng 2, Teacher = SAM, Student = PIDNet-S: test mIoU 0.594, Acc 0.960, Prec 0.891, Rec 0.645, F1 0.707** (80 ảnh test thủ công).
- Hàng mục tiêu phụ: Bảng 4 (SAM → PIDNet-S: UAV 0.689, AIFM 0.498); Bảng 2 hàng `SAM*` (pseudo-label so với mask thủ công: 0.636/0.958/0.912/0.693/0.754); Bảng 8 (cấu hình đầy đủ: val 0.540 / test 0.594).
- Kiến trúc chạy đầu tiên: **PIDNet-S** (`get_seg_model`, `planes=32, ppm_planes=96, head_planes=128, m=2, n=3`, `num_classes=1`), 7 716 291 tham số (paper Bảng 3: 7.72M ✓).

## 3. Cấu hình: paper ↔ code phát hành ↔ chạy thực tế (E0)

| Mục | Paper (văn bản) | Code phát hành (HEAD 2382c29) | E0 thực tế | Lý do sai khác |
|---|---|---|---|---|
| Backbone/khởi tạo | "pretrained on ImageNet" | `get_seg_model`: khối nạp ImageNet nằm trong `if False:` → **train từ đầu** (Kaiming init) | Train từ đầu (như code) | Trọng số ImageNet của PIDNet (Google Drive của tác giả PIDNet) **không còn truy cập được** (gdown: link file & thư mục lỗi quyền); và code phát hành vốn không nạp chúng. |
| Optimizer | AdamW, lr **1e-3**, wd 1e-2 | AdamW `lr=1e-4`, wd mặc định **1e-2** | AdamW lr 1e-4, wd 1e-2 | E0 bám code. Biến thể lr 1e-3 (E0-lr3) là thí nghiệm phụ, đăng ký trước, chạy nếu đủ thời gian GPU. |
| Scheduler | không nêu | không có | không có | — |
| Epoch | 50 | 50 | 50 | — |
| Batch | 16 | 16 | **batch logic 16 = 4 micro-batch × 4 (tích lũy gradient)** | 6 GB VRAM: fp32 bs4 = 4.82 GiB, bs8 = 9.5 GiB (đo thật). Thành phần batch, xáo trộn và trọng số loss giống hệt; **BatchNorm tính thống kê trên 4 ảnh thay vì 16**; `weighted_bce` (l2) và BAS tính trên micro-batch. |
| Input | 1080×1920 | resize mọi ảnh về 1080×1920 (`antialias=True`), chuẩn hóa ImageNet mean/std | như code | — |
| Loss | λ0=0.4 (l0 BCE), λ1=20 (l1 BCE), λ2=1 (boundary, Gated-SCNN), λ3=1 (BAS, t=0.8) | như paper + `BCEWithLogitsLoss(pos_weight=2.5)` cho l0,l1,l3; `BondaryLoss` hệ số nội 20 | như code | `pos_weight=2.5` **không có trong paper**. **l2 ≡ 0 trong code** (xem §6, phát hiện D-2). |
| Augmentation | kiểu TrivialAugment: chọn 1 phép ngẫu nhiên đồng xác suất từ {crop, vflip, rotate, perspective, erasing, grayscale, blur, invert, sharpness, color jitter} + hflip 50% | `np.random.choice(list, 1, p)` → `p` rơi vào tham số `replace` → **chọn đều 12 lựa chọn**; khóa `'crop'` không khớp `'random_crop'` → **crop không bao giờ được áp dụng** (thành no-op) | như code (không sửa) | Ghi nhận là sai khác paper↔code (D-4, D-5). |
| Seed | không nêu | `np.random.seed(42)`; torch không seed | numpy 42 + `torch.manual_seed(42)` + `random.seed(42)` | Thêm seed torch để truy vết; không đổi phân phối. Không bật deterministic cuDNN. |
| Chọn checkpoint | theo **val mIoU** tốt nhất | `train.py` lưu cả `exp.pt` (val loss nhỏ nhất) và `exp_miou.pt` (val "mIoU" lớn nhất); **`eval.py` nạp `exp.pt` (val loss)** | Lưu cả hai. **Chính: best val mIoU (theo paper). Phụ: best val loss (theo code)**. Cả hai được đăng ký trước và đều báo cáo. | Mâu thuẫn paper↔code, không xác minh được bản nào sinh ra checkpoint công bố. |
| "val mIoU" khi train | — | `BinaryJaccardIndex` trên **toàn batch 16** (pixel gộp), batch không có pixel dương → 1.0; trung bình theo batch | tái hiện y hệt (forward theo micro-batch ở eval mode rồi ghép lại → chính xác) + log thêm mIoU theo từng ảnh | Chỉ số chọn checkpoint phụ thuộc batch size → giữ batch val = 16. |
| "val loss" khi train | — | chỉ thành phần l1 (BCE pos_weight 2.5) | y hệt | — |
| Độ chính xác số | — | fp32 | fp32 (không AMP) | — |
| cuDNN benchmark | — | không áp dụng (config có `BENCHMARK=True` nhưng không dùng) | không bật | đo được chỉ nhanh hơn ~6%. |

Bảng chi tiết dạng CSV: `configs/paper_config_vs_actual_config.csv`.

## 4. Nguồn dữ liệu paper yêu cầu

| Nguồn | Dùng cho | Trạng thái |
|---|---|---|
| Boreal Forest Fire **Subset C** (Fairdata, CC BY 4.0) | train/val/test UAV | ✅ Package Subset C (479 026 065 B, SHA-256 `5a1e936f…d3ab` khớp checksum server) → **4 392/4 392 file khớp SHA-256 Metax từng file** |
| "Remixed" Wildfire Smoke Dataset v1/v2 (AI For Mankind & HPWREN) — bản sắp xếp lại của tác giả, README trỏ Google Drive `1mbthNoeld-…` và Kaggle `juliuspesonen/remixed-wildfire-smoke` | train/val/test AIFM | ✅ Google Drive zip 245 639 861 B, SHA-256 `1f4ff59a…b57c`, CRC zip OK. **Không có checksum chính thức** để đối chiếu; bản Kaggle chưa đối chiếu (cần đăng nhập). |
| Croatian Center for Wildfire Research (FESB) 49 ảnh, dùng 41 ảnh nhãn "maybe smoke" | Bảng 6 (tổng quát hóa) | ❌ Chưa lấy; paper không cho danh sách 8 ảnh bị loại → **không tái lập chính xác được**. Ngoài phạm vi Giai đoạn 1. |
| AI For Mankind *classification* dataset, 623 ảnh không khói | Bảng 7 (FP rate) | ❌ Chưa lấy; paper không cho danh sách ảnh/định nghĩa FP rate chi tiết (có thể ở supplementary). Ngoài phạm vi Giai đoạn 1. |
| SAM ViT-H checkpoint | sinh pseudo-label | Không cần: pseudo-label SAM đã được phát hành kèm dữ liệu (`sam_masks/`). Không tái sinh mask. |

## 5. Danh sách mẫu và split

Split do tác giả phát hành (không chia lại). Khớp Bảng 1 WACV:

| Split | AIFM | BFF (Subset C) | Tổng | Paper Bảng 1 |
|---|---|---|---|---|
| train | 2068 | 1184 | 3252 | 2068 / 1184 / 3252 ✅ |
| valid | 453 | 248 | 701 | 453 / 248 / 701 ✅ |
| test (mask thủ công) | 40 | 40 | 80 | 40 / 40 / 80 ✅ |

Cách paper chia: BFF dùng khoảng cách ghi hình ~120 s giữa các khung để tránh trùng lặp; AIFM chia sao cho không chung vị trí/hướng nhìn (phân cụm + đánh giá thị giác). Manifest đầy đủ kèm SHA-256: `data/manifests/protocol_P_manifest.csv` (SHA-256 manifest `1e7a0423…9708`), `data/manifests/protocol_B_manifest.csv` (`b3dacb00…5d9c`).

## 6. Phát hiện quan trọng về code (đã kiểm chứng bằng chạy thật)

- **D-1 Không nạp ImageNet**: `models/pidnet.py:193` `if False:`.
- **D-2 Boundary loss l2 luôn bằng 0**: biên được tạo bằng Canny+dilate với giá trị {0,1}, ép `uint8`, rồi `to_dtype(float32, scale=True)` chia 255 → nhãn biên ∈ {0, 1/255}. `weighted_bce` đếm `target==1` → 0 pixel dương → mọi trọng số = 0 → **l2 = 0.0**. Kiểm chứng trên batch train/valid thật (`reports/figures/d4_batch_report.json`: `l2_boundary_loss_upstream_target = 0.0`, `weighted_bce_pos_count_upstream = 0`; nếu nhị phân hóa thì l2 ≈ 3.3–5.3). Hệ quả: nhánh dự đoán biên (`seghead_d`) không nhận gradient; l3 (BAS) dùng ngưỡng `sigmoid(bd)>0.8` từ một đầu ra không được huấn luyện. Hàng "bỏ l2" và hàng "đầy đủ" ở Bảng 8 vì vậy là **cùng một phép tính**, chỉ khác ngẫu nhiên: val 0.535 vs 0.540, **test 0.633 vs 0.594** → ước lượng nhiễu giữa lần chạy trên test ≈ **0.04 mIoU**. (Giả định: tác giả dùng đúng code đã phát hành.)
- **D-3 Chọn checkpoint**: paper = val mIoU; `eval.py` = val loss (`exp.pt`).
- **D-4 Xác suất augmentation**: `p` không được truyền đúng → đều 12 lựa chọn (trùng ý "same probability" của paper nhưng khác ý định trong code: `none` 0.1).
- **D-5 Crop không bao giờ chạy** (key `'crop'` ≠ `'random_crop'`).
- **D-6 `pos_weight=2.5`** cho BCE không có trong paper.
- **D-7 lr** 1e-4 (code) vs 1e-3 (paper).
- **D-8 IoU khi GT rỗng = 1** bất kể dự đoán (cả val và test). Ảnh test đều có file mask thủ công; chưa kiểm tra mask test nào rỗng (khóa test).

## 7. Metric chính thức (theo code `eval.py`, khớp Eq. 1 của paper)

- Test: `WFSeg(mode='test', manual_masks=True)` → chỉ ảnh có mask trong `manual_masks/test` (80 ảnh), `batch_size=1`, resize ảnh & mask về 1080×1920 (mask: resize antialias rồi `round`), đầu ra upsample bilinear `align_corners=True`.
- **mIoU** = trung bình theo ảnh của IoU lớp khói, `BinaryJaccardIndex(threshold=0.5)` trên `sigmoid(logit)`; ảnh GT rỗng → IoU = 1.
- **Accuracy/Precision/Recall/F1**: `torcheval` binary trên ảnh đã làm tròn `round(sigmoid)`, tính **theo từng ảnh rồi lấy trung bình** (macro theo ảnh), không phải gộp pixel.
- Không có lớp ignore. Ngưỡng 0.5 cố định (không tuning ngưỡng — paper không mô tả tuning ngưỡng).
- Breakdown theo nguồn (Bảng 4): đánh giá riêng trên `data/raw/bff/Boreal-Forest-Fire-Subset-C` (40 ảnh BFF) và `data/raw/aifm/data` (40 ảnh AIFM) bằng cùng evaluator.

## 8. Phạm vi tái lập trên máy hiện tại (RTX 3050 Laptop 6 GB, Windows 11)

| Hạng mục | Khả thi? |
|---|---|
| Đánh giá checkpoint chính thức PIDNet-S/M/L & loss-ablation trên test (Bảng 2 hàng 10/15/20, Bảng 8 hàng 3, Bảng 4 hàng 5) | ✅ (chỉ suy luận) |
| Đánh giá SAM pseudo-label vs mask thủ công (Bảng 2 hàng `SAM*`) | ✅ |
| Train lại PIDNet-S, Protocol P, 50 epoch | ✅ với tích lũy gradient; ~2.9 s GPU + ~1 s data mỗi batch 16 → **ước tính 16–17 giờ/lần chạy** |
| Train lại PIDNet-M/L | ⚠️ Về lý thuyết được (micro-batch nhỏ hơn) nhưng ~3–5× thời gian; **không thực hiện** ở Giai đoạn 1 |
| Teacher BoxSnake (Bảng 2 hàng 1–4, 6–9, 11–14, 16–19) | ❌ Không có code/checkpoint BoxSnake trong repo; ngoài phạm vi |
| FPS trên Jetson Orin NX + TensorRT (Bảng 3) | ❌ Không có phần cứng; sẽ báo FPS trên RTX 3050 (PyTorch) và ghi rõ không so sánh được |
| Bảng 6, 7 | ❌ Thiếu dữ liệu (xem §4) |

## 9. Không đủ thông tin để xác minh

1. Supplementary của WACV (Eq. 3–6, Bảng 9–17, Bảng 14 tốc độ, Bảng 16 kết quả Croatia đầy đủ) — chưa có bản truy cập được trong phiên này.
2. Checkpoint công bố được chọn theo val loss hay val mIoU; dùng lr nào; có ImageNet pretrain không; seed; số lần chạy.
3. **Mâu thuẫn số liệu SAM**: Scientific Data nói chất lượng SAM mask được đo "with the 40 manually annotated images" = mIoU 0.636 / Acc 0.958 / P 0.912 / R 0.693 / F1 0.754 — **trùng khít hàng `SAM*` Bảng 2 WACV, vốn tính trên 80 ảnh test (40 AIFM + 40 UAV)**. Sẽ kiểm chứng ở bước đánh giá test baseline (tính cả 80 ảnh và riêng 40 ảnh BFF).
4. Scientific Data nói ảnh Subset C ở "1080×1920"; thực tế 906 ảnh train + 210 ảnh val BFF là **2048×1080** (loader resize về 1080×1920 → méo tỉ lệ nhẹ). Ảnh AIFM 640×480 (4:3) cũng bị resize méo về 16:9.
5. 9 ảnh train BFF không có file bbox label và không có SAM mask; 25 ảnh train + 21 ảnh val có file bbox rỗng và không có SAM mask → loader coi là **mẫu âm (mask rỗng)**. 1 ảnh (`evoDJI_0001_frame23`) có file bbox rỗng nhưng lại có SAM mask khác rỗng. Không sửa dữ liệu; giữ nguyên hành vi loader.

## 10. Protocol khóa cho nghiên cứu chính

- **Protocol P** (WACV, AIFM + BFF Subset C, split của tác giả, 3252/701/80) được khóa làm protocol chính cho cả Giai đoạn 1 và Giai đoạn 2. Lý do: (a) là protocol duy nhất có số liệu công bố + checkpoint chính thức để kiểm định evaluator; (b) test có mask thủ công cho cả hai nguồn; (c) paper đã có breakdown UAV/AIFM (Bảng 4) cho phép báo cáo riêng 40 ảnh Boreal.
- **Protocol B** (Boreal Subset C only 1184/248/40): dữ liệu và manifest đã chuẩn bị, giữ nguyên danh sách chính thức; **không train** trong Giai đoạn 1. Kết quả "mô hình train trên P, đánh giá trên 40 ảnh test BFF" sẽ được gắn nhãn **P→BFF-test**, không gọi là Protocol B, và không đặt cùng bảng với bất kỳ kết quả train-on-B nào.
- Không thêm dataset ngoài, không tái sinh pseudo-label, không chia lại split.

## 11. Thí nghiệm baseline đăng ký trước

| ID | Mô tả | Trạng thái đăng ký |
|---|---|---|
| E0-official | Đánh giá 4 checkpoint chính thức + SAM* trên test (chỉ suy luận) | Chạy tại bước khóa test Giai đoạn 1 |
| **E0** | PIDNet-S, Protocol P, cấu hình §3 (code-faithful), seed 42 | **Chính** |
| E0-lr3 | Như E0 nhưng lr 1e-3 (theo văn bản paper) | Phụ, nếu đủ GPU |
| E0-s2 | Như E0, seed khác (ước lượng nhiễu) | Phụ, nếu đủ GPU |

Báo cáo test của E0 cho **cả hai** checkpoint (best val mIoU = chính; best val loss = phụ).

## 12. Quy trình khóa test

- Code train (`research_extension/repro/train_repro.py`) chỉ dựng dataset `train`/`valid`; có `assert_not_test()` chặn mọi split khác. Không có tùy chọn đánh giá test trong vòng train.
- Đánh giá test là entry point riêng (`research_extension/eval/final_test_eval.py`), chỉ chạy khi có file khóa hợp lệ.
- **Khóa test Giai đoạn 1** (`reports/STAGE1_TEST_LOCK.md`): trước khi chạy test cho E0-official/E0, đóng băng code + checkpoint + quy tắc báo cáo. Chỉ in **số liệu tổng hợp** (toàn bộ, BFF, AIFM); không in theo ảnh, không lưu/hiển thị ảnh/mask/dự đoán test.
- Kết quả test Giai đoạn 1 là thông tin công khai tương đương Bảng 2/4 của paper; **không được dùng để thiết kế/tuning BRRC**. Mọi lựa chọn Giai đoạn 2 chỉ dựa vào validation.
- **Khóa test cuối** (`reports/FINAL_EVALUATION_LOCK.md`) cho Giai đoạn 2.

---

## PHIÊN BẢN 2 — bổ sung ngày 2026-10-12 (trước mọi đánh giá test)

Bằng chứng mới (chỉ dùng validation và trọng số checkpoint; **không dùng test**):

1. **Validation, cùng một routine** (`research_extension/repro/val_eval_checkpoints.py`, `reports/val_diagnostic_official_vs_E0.json`): checkpoint chính thức PIDNet-S đạt val batch-mIoU **0.5397** (paper Bảng 8: 0.540), loss-ablation **0.5365** (paper: 0.535) → dữ liệu + routine validation của chúng tôi tái tạo đúng số val của paper. E0 (code-faithful) chỉ đạt **0.3764** (epoch 44).
2. **Pháp y lr** (`research_extension/probes/forensic_lr_exact.py`, `reports/forensic_lr_exact_match.json`): BN weight của đầu biên (không nhận gradient do l2 ≡ 0) chỉ bị co bởi weight decay AdamW. Mô phỏng float32 từng bước khớp **đúng bit** với: chính thức S = 7 548 bước = 37 epoch × 204, M = 50 epoch, L = 21 epoch, **chỉ với lr 1e-4** (không có nghiệm với lr 1e-3). Kiểm chứng trên E0 (đã biết): ep44 → 45 epoch ✓, ep48 → 49 epoch ✓. → Tác giả dùng **lr 1e-4** (đúng code; văn bản paper ghi 1e-3 là sai lệch), wd 1e-2, 204 bước/epoch, và **l2 ≡ 0 cũng có trong lần train của tác giả** (checkpoint loss-ablation có BN weight đầu biên = 1.0 đúng, tức tham số không hề được cập nhật).
3. **Pháp y khởi tạo** (`forensic_init.py`, `reports/forensic_init.json`; `convert_mmseg_pidnet_imagenet.py`): conv1 của checkpoint chính thức có tương quan không gian 0.45 / tương quan RGB 0.44 (E0 và Kaiming ≈ 0). Trọng số ImageNet PIDNet-S do OpenMMLab chuyển đổi từ repo PIDNet chính thức (`pidnet-s_imagenet1k_20230306-715e6273.pth`, SHA-256 khớp tiền tố trong tên) sau khi đổi tên key về tên gốc có cosine trung bình **0.964** với checkpoint chính thức trên 50 lớp conv (E0: 0.0006). → Tác giả **có khởi tạo ImageNet**; code phát hành tắt bước này.

Quyết định (đăng ký trước khi xem test):

| ID | Mô tả | Vai trò |
|---|---|---|
| E0 | như §11 (code-faithful, train từ đầu) | giữ nguyên; báo cáo test như đã đăng ký |
| **E0-IN** | như E0 + khởi tạo nhánh P/I (conv1, layer1–5, compression3/4, pag3/4, layer3_–5_) từ trọng số ImageNet PIDNet-S (OpenMMLab, đổi tên key); nhánh D, SPP, DFM, các head khởi tạo Kaiming như code | **baseline chính cho Giai đoạn 2** (khớp quy trình tác giả thực sự dùng theo bằng chứng pháp y) |
| E0-lr3 | lr 1e-3 | **hủy** — pháp y cho thấy tác giả dùng lr 1e-4 |

Quy tắc checkpoint của E0-IN: giống E0 (chính: best val batch-mIoU; phụ: best val loss). Seed 42. Mọi thứ khác giữ nguyên.

---

## Ghi chú 2026-10-17 — đã đọc supplementary WACV 2025

Nguồn: `paper/PIDNet_teacher_student/wacv2025_pesonen_supplemental.pdf` (CVF open access), text: `references/papers/wacv2025_pesonen_supplemental.txt`. Giải quyết mục §9.1.
- Eq. 3–6: Accuracy = (TP+TN)/(P+N), Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN) — khớp evaluator (code gốc tính theo từng ảnh rồi lấy trung bình).
- Bảng 14: FPS trên Jetson Orin NX, suy luận **fp32**, đầu vào/ra 1080×1920 trên GPU; PIDNet-S 25.88 FPS (TensorRT, MAXN). Không tái lập được (không có phần cứng).
- Bảng 16: chỉ số đầy đủ trên dữ liệu Croatia (SAM→PIDNet-S mIoU 0.410). Không tái lập được (thiếu dữ liệu).
- **Không có thêm thông tin cấu hình train** (lr, seed, phần cứng train) → các kết luận pháp y ở Phiên bản 2 giữ nguyên.

## Ràng buộc tài nguyên từ người dùng (2026-10-17)

- Mọi run mới: VRAM ≤ 5 GB (`--max-vram-gb 5`), CPU tải nhẹ (worker 1 thread, `--cpu-threads 2`, ưu tiên BelowNormal). Ngoại lệ được người dùng chấp thuận: 8 epoch cuối của E0-IN chạy ở cấu hình cũ (~5,2 GB) khi GPU rảnh.
- Phương pháp so sánh được chọn (tổng 5): PIDNet-S teacher–student (baseline), SegFormer-B2 (công thức MultiNatSmoke), DSA trên PIDNet-S (cài đặt lại từ paper), FoSp, Mask2Former — tất cả trên Protocol P, cùng split/nhãn/evaluator; điều chỉnh do phần cứng phải ghi vào bảng sai khác của từng phương pháp.

---

## PHIÊN BẢN 3 — 2026-10-17: đăng ký trước các phương pháp so sánh (trước khi train/chấm)

Người dùng giao toàn quyền xử lý phần còn lại, không phản hồi trong thời gian chạy. Tất cả chạy qua hàng đợi `configs/queue.json` (`tools/scripts/queue_runner.py`), VRAM ≤ 5 GB, ưu tiên CPU thấp.

| ID | Phương pháp | Config (nguồn) | Điều chỉnh do phần cứng / môi trường | Checkpoint (đăng ký trước) | Ngưỡng |
|---|---|---|---|---|---|
| E0-IN | PIDNet-S teacher–student | như v2 | micro-batch 4 (đã có) | best val batch-mIoU (chính), best val loss (phụ) | 0.5 |
| E0-IN-m3 | như E0-IN | như v2 | **micro-batch 3** (giới hạn 5 GB) — baseline cho mọi so sánh PIDNet dưới ràng buộc mới | như trên | 0.5 |
| SegFormer-B2 | MultiNatSmoke WACV 2026 (paper §4.2 + code @f3ab188) | MiT-b2 ImageNet (`nvidia/mit-b2`), 512×512, không augmentation, BCE, AdamW 1e-4/0.01, 20 epoch, batch 32 | batch 32 = micro 8 × 4 (LayerNorm → tương đương chính xác); paper vs code: code ghi 15 epoch & `mit-b5` → theo paper | best val IoU (`compute_metrics` của họ) trên **val P**; họ chọn trên test → không theo | 0.5 |
| DSA | ECCV 2024, **cài đặt lại** (không có code) trên PIDNet-S cấu hình E0-IN-m3 | K=5, λ=0.4, γ=1e-5, Sinkhorn 3 vòng (paper) ; ε=0.05, chuẩn hóa L2, embedding = đầu vào head cuối (giả định) | micro-batch 3 | như E0-IN | 0.5 |
| Mask2Former | MultiNatSmoke recipe | Swin **Tiny** ADE20K (`facebook/mask2former-swin-tiny-ade-semantic`; paper không nêu kích thước), 512, BCE trên logits gộp theo code (chỉ số lớp 1), AdamW 1e-4/0.01, 20 epoch, batch 32 (paper; code: 16) | batch 32 = micro 4 × 8 (LayerNorm/GroupNorm → tương đương) | best val IoU | 0.5 |
| FoSp | AAAI 2024, **code chính thức không sửa** (`tools/train.py`, mmseg 0.27 đi kèm, mmcv 1.6.2 bản thuần Python) | `configs/fosp_P.py` = config gốc (MiT-B3, 512, aug, AdamW 6e-5 poly, 40k cập nhật, batch 6, seed 2077 mặc định) | batch 6 = micro 2 × 3 (BN ở decode head thấy 2 ảnh); lịch theo vòng lặp ×3 giữ nguyên 40k cập nhật; workers 2; dữ liệu P dạng VOC (mask SAM 255→1, ảnh .jpeg → hardlink .jpg) | best val mF-beta (quy tắc trong `mmseg/apis/train.py` gốc) | **0.4** (config FoSp, chính) và 0.5 (phụ) |

Quy tắc chung:
- Train trên P train (nhãn SAM), chọn checkpoint trên P val (nhãn SAM), test 80 mask thủ công bằng cùng evaluator (GT = pipeline WFSeg gốc, IoU theo ảnh, ảnh GT rỗng → 1). Mô hình không phải PIDNet: dự đoán theo pipeline suy luận của chính phương pháp, xác suất được nội suy song tuyến về 1080×1920.
- Mỗi phương pháp **cấu hình cố định** ở trên được chấm test **một lần** ngay sau khi train, dưới khóa riêng. Kết quả này **không** được dùng để thiết kế/tuning BRRC. BRRC (Giai đoạn 2) chỉ chấm test ở `FINAL_EVALUATION_LOCK`.
- Crash: hàng đợi tự resume tối đa 3 lần; quá số lần → FAILED, ghi lỗi, chuyển việc kế tiếp. Không chạy lại một khóa test đã có kết quả.

### Đính chính v3 sau chạy thử (2026-10-18, trước khi train/chấm các phương pháp so sánh)

- **SegFormer-B2**: decode head có **1 lớp BatchNorm** (đếm trực tiếp: 53 LayerNorm + 1 BatchNorm2d) → chia micro-batch **không** tương đương chính xác như đã ghi; micro-batch **2** (micro 8 và 4 vượt 5 GB; micro 2 đỉnh 2.85 GiB) → BN ở head thấy 2 ảnh.
- **Mask2Former Swin-T**: không có BatchNorm (72 LayerNorm + 5 GroupNorm) → chia micro-batch tương đương chính xác; chọn micro-batch **1** để VRAM chắc chắn < 5 GB (micro 4 vượt; micro 2 đỉnh 4.44 GiB).
- **DSA trên PIDNet-S**: micro-batch 3, đỉnh 3.86 GiB; L_DSA được ghi vào log (~3e-3 ở đầu train; nhỏ so với loss gốc do dạng nghịch đảo của Eq. 8).
- **FoSp** — 3 xử lý môi trường, không sửa code FoSp/mmcv, không đổi phép tính:
  1. `yapf==0.32.0` (mmcv 1.6 gọi `FormatCode(verify=...)`, yapf mới đã bỏ tham số).
  2. `evaluation.rule='greater'` trong config: `train.py` gốc đăng ký `EvalHook(save_best='mf_beta')` nhưng EvalHook của mmseg chỉ suy ra quy tắc cho `mIoU/mAcc/aAcc` → code phát hành báo lỗi "Cannot infer the rule for key mf_beta". F-beta càng lớn càng tốt → giữ đúng ý định tác giả.
  3. Shim `_get_stream(torch.device)` cho mmcv < 1.7.1 với PyTorch 2.x (đúng bản vá chính thức của mmcv 1.7.1), và đánh dấu sẵn plugin CUDA của MAT là "đã khởi tạo, không có plugin" → dùng thẳng nhánh tham chiếu PyTorch thuần mà code gốc vẫn dùng khi không biên dịch được, tránh việc code gốc thử biên dịch lại ở **mỗi** lần gọi (làm GPU chỉ chạy ~4%, CPU bị chiếm).

### Điều chỉnh tốc độ (2026-10-21, trước khi Mask2Former bắt đầu train; chưa có kết quả nào của Mask2Former)

- Người dùng yêu cầu chạy nhanh hơn, tận dụng VRAM nhưng không OOM. **Mask2Former: micro-batch 1 → 2** (batch logic vẫn 32 = 2 × 16; đỉnh đo được 4.44 GiB < trần 5 GiB). Mô hình không có BatchNorm nên phép tính gradient tương đương chính xác; chỉ khác thứ tự tiêu thụ RNG (dropout/drop-path) → nhiễu cỡ seed.
- **Không đổi** learning rate (PIDNet/BRRC/DSA 1e-4 khôi phục từ checkpoint tác giả; SegFormer/Mask2Former 1e-4 theo MultiNatSmoke; FoSp 6e-5 theo config gốc): LR không làm epoch nhanh hơn, và đổi LR/số epoch giữa chừng phá tính so sánh với baseline E0IN_m3 và các lần BRRC đã chạy.
- **Không đổi** micro-batch PIDNet/BRRC (3) và FoSp (2 × 3): có BatchNorm → đổi micro-batch đổi thống kê BN, các lần chạy trong lưới ablation không còn cùng điều kiện với baseline.
- **Đo tốc độ (2026-10-22, `research_extension/repro/probe_speed.py`, `probe_val_speed.py`; tensor ngẫu nhiên + tập valid, không chạm test)**: bước train PIDNet-S = 3.16 s (cudnn.benchmark tắt) vs 3.18 s (bật) → không lợi, giữ nguyên; bước train trong log thật ≈ 3.19 s → train hoàn toàn bị giới hạn bởi GPU, loader theo kịp. Validation: forward 42 ms/ảnh, loader valid 113 ms/ảnh (1 worker) vs 57 ms/ảnh (2 worker) → **`--val-num-workers` 1 → 2** cho các job PIDNet chưa xong (E2a chạy lại từ đầu vì chưa có checkpoint, E2b, E3, 4 seed). Loader valid không xáo trộn, không augmentation → batch và metric validation giống hệt; chỉ đổi tốc độ. Đo thực tế trên E2a: epoch 645–669 s (trước đó ~800–850 s ở E1b), validation ~60 s thay vì ~175 s → ~9 h/lần chạy 50 epoch thay vì ~11.5 h. `config_sha256` trong run_meta có tính số worker nên sẽ khác các lần chạy trước — khác biệt này không ảnh hưởng phép tính.
- **FoSp resume (2026-10-24)**: `tools/train.py` gốc gán `cfg.auto_resume = args.auto_resume` (mặc định False) nên `auto_resume=True` trong config bị bỏ qua; lần chạy lại 21:59 bắt đầu từ iter 0 và bị dừng ở iter ~150 (chưa ghi checkpoint nào, log `20261024_215949.log` giữ lại). `run_fosp.py` nay truyền cờ chính thức `--auto-resume` → tiếp tục từ `latest.pth` (iter 36000). mmseg không lưu trạng thái RNG/thứ tự dữ liệu khi resume → phần sau resume khác thứ tự mẫu so với chạy liền một mạch (cỡ nhiễu seed). Mất iter 36000–~39000 do máy tắt lúc 11:31 trước checkpoint kế tiếp.
- **Không bật AMP/fp16**: nhanh hơn nhưng đổi số học so với baseline E0IN_m3 và E1a/E1b đã chạy → phá tính so sánh.
- **Không nâng** trần VRAM quá 5 GiB: GPU 6144 MiB, nhưng CUDA context + Windows desktop đã chiếm ~0.8 GB ngoài bộ cấp phát PyTorch (đo: tổng 5125 MiB khi PIDNet chạy) → trần cao hơn có nguy cơ tràn sang bộ nhớ chia sẻ (chậm hơn) hoặc OOM.
