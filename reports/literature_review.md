# Khảo sát phương pháp (Literature review)

Ngày cập nhật: 2026-10-12. Bảng tóm tắt: `reports/method_comparison.csv`. Bản PDF/text đã đọc nằm trong `references/papers/`.

**Quy ước độ tin cậy**
- ✅ *đọc toàn văn*: số liệu chép từ bảng trong bản PDF/HTML chính thức.
- 🟡 *chỉ abstract/trang publisher*: số liệu lấy từ abstract hoặc trang publisher; chưa kiểm tra bảng.
- ❌ *không xác minh được*.

**Phân loại nhiệm vụ** (không trộn lẫn khi so sánh): *smoke segmentation* (mask khói theo pixel), *flame segmentation*, *flame+smoke segmentation* (đa lớp), *fire/smoke detection* (bounding box), *smoke classification* (ảnh/clip có khói hay không).

**Cảnh báo so sánh chung.** Trong toàn bộ tài liệu dưới đây không có hai paper nào dùng cùng dataset + split + nhãn test + cách gộp metric. "mIoU" có ít nhất 3 nghĩa: (a) trung bình IoU theo ảnh của lớp khói (WACV 2025, Eq. 1); (b) IoU lớp khói gộp pixel toàn tập; (c) trung bình IoU của 2 lớp (khói + nền) kiểu MMSegmentation. Các con số giữa các mục **không được đặt cạnh nhau như cùng một benchmark**.

---

## 1. PIDNet teacher–student (paper mục tiêu)

1. **Pesonen J., Hakala T., Karjalainen V., Koivumäki N., Markelin L., Raita-Hakola A.-M., Suomalainen J., Pölönen I., Honkavaara E.** *Detecting Wildfires on UAVs with Real-Time Segmentation Trained by Larger Teacher Models.* WACV 2025, pp. 5166–5176. Open access CVF: <https://openaccess.thecvf.com/content/WACV2025/html/Pesonen_Detecting_Wildfires_on_UAVs_with_Real-Time_Segmentation_Trained_by_Larger_WACV_2025_paper.html>. arXiv 2408.10843. DOI IEEE: không tra trong phiên này. ✅
2. **Dataset**: AI For Mankind (HPWREN) v1+v2 bản sắp xếp lại (2068/453/40) + UAV Boreal (= Boreal Forest Fire Subset C, 1184/248/40). Test phụ: FESB Croatia (41 ảnh), AIFM ảnh không khói (623).
3. **Nhãn**: train/val = **pseudo-label SAM ViT-H** sinh từ bbox thủ công (hoặc mask từ teacher BoxSnake); test = 80 mask **vẽ tay**.
4. **Kiến trúc**: PIDNet-S/M/L (student, 7.72M/28.8M/37.3M tham số); teacher = SAM (zero-shot, prompt bbox) hoặc Mask R-CNN-FPN (ResNet-50/101, Swin-B/L) huấn luyện bằng BoxSnake.
5. **Cải tiến chính**: chưng cất teacher→student chỉ cần bbox; chạy thời gian thực trên Jetson Orin NX (~25 fps PIDNet-S, TensorRT).
6. **Loss/huấn luyện**: 0.4·BCE(P) + 20·BCE(main) + 1·boundary (Gated-SCNN) + 1·BAS (t=0.8); AdamW lr 1e-3 (code: 1e-4), wd 1e-2, 50 epoch, batch 16, augmentation kiểu TrivialAugment. *Phát hiện của chúng tôi*: trong code phát hành boundary loss ≡ 0 (xem `PROTOCOL_LOCK.md` D-2).
7. **Số liệu gốc (test 80 ảnh, mIoU theo ảnh)**: SAM→PIDNet-S 0.594 (Acc 0.960, P 0.891, R 0.645, F1 0.707); SAM→PIDNet-M 0.606; SAM→PIDNet-L 0.594; pseudo-label SAM* 0.636 (P 0.912, R 0.693); tốt nhất: Swin-B-FPN→PIDNet-M 0.643, teacher Swin-L-FPN 0.651. Bảng 4 (PIDNet-S, SAM): UAV 0.689, AIFM 0.498. Bảng 8: biến thể bỏ l2 → test 0.633 (README: checkpoint "loss ablation" 0.633).
8. **Code/trọng số**: <https://gitlab.com/fgi_nls/public/wildfire-real-time-segmentation> (MIT License, © 2024 Julius Pesonen), 4 checkpoint PIDNet công bố (Google Drive). Chạy được trên Windows, Python 3.8.
9. **Ưu điểm**: thực sự real-time trên phần cứng UAV; chỉ cần bbox; có test thủ công; code + checkpoint công khai. **Hạn chế**: pseudo-label SAM bảo thủ (recall 0.693); test nhỏ (80 ảnh) nên dao động lớn (ước lượng ±0.04 từ Bảng 8); boundary loss vô hiệu trong code; checkpoint selection mâu thuẫn paper↔code; FP rate 0.14 trên ảnh không khói.
10. **So sánh trực tiếp với protocol khóa (P)**: **Có** — đây chính là protocol P.

## 2. PIDNet (kiến trúc student)

1. **Xu J., Xiong Z., Bhattacharyya S. P.** *PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers.* CVPR 2023, pp. 19529–19539 (theo trích dẫn trong paper WACV). Code: <https://github.com/XuJiacong/PIDNet>. 🟡
2. **Dataset**: Cityscapes, CamVid (không phải khói).
3. **Nhãn**: pixel thủ công (19 lớp Cityscapes).
4. **Kiến trúc**: 3 nhánh P (chi tiết), I (ngữ cảnh), D (biên) + Bag/Light-Bag fusion.
5. **Cải tiến**: dùng nhánh D (biên) để điều phối hợp nhất P/I, giảm "overshoot" của mạng hai nhánh.
6. **Loss**: OHEM CE ×2 đầu ra, boundary weighted BCE, BAS-loss (t=0.8).
7. **Số liệu (README PIDNet, Cityscapes val/test mIoU, FPS)**: PIDNet-S 78.8 / 93.2 FPS; PIDNet-M 79.9 / 42.2; PIDNet-L 80.9 / 31.1. 🟡 (từ README, chưa đối chiếu bảng paper)
8. **Trọng số**: ImageNet & Cityscapes trên Google Drive; **link ImageNet hiện không truy cập được** (đã thử 2026-10-11).
9. **Ưu/nhược**: rất nhanh; nhưng cho khói (biên mờ) nhánh D/biên có thể có hại (WACV Bảng 8 gợi ý).
10. **So sánh trực tiếp**: là backbone của baseline; không có số trên dữ liệu khói.

## 3. FCRF-Net

1. **Nguyen H. N., Tran T. T. D., Thai C.-C., Le H. T., Nguyen K. L., Nguyen M. T., Tran H. M., Nguyen B. T.** *Flame Context Routing–Fusion Networks for UAV-Based Wildfire Segmentation.* Frontiers in Artificial Intelligence and Applications vol. 430 (SoMeT'26), pp. 151–160, 2026. DOI 10.3233/FAIA260734. CC BY-NC 4.0. ✅ (đọc toàn văn PDF)
2. **Dataset**: FLAME (chia 6:2:2 tự làm) và Boreal Forest Fire ("followed the official split").
3. **Nhãn**: không nói rõ. Với Boreal, **không nêu GT test là 40 mask thủ công hay SAM mask** → ❌ không xác minh được.
4. **Kiến trúc**: ResNet-50 (ImageNet) + ACRM (FiLM đa tỉ lệ, nhánh local/global, *uncertainty-guided routing*) + decoder GDFR (gated local/dilated, deep supervision). 36.4M tham số, 138.9 GFLOPs @512², 17.5 FPS (không nêu GPU).
5. **Cải tiến**: định tuyến đặc trưng local/global theo bản đồ "uncertainty" học được (không có giám sát riêng cho uncertainty).
6. **Loss/huấn luyện**: không nêu cụ thể loss chính; deep supervision phụ; AdamW lr 3e-4, wd 1e-3, cosine, 30 epoch, ảnh 512×512, chọn checkpoint theo val IoU.
7. **Số liệu gốc (Bảng 1)**: Boreal IoU 0.8607, P 0.9298, R 0.9206, F1 0.9251 (U-Net 0.8573, SegFormer-B2 0.8579, DeepLabV3+ 0.8537). FLAME IoU 0.8352. Cách gộp IoU (theo ảnh hay toàn tập): không nêu.
8. **Code/trọng số**: không công bố.
9. **Ưu**: có đánh giá trên Boreal. **Hạn chế**: thiếu chi tiết protocol (GT test, cách gộp IoU), không có code; tất cả baseline cách nhau ≤ 0.013 IoU trên Boreal nhưng không báo độ lệch giữa lần chạy; tiêu đề nói "flame" trong khi Boreal là khói.
10. **So sánh trực tiếp**: **Không**. Khác resolution (512 vs 1080×1920), dataset (Boreal-only vs P), không rõ GT test và cách gộp metric. Con số 0.86 **không** được so với 0.689 (WACV Bảng 4).

## 4. FSSNet (theo tên người dùng cung cấp)

1. **Liu Z., Sun Y., Jiang X., Duan P., Li M.** *A semantic segmentation model for early-stage fire detection from aerial remote sensing.* Engineering Applications of Artificial Intelligence, vol. 171, art. 114217, 2026. DOI 10.1016/j.engappai.2026.114217 (Crossref ✅). Toàn văn: ScienceDirect trả 403 → 🟡/❌.
2. **Dataset**: FLAME-SMOKE (tác giả tự xây; "first semantic segmentation dataset with flame and smoke labeled simultaneously per frame from a drone perspective" — theo abstract qua kết quả tìm kiếm).
3. **Nhãn**: pixel, 2 lớp flame + smoke (từ abstract); cách gán nhãn: ❌.
4. **Kiến trúc**: Compact ASPP; Bottom-up Detail-informed Feature Fusion Module (mục tiêu nhỏ); Foreground Emphasis Module (dùng đầu ra phân loại foreground để hướng phân đoạn). Backbone: ❌.
5. **Cải tiến**: phát hiện đồng thời lửa + khói ở giai đoạn sớm từ UAV.
6. **Loss/huấn luyện**: ❌.
7. **Số liệu**: ❌ không xác minh được. **Tên "FSSNet" không xuất hiện trong các nguồn truy cập được** — cần đối chiếu bản toàn văn.
8. **Code**: ❌.
9. **Ưu/nhược**: hướng đa lớp lửa+khói; không đánh giá được.
10. **So sánh trực tiếp**: **Không** (nhiệm vụ flame+smoke, dataset khác).

## 5. AusSmoke / MultiNatSmoke

1. **Li W., Zhao H., Zhu G., Ji G.-P., Wilson N., Yebra M., Barnes N.** *AusSmoke meets MultiNatSmoke: a fully-labelled diverse smoke segmentation dataset.* WACV 2026, pp. 7996–8006. DOI 10.1109/WACV61042.2026.00772. arXiv 2604.23542. Code: GitHub (abstract nói "available at Github"; repo `henryzhao0615/MultiNatSmoke` theo trang tổng hợp bên thứ ba — chưa mở kiểm tra). ✅
2. **Dataset**: AusSmoke 15 248 ảnh (Úc) + MultiNatSmoke 70 818 ảnh (Mỹ/FIgLib, **Boreal Forest Fire**, FireSpot, D-Fire, FESB-MLID, USTC, WSDataset); split riêng 59 735 train / 11 083 test.
4. **Nhãn**: giữ mask có sẵn (SmokeSeg, Smoke5K) sau kiểm tra; dataset chỉ có bbox → SAM (prompt bbox) → huấn luyện FoSp → pseudo-label → self-training; kiểm tra thủ công, ~7K ảnh gán lại bằng SAM tương tác.
4. **Kiến trúc**: benchmark U-Net, DeepLabV3+ (ResNet-50), SegFormer (MiT-B2), Mask2Former (Swin, ADE20K), OneFormer, Trans-BVM, FoSp.
5. **Cải tiến**: dữ liệu (quy mô, đa dạng địa lý).
6. **Huấn luyện**: 512×512, 20 epoch, AdamW lr 1e-4 wd 0.01, batch 32, BCE, early stopping.
7. **Số liệu gốc (Bảng 3, test MultiNatSmoke)**: SegFormer IoU 73.47 / F1 84.21; FoSp 72.20 / 83.32 (recall cao nhất 84.92); Mask2Former 70.92; Trans-BVM 67.35. **Bảng 6 (SegFormer test trên "Boreal")**: train SmokeSeg 72.80 IoU; train MultiNatSmoke-sub (không chứa Boreal) 78.78.
8. **Code/trọng số**: có repo (chưa kiểm tra trọng số).
9. **Ưu**: tập lớn, đa quốc gia, có Boreal. **Hạn chế**: nhãn Boreal là **nhãn mới của họ** (bán tự động), split mới, độ phân giải 512 → không cùng protocol với Subset C; nguy cơ **rò rỉ**: nếu dùng mô hình/dữ liệu MultiNatSmoke, ảnh test Boreal chính thức có thể nằm trong train của họ.
10. **So sánh trực tiếp**: **Không**. Không được dùng dữ liệu MultiNatSmoke trong protocol khóa (ngoài protocol).

## 6. FCSNet

1. **Wang K., Zeng Y., Zhang Z., Tan Y., Wen G.** *FCSNet: A Frequency-Domain Aware Cross-Feature Fusion Network for Smoke Segmentation.* Journal of Electronics & Information Technology (电子与信息学报) 47(7):2320–2333, 2025. DOI 10.11999/JEIT241021 (DOI do JEIT cấp; **không có trên Crossref** khi tra 2026-10-11). Trang chính thức: <https://jeit.ac.cn/en/article/doi/10.11999/JEIT241021>. 🟡 (abstract trang chính thức; PDF chưa tải được)
2. **Dataset**: RealSmoke-1, RealSmoke-2 (tác giả xây), SMOKE5K.
3. **Nhãn**: pixel (chi tiết ❌).
4. **Kiến trúc**: lai Transformer–CNN: Frequency Transformer (Fourier, lấy thành phần tần số thấp), Domain Interaction Module, Multi-level High-Frequency Perception Module (biên ở độ phân giải đầy đủ), Multi-Head Cross Attention ở decoder.
5. **Cải tiến**: kết hợp miền tần số + không gian, tăng cường biên.
6. **Loss/huấn luyện**: ❌.
7. **Số liệu (abstract chính thức)**: mIoU 58.59% (RealSmoke-1), 63.92% (RealSmoke-2), 78.94% (SMOKE5K). *Ghi chú*: một bản tóm tắt bên thứ ba ghi 58.95% cho RealSmoke-1 — mâu thuẫn chưa giải quyết; dùng số của trang chính thức.
8. **Code**: không thấy.
9. **Ưu/nhược**: chú trọng biên khói; FLOPs cao hơn TransFuse; không có code.
10. **So sánh trực tiếp**: **Không**.

## 7. CoSW

1. **Yao L., Zhao H., Wang Z., Zhao K., Peng J.** *CoSW: Conditional Sample Weighting for Smoke Segmentation with Label Noise.* NeurIPS 37 (2024), pp. 106743–106767. DOI 10.52202/079017-3389. <https://proceedings.neurips.cc/paper_files/paper/2024/hash/c1366167e310460f2386c36152a897d6-Abstract.html>. ✅
2. **Dataset**: SmokeSeg (val **gán lại sạch** bởi tác giả), SMOKE5K (val gán lại), NS-1K (1 000 ảnh SmokeSeg gán lại sạch; 700/300; nhiễu tổng hợp *erode/dilate/edge-distort* 20–80%, 2 mức).
3. **Nhãn**: thủ công có nhiễu thực + nhiễu tổng hợp ở biên/hình thái.
4. **Kiến trúc**: SegFormer MiT-B0/B3, ResNet-50 (MMSegmentation); thay decoder bằng đa prototype (K prototype/lớp, gán bằng Sinkhorn).
5. **Cải tiến**: trọng số mẫu theo pixel **có điều kiện theo cụm prototype** (regularized within-prototype entropy, RWE); cập nhật prototype có trọng số (momentum); regularized scatter metric learning.
6. **Loss**: CE có trọng số v_n + α·scatter loss; AdamW 6e-5, poly; crop 512 (SmokeSeg) / 480 (SMOKE5K); backbone ImageNet-1K.
7. **Số liệu gốc**: SmokeSeg MiT-B3: F1 72.32 / mIoU 59.25 (SegFormer 67.70/53.37); ResNet-50 68.49/54.09; MiT-B0 62.98/48.62. SMOKE5K: Fβ 81.71 / mIoU 71.24. NS-1K 80% nhiễu cao: F1 42.37 / mIoU 29.89 (Trans-BVM 37.38/25.58).
8. **Code**: không thấy link code trong paper/trang proceedings.
9. **Ưu**: nghiên cứu đầu tiên chuyên về nhiễu nhãn khói; xử lý cả nhiễu ở biên (NS-1K có edge-distorted). **Hạn chế** (tự nêu): giả định đa số pixel sạch; cần tập val sạch (không có trong Boreal/P); chi phí prototype + Sinkhorn; chưa thử với pseudo-label SAM hay mô hình real-time.
10. **So sánh trực tiếp**: **Không** (dataset/metric khác; không có code để chạy lại công bằng trên P).

## 8. DSA

1. **Yao L., Zhao H., Peng J., Wang Z., Zhao K.** *DSA: Discriminative Scatter Analysis for Early Smoke Segmentation.* ECCV 2024, LNCS, pp. 467–484. DOI 10.1007/978-3-031-72784-9_26. PDF ECVA: <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06090.pdf>. ✅
2. **Dataset**: SmokeSeg (500 ảnh test, chia small/medium/large), SMOKE5K (400 test), SYN70K (DS01–03).
3. **Nhãn**: thủ công (không xử lý nhiễu nhãn).
4. **Kiến trúc**: plug-in loss cho nhiều mô hình (AFFormer, SegFormer B0/B3, FoSp, DeepLabV3+, SegNeXt, Mask2Former Swin-B, Trans-BVM, DMLNet).
5. **Cải tiến**: đánh giá độ chặt/độ tách của không gian embedding qua **ma trận scatter** pixel–proxy (IOS) và proxy–proxy (OOS) thay vì khoảng cách Euclid; chỉ dùng lúc train.
6. **Loss**: loss gốc + λ·DSA; AdamW 6e-5 poly; MMSegmentation; ImageNet-1K.
7. **Số liệu gốc (SmokeSeg Total F1/mIoU)**: FoSp MiT-B3 72.05/59.03 → +DSA 73.40/59.91; SegFormer MiT-B3 67.99/54.98 → 70.15/56.54; Trans-BVM 69.12/55.57 → 70.94/56.60. SMOKE5K F1: FoSp+DSA 82.0.
8. **Code**: không thấy trên trang ECCV.
9. **Ưu**: không tốn chi phí suy luận; cải thiện ổn định 0.5–2 điểm. **Hạn chế**: không nhằm vào nhiễu nhãn/biên không chắc chắn; cải thiện nhỏ, không báo độ lệch giữa lần chạy.
10. **So sánh trực tiếp**: **Không**.

## 9. FoSp

1. **Yao L., Zhao H., Peng J., Wang Z., Zhao K.** *FoSp: Focus and Separation Network for Early Smoke Segmentation.* AAAI 2024, 38(7):6621–6629. DOI 10.1609/aaai.v38i7.28484. arXiv 2306.04474. Code: <https://github.com/LujianYao/FoSp> (theo MultiNatSmoke). ✅
2. **Dataset**: SmokeSeg (6 144 ảnh thật, nguồn FIgLib; tác giả công bố), SMOKE5K, SYN70K.
3. **Nhãn**: pixel thủ công.
4. **Kiến trúc**: MiT-B3 (hoặc B0 cho bản nhẹ) + Focus module (cascade hai chiều đa độ phân giải), Separation module (tách tiền cảnh khói/nền), Domain Fusion.
5. **Cải tiến**: tăng recall cho khói nhỏ/trong suốt và tách nền để tăng precision.
6. **Loss**: BCE đa mức: L = 0.1·BCE(FM) + Σ0.1·BCE(G_i) + 0.5·BCE(P); AdamW 6e-5, wd 0.01, 40k iter, batch 6, 512×512; thêm 4k ảnh tổng hợp khi train SmokeSeg.
7. **Số liệu gốc**: SmokeSeg Total Fβ 72.05 / mIoU 59.03 (small Fβ 66.03); SMOKE5K Fβ 81.6; SYN70K DS01 mIoU 83.00.
8. **Code/trọng số**: repo GitHub (chưa kiểm tra trọng số).
9. **Ưu**: tối ưu recall khói sớm. **Hạn chế**: mô hình trung bình (47.5M tham số), không real-time trên UAV; không xét nhiễu nhãn.
10. **So sánh trực tiếp**: **Không**.

## 10. Trans-BVM

1. **Yan S., Zhang J., Barnes N.** *Transmission-Guided Bayesian Generative Model for Smoke Segmentation.* AAAI 2022, 36(3):3009–3017. DOI 10.1609/aaai.v36i3.20207. Code + SMOKE5K: <https://github.com/redlessme/Transmission-BVM>. ✅
2. **Dataset**: SYN70K (DS01–03), SMOKE5K (1 360 thật + 4 000 tổng hợp; 400 test thật, 100 "difficult").
3. **Nhãn**: pixel.
4. **Kiến trúc**: Bayesian VAE sinh dự đoán ngẫu nhiên, encoder ResNet-50; mạng ước lượng uncertainty (aleatoric + epistemic) không cần lấy mẫu lúc test.
5. **Cải tiến**: mô hình hóa **uncertainty mức nhãn (aleatoric) và mức tri thức (epistemic)**; transmission-guided local coherence loss; uncertainty-calibrated entropy loss.
6. **Loss**: CE + IoU loss + KL + transmission loss + uncertainty consistency + entropy hiệu chỉnh; Adam 2.5e-5/1.5e-5, 50 epoch, 480×480, batch 6.
7. **Số liệu gốc**: SMOKE5K Total Fβ 0.791 / mMSE 0.002; Difficult Fβ 0.741. SYN70K mMSE DS01 0.1210.
8. **Code**: có.
9. **Ưu**: có xét uncertainty do khó gán nhãn. **Hạn chế**: không trọng số/loại bỏ nhãn nhiễu một cách tường minh; chậm; theo CoSW giảm mạnh khi nhiễu nhãn tăng (NS-1K).
10. **So sánh trực tiếp**: **Không**.

## 11. SegFormer

1. **Xie E., Wang W., Yu Z., Anandkumar A., Alvarez J. M., Luo P.** *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS 34 (2021), pp. 12077–12090. arXiv 2105.15203. Code: <https://github.com/NVlabs/SegFormer>. DOI: NeurIPS 2021 không cấp DOI (không xác minh được DOI). ✅
2. **Dataset**: ADE20K, Cityscapes, COCO-Stuff.
3. **Nhãn**: pixel thủ công.
4. **Kiến trúc**: encoder Mix Transformer (MiT-B0…B5) phân cấp, không positional encoding; decoder MLP nhẹ.
5. **Cải tiến**: hiệu quả + robust; decoder đơn giản.
6. **Loss**: CE; AdamW 6e-5, poly.
7. **Số liệu gốc**: ADE20K MiT-B0 37.4 mIoU (3.8M tham số); MiT-B5 51.0 (đơn tỉ lệ) / 51.8 (đa tỉ lệ); Cityscapes MiT-B5 84.0. Trên khói: là baseline mạnh nhất ở MultiNatSmoke (IoU 73.47) và baseline của CoSW/DSA/FoSp.
8. **Code/trọng số**: có (ImageNet, ADE20K, Cityscapes).
9. **Ưu**: baseline mạnh, ổn định. **Hạn chế**: MiT-B2+ nặng cho Jetson ở 1080p.
10. **So sánh trực tiếp**: chưa có kết quả trên protocol P (có thể chạy được trong tương lai — ngoài phạm vi thử nghiệm loss đầu tiên vì yêu cầu giữ nguyên backbone).

## 12. Mask2Former

1. **Cheng B., Misra I., Schwing A. G., Kirillov A., Girdhar R.** *Masked-attention Mask Transformer for Universal Image Segmentation.* CVPR 2022, pp. 1280–1289. DOI 10.1109/CVPR52688.2022.00135. arXiv 2112.01527. Code: <https://github.com/facebookresearch/Mask2Former>. ✅
2. **Dataset**: COCO (panoptic/instance), ADE20K, Cityscapes, Mapillary Vistas.
3. **Nhãn**: pixel thủ công.
4. **Kiến trúc**: backbone (R50…Swin-L) + pixel decoder đa tỉ lệ + transformer decoder với *masked attention*.
5. **Cải tiến**: một kiến trúc cho mọi tác vụ phân đoạn; masked attention hội tụ nhanh.
6. **Loss**: mask classification (CE lớp + BCE/Dice mask, point sampling), Hungarian matching.
7. **Số liệu gốc**: COCO panoptic 57.8 PQ; COCO instance 50.1 AP; ADE20K 57.7 mIoU (Swin-L); Cityscapes Swin-L 84.3 mIoU (đa tỉ lệ). Trên khói: MultiNatSmoke IoU 70.92; SmokeSeg (theo DSA) Swin-B F1 67.02 / mIoU 53.42.
8. **Code/trọng số**: có.
9. **Ưu**: mạnh, linh hoạt. **Hạn chế**: nặng, không real-time trên UAV; với khói nhị phân không vượt SegFormer trong các benchmark trên.
10. **So sánh trực tiếp**: **Không** (chưa chạy).

---

## 13. Công trình liên quan trực tiếp tới GAP/BRRC (nhiễu nhãn, biên, cân bằng FN/FP)

| Công trình | Nội dung liên quan | Độ tin cậy |
|---|---|---|
| Islam M., Glocker B. *Spatially Varying Label Smoothing: Capturing Uncertainty from Expert Annotations.* IPMI 2021, arXiv 2104.05788 | Làm mềm nhãn theo **lân cận không gian** (kernel Gauss) → giảm độ tin nhãn ở vùng biên; cải thiện dự đoán biên và calibration | 🟡 |
| Khoreva A. et al. *Simple Does It: Weakly Supervised Instance and Semantic Segmentation.* CVPR 2017 (được WACV 2025 trích dẫn [26]) | Từ bbox tạo nhãn với **vùng "ignore"** cho pixel không chắc chắn (giữa lõi và biên) | 🟡 (kiến thức chung, chưa mở lại paper) |
| PASCAL VOC (quy ước nhãn) | Dải biên 5 px được gán `void/255` (bỏ qua khi train/đánh giá) | 🟡 |
| Liu S. et al. *Adaptive Early-Learning Correction for Segmentation from Noisy Annotations (ADELE).* CVPR 2022, arXiv 2110.03740, code `Kangningthu/ADELE` | Mạng học pixel sạch trước rồi mới "nhớ" nhãn sai; sửa nhãn thích nghi theo lớp | 🟡 |
| Yao J. et al. *Learning to Segment from Noisy Annotations: A Spatial Correction Approach (SC).* arXiv 2308.02498 (venue ICLR 2023: chưa xác minh) | Mô hình hóa nhiễu nhãn phân đoạn **có tương quan không gian** (lệch biên) | 🟡 |
| Salehi S. S. M., Erdogmus D., Gholipour A. *Tversky loss function for image segmentation using 3D FCNs.* arXiv 1706.05721 (MLMI 2017) | Loss Tversky (α/β) cân bằng FP/FN; β=0.7 ưu tiên recall | 🟡 |
| *False Alarm Rectification for Early Smoke Segmentation.* WACV 2026 (poster 715; tác giả: chưa xác minh) | Dùng độ tin cậy bộ phân loại ảnh có/không khói làm prior để giảm FP; SmokeSeg IoU 61.83%, FPR 0.28% | 🟡 (trang poster) |

Nhận xét: (i) "giảm trọng số/bỏ qua vùng biên không chắc chắn" đã có tiền lệ rõ (SVLS, ignore-band VOC, Simple Does It); (ii) "loss cân bằng recall" có tiền lệ rõ (Tversky, Focal-Tversky); (iii) CoSW đã nghiên cứu nhiễu nhãn khói kể cả nhiễu ở biên, nhưng bằng trọng số theo đặc trưng (prototype), không dùng hình học mask, cần backbone/decoder prototype, chưa có code; (iv) **chưa thấy** công trình nào đánh giá các kỹ thuật này trên pseudo-label SAM của Boreal/WACV với mô hình real-time và test thủ công. Phân tích chi tiết ở `reports/research_gap.md`.
