# Paper library

Tất cả paper liên quan tới dự án, tải từ nguồn chính thức / open access (2026-10-17). File PDF không đưa vào git (xem `.gitignore`); script tải lại: `tools/scripts/collect_papers.ps1`. Bản trích text dùng cho báo cáo nằm ở `references/papers/*.txt`.

## 5 phương pháp so sánh

### `PIDNet_teacher_student/` — baseline đang tái thực nghiệm
| File | Paper | Vai trò |
|---|---|---|
| `wacv2025_pesonen.pdf` | Pesonen et al., *Detecting Wildfires on UAVs with Real-Time Segmentation Trained by Larger Teacher Models*, WACV 2025, pp. 5166–5176 | **Paper mục tiêu tái thực nghiệm** |
| `wacv2025_pesonen_supplemental.pdf` | Supplementary của paper trên (CVF) | Eq. 3–6, Bảng 9–17 |
| `pidnet_cvpr2023.pdf` | Xu, Xiong, Bhattacharyya, *PIDNet*, CVPR 2023 | Kiến trúc student |
| `segment_anything_iccv2023.pdf` | Kirillov et al., *Segment Anything*, ICCV 2023 | Teacher sinh pseudo-label |
| `boxsnake_iccv2023.pdf` | Yang et al., *BoxSnake*, ICCV 2023 | Teacher thay thế (không tái thực nghiệm) |
| `gated_scnn_iccv2019.pdf` | Takikawa et al., *Gated-SCNN*, ICCV 2019 | Nguồn của boundary loss l2 |
| `trivialaugment_iccv2021.pdf` | Müller & Hutter, *TrivialAugment*, ICCV 2021 | Chiến lược augmentation |

### `SegFormer/`
| File | Paper | Vai trò |
|---|---|---|
| `segformer_neurips2021.pdf` | Xie et al., *SegFormer*, NeurIPS 2021 (arXiv 2105.15203) | Phương pháp |
| `multinatsmoke_wacv2026.pdf` | Li et al., *AusSmoke meets MultiNatSmoke*, WACV 2026, DOI 10.1109/WACV61042.2026.00772 | Công thức train SegFormer-B2 cho khói (benchmark) |

### `DSA/`
| File | Paper | Vai trò |
|---|---|---|
| `dsa_eccv2024.pdf` | Yao et al., *DSA: Discriminative Scatter Analysis for Early Smoke Segmentation*, ECCV 2024, DOI 10.1007/978-3-031-72784-9_26 | Phương pháp (không có code chính thức → cài đặt lại từ paper) |

### `FoSp/`
| File | Paper | Vai trò |
|---|---|---|
| `fosp_aaai2024.pdf` | Yao et al., *FoSp: Focus and Separation Network for Early Smoke Segmentation*, AAAI 2024, DOI 10.1609/aaai.v38i7.28484 | Phương pháp (code: github.com/LujianYao/FoSp) |

### `Mask2Former/`
| File | Paper | Vai trò |
|---|---|---|
| `mask2former_cvpr2022.pdf` | Cheng et al., *Masked-attention Mask Transformer*, CVPR 2022, DOI 10.1109/CVPR52688.2022.00135 | Phương pháp |
| `swin_transformer_iccv2021.pdf` | Liu et al., *Swin Transformer*, ICCV 2021 | Backbone |
| `multinatsmoke_wacv2026.pdf` | (bản sao) | Công thức train Mask2Former cho khói |

## `related/` — dataset, nhiễu nhãn, loss liên quan
| File | Paper | Liên quan |
|---|---|---|
| `boreal_forest_fire_scidata2025_fulltext_europepmc.xml/.txt` | Pesonen et al., *Boreal Forest Fire dataset*, Scientific Data 2025, DOI 10.1038/s41597-025-05634-0 | Dataset (PDF bị chặn 403 ở nature.com/PMC → lưu toàn văn từ Europe PMC) |
| `raita_hakola_yolov5_boreal_isprs2023.pdf` | Raita-Hakola et al., ISPRS Archives 2023 | Nguồn bbox UAV của dataset |
| `figlib_smokeynet_2021.pdf` | Dewangan et al., *FIgLib & SmokeyNet* (arXiv 2112.08598) | Nguồn ảnh HPWREN/AI For Mankind |
| `multinatsmoke_wacv2026.pdf` | (bản sao) | Dataset có ảnh Boreal gán nhãn lại |
| `cosw_neurips2024.pdf` | Yao et al., *CoSW*, NeurIPS 2024 | Nhiễu nhãn khói (không có code) |
| `transbvm_aaai2022.pdf` | Yan et al., *Trans-BVM*, AAAI 2022 | Uncertainty nhãn khói |
| `fcrfnet_faia2026.pdf` | Nguyen et al., *FCRF-Net*, FAIA 2026 (CC BY-NC 4.0) | Phân đoạn trên Boreal (không có code) |
| `false_alarm_rectification_smoke_wacv2026.pdf` | Zhao et al., *False Alarm Rectification for Early Smoke Segmentation*, WACV 2026 | Giảm FP cho khói |
| `svls_ipmi2021.pdf` | Islam & Glocker, *Spatially Varying Label Smoothing*, IPMI 2021 | Tiền lệ giảm tin cậy vùng biên |
| `simple_does_it_cvpr2017.pdf` | Khoreva et al., *Simple Does It*, CVPR 2017 | Vùng "ignore" từ bbox |
| `adele_cvpr2022.pdf` | Liu et al., *ADELE*, CVPR 2022 | Nhãn phân đoạn nhiễu |
| `spatial_correction_noisy_seg_2023.pdf` | Yao et al., *Learning to Segment from Noisy Annotations: A Spatial Correction Approach* (arXiv 2308.02498) | Nhiễu nhãn có tương quan không gian |
| `tversky_loss_2017.pdf` | Salehi et al., *Tversky loss* (arXiv 1706.05721) | Loss cân bằng FP/FN |
| `focal_tversky_loss_2019.pdf` | Abraham & Khan, *Focal Tversky loss* (arXiv 1810.07842) | Loss cân bằng FP/FN |
| `boundary_loss_midl2019.pdf` | Kervadec et al., *Boundary loss* (arXiv 1812.07032) | Loss dựa trên khoảng cách biên |

## Không tải được
| Paper | Lý do |
|---|---|
| FCSNet, J. Electronics & Information Technology 47(7):2320–2333, 2025, DOI 10.11999/JEIT241021 | Nút tải PDF của JEIT chạy bằng JavaScript; link trực tiếp trả 400 |
| Liu et al., *A semantic segmentation model for early-stage fire detection from aerial remote sensing* (FSSNet), EAAI vol. 171, 2026, DOI 10.1016/j.engappai.2026.114217 | ScienceDirect trả 403 (không open access) |
