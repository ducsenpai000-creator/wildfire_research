# Research gap

Cập nhật: 2026-10-12. Protocol khóa: **P** (WACV 2025; AIFM + Boreal Subset C; test 80 mask thủ công).

**Quy tắc bằng chứng.** Tài liệu này chỉ dùng: (a) số liệu **đã công bố** trong các paper, (b) dữ liệu **train/validation**, (c) phân tích trọng số checkpoint. **Không** dùng bất kỳ số test nào do chúng tôi tính (S1a/S1b) để lập luận hay thiết kế — kể cả các số test đã tái lập trùng với paper được dẫn theo **paper**.

## 1. Khoảng trống ứng viên ban đầu và kiểm chứng

> *Mask train/val của Boreal (và AIFM trong protocol P) là pseudo-label SAM, test là mask vẽ tay; pseudo-label bỏ sót khói đáng kể.*

| Câu hỏi kiểm chứng | Trả lời | Bằng chứng |
|---|---|---|
| Nhãn train/val là SAM, test là thủ công? | **Đúng** | Sci. Data (Subset C); WACV §3.1–3.2; repo (`sam_masks/` vs `manual_masks/`) |
| SAM bỏ sót khói (FN) nhiều hơn gán thừa (FP)? | **Đúng (mức tổng thể)** | WACV Bảng 2 `SAM*`: P 0.912, R 0.693 trên 80 ảnh test; Sci. Data: "the low recall shows that the SAM-masks were more conservative" |
| Student kế thừa thiên lệch này? | **Đúng** | WACV Bảng 2: SAM→PIDNet-S P 0.891, R 0.645 (recall là điểm yếu nhất) |
| Thiên lệch bảo thủ cũng thấy trên train? | **Phù hợp** | Train: mask SAM phủ trung vị 47–48% diện tích bbox người gán và chỉ 1–5% nằm ngoài bbox (`gap_evidence_train_sam_vs_bbox.json`). *Hạn chế*: bbox rộng hơn khói thật nên tỉ lệ phủ không đo trực tiếp recall |
| Bài toán đã được giải quyết trên chính protocol này? | **Chưa thấy** | Không có công trình nào trong khảo sát vừa (i) dùng protocol P/Subset C với test thủ công, (ii) xử lý tường minh nhiễu pseudo-label SAM, (iii) giữ mô hình real-time |
| Supervision dựa trên biên đã được thử đúng cách trên protocol này? | **Không** | Phát hiện D-2/G3: boundary loss của WACV ≡ 0 cả trong code lẫn checkpoint tác giả → kết luận Bảng 8 về loss biên không có hiệu lực |

## 2. Các công trình đã xử lý khía cạnh liên quan (không được tuyên bố "chưa ai làm")

| Công trình | Đã xử lý gì | Chưa xử lý gì so với protocol P |
|---|---|---|
| **CoSW** (NeurIPS 2024) | Nhiễu nhãn khói, kể cả nhiễu ở biên (NS-1K: erode/dilate/edge-distort); trọng số pixel theo prototype | Không thử với pseudo-label SAM, không trên Boreal/AIFM-P, không real-time; tự nêu cần val sạch; **không có code** → không tái thực nghiệm công bằng được |
| **DSA** (ECCV 2024) | Loss phân biệt embedding cho khói sớm | Không nhắm nhiễu nhãn hay biên không chắc chắn |
| **FCRF-Net** (FAIA 2026) | Kiến trúc có "uncertainty-guided routing"; đánh giá trên Boreal | Uncertainty không giám sát, không liên quan độ tin cậy nhãn; không rõ GT test thủ công hay SAM; không có code |
| **Trans-BVM** (AAAI 2022) | Uncertainty aleatoric (khó gán nhãn) | Không trọng số hóa/khử nhiễu nhãn tường minh; giảm mạnh khi nhiễu tăng (theo CoSW) |
| **FoSp** (AAAI 2024) | Kiến trúc hướng recall cho khói nhỏ | Không xét nhiễu nhãn |
| **SVLS** (IPMI 2021) | Làm mềm nhãn theo lân cận không gian → giảm độ tin ở biên | Đối xứng; y tế; không xét thiên lệch FN một phía |
| Vùng "ignore" quanh biên (VOC; Khoreva et al. 2017) | Bỏ qua pixel biên không chắc chắn | Đối xứng, bán kính cố định |
| **Tversky loss** (Salehi 2017) | Cân bằng FP/FN | Tổng quát, không gắn với cấu trúc nhiễu |
| **ADELE** (CVPR 2022) | Sửa nhãn nhiễu thích nghi | Sửa/thay nhãn — không phù hợp ràng buộc "không sửa pixel nhãn gốc" của dự án |

**Kết luận về novelty.** Hai thành phần của BRRC (giảm trọng số vùng biên; loss cân bằng recall) **đều đã có tiền lệ**. Không có cơ sở để gọi BRRC là thuật toán mới. Đóng góp trung thực có thể là: *một nghiên cứu thực nghiệm có kiểm soát* về việc các kỹ thuật này có giúp student real-time học từ pseudo-label SAM bảo thủ hay không, trên protocol công khai có test thủ công, với ước lượng nhiễu giữa các lần chạy.

## 3. Khoảng trống hẹp được đề xuất

**G-narrow:** Pseudo-label SAM trong protocol P có nhiễu **không đối xứng** (bảo thủ: ít FP, nhiều FN). Các kỹ thuật làm mềm/bỏ qua biên hiện có (SVLS, ignore-band) xử lý **đối xứng** và với bán kính cố định. Chưa có đánh giá có kiểm soát của:
1. giảm trọng số **một phía** — chỉ pixel *nền* ngay ngoài biên mask SAM (nơi khói bị bỏ sót có khả năng nằm), giữ nguyên giám sát cho pixel khói;
2. bán kính dải biên **theo tỉ lệ kích thước đối tượng** thay vì cố định (dữ liệu train cho thấy dải 8 px cố định chiếm 65% pixel khói AIFM, 10% với BFF);
3. kết hợp với số hạng recall (Tversky) — trên student real-time và đánh giá bằng mask thủ công.

Giả thuyết kiểm định: *so với E0-IN cùng ngân sách, biến thể có (1)+(3) tăng recall trên test thủ công mà không giảm mIoU quá mức nhiễu giữa lần chạy.*

## 4. Giới hạn của kết luận (phải ghi trong mọi báo cáo kết quả)

1. **Validation không sạch.** Val là pseudo-label SAM. Một phương pháp dự đoán *nhiều khói hơn SAM* (đúng mục tiêu) sẽ bị val phạt như FP. Chọn siêu tham số/checkpoint trên val vì vậy **thiên vị chống lại** hiệu ứng cần tìm; không có cách khắc phục trong protocol khóa (không được thêm nhãn).
2. **Nhiễu giữa lần chạy lớn.** Hai lần chạy cùng phép tính của tác giả cho test 0.594 và 0.633 (Bảng 8) → khác biệt < ~0.04 mIoU từ một seed không có ý nghĩa. Cần nhiều seed cho cả baseline và phương pháp.
3. **Test nhỏ** (80 ảnh; 40 mỗi nguồn) và chỉ dùng **một lần** sau khóa cuối.
4. **Không so trực tiếp** được với CoSW/FCRF-Net/MultiNatSmoke (dataset, nhãn, metric khác; không có code).
5. Bằng chứng "SAM bảo thủ" ở mức tổng hợp; phân bố không gian của FN (gần biên hay ở vùng khói mỏng tách rời) **chưa được xác minh** — đây là một giả định của thiết kế.
