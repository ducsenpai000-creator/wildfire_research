# Môi trường thực nghiệm

Ngày ghi: 2026-10-11.

## Phần cứng / hệ điều hành

| Thành phần | Giá trị |
|---|---|
| OS | Windows 11 Home Single Language 10.0.26200 |
| CPU | AMD Ryzen 7 7840HS (8 nhân / 16 luồng) |
| RAM | 15.3 GB |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU (compute capability 8.6), driver 617.42 (WDDM), ~0.8–1.0 GB VRAM bị ứng dụng nền chiếm |
| WSL2 | Không cài (không cần: repo chạy được trên Windows) |

Lưu ý WDDM: khi vượt VRAM, CUDA trên Windows có thể tràn sang RAM chung (sysmem fallback) thay vì báo OOM, làm chậm mạnh. Probe bs=8 fp32 báo "peak 9.5 GiB" là do cơ chế này → coi như không khả thi.

## Python / thư viện (môi trường `envs/wfseg_py38`)

README yêu cầu Python 3.8 và `requirements.txt` không ghim phiên bản. Chọn Python 3.8 + phiên bản cuối cùng hỗ trợ 3.8 của PyTorch (2.4.1) để gần với thời điểm code (08/2024) nhất mà vẫn có wheel CUDA cho Ampere.

| Gói | Phiên bản |
|---|---|
| Python | 3.8.20 (CPython do `uv` quản lý, cài trong `tools/uv_python/`) |
| torch | 2.4.1+cu118 (CUDA runtime 11.8, cuDNN 9.1.0) |
| torchvision | 0.19.1+cu118 |
| torchmetrics | 1.5.2 |
| torcheval | 0.0.7 |
| numpy | 1.24.4 |
| opencv-python | 4.10.0.84 (hạ từ 5.0.0.93 mà resolver chọn, vì nhãn biên dùng `cv2.Canny/dilate`; tránh thay đổi major version so với 2024) |
| scikit-image / pandas / matplotlib | 0.21.0 / 2.0.3 / 3.7.5 |
| einops / yacs / natsort / tqdm | 0.8.1 / 0.1.8 / 8.4.0 / 4.70.1 |

Danh sách đầy đủ: `configs/requirements_lock_py38.txt`.

Phiên bản thư viện mà tác giả dùng **không được công bố** → mọi sai khác số liệu do thư viện không thể loại trừ hoàn toàn.

## Công cụ (cô lập, không cài vào hệ thống)

| Công cụ | Vị trí | Lý do |
|---|---|---|
| MinGit 2.56.0.windows.2 (portable) | `tools/mingit/` | Máy không có git |
| uv 0.13.0 | `tools/uv/` | Tạo Python 3.8 + venv cô lập (máy không có conda) |
| `envs/docs` (Python 3.10 hệ thống + pypdf 5.1.0, gdown 5.2.0) | `envs/docs/` | Trích text PDF, tải Google Drive |

## Lệnh cài đặt có thể chạy lại (PowerShell)

```powershell
$env:UV_PYTHON_INSTALL_DIR='D:\wildfire_research\tools\uv_python'
$env:UV_CACHE_DIR='D:\wildfire_research\tools\uv_cache'
$uv='D:\wildfire_research\tools\uv\uv.exe'
& $uv python install 3.8
& $uv venv D:\wildfire_research\envs\wfseg_py38 --python 3.8 --python-preference only-managed
$py='D:\wildfire_research\envs\wfseg_py38\Scripts\python.exe'
& $uv pip install --python $py torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
& $uv pip install --python $py -r D:\wildfire_research\configs\requirements_lock_py38.txt
```

Kiểm tra: `python -c "import torch; print(torch.cuda.is_available())"` → `True` (đã xác nhận).

## Kiểm tra mô hình

- 4 checkpoint chính thức nạp vào `FullModel(PIDNet)` với 0 key thiếu / 0 key thừa.
- Số tham số: PIDNet-S 7 716 291; M 28 758 467; L 37 305 059 (paper: 7.72M / 28.8M / 37.3M).
- Forward GPU 1080×1920: 3 đầu ra `[B,1,1080,1920]` (P-head, main, boundary) sau upsample bilinear `align_corners=True`; dự đoán mask = `sigmoid(main) > 0.5`.
- VRAM train fp32 PIDNet-S: bs2 2.46 GiB, bs3 3.64, bs4 4.82, bs8 9.51 (tràn); AMP bs4 3.06 (không dùng AMP).
- Thời gian GPU thuần / batch logic 16 (4×4, fp32): 2.89 s (cuDNN benchmark tắt), 2.71 s (bật; không dùng).
