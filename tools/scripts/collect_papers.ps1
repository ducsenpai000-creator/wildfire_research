# Collect all project-related papers into D:\wildfire_research\paper\<method>\ and paper\related\.
# Existing PDFs in references\papers are moved (text extracts stay there); others are downloaded from
# official/open-access sources. Every file is checked for a PDF header; failures are reported, not hidden.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$root = 'D:\wildfire_research\paper'
$ref = 'D:\wildfire_research\references\papers'
$ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36'
$dirs = 'PIDNet_teacher_student', 'SegFormer', 'DSA', 'FoSp', 'Mask2Former', 'related'
foreach ($d in $dirs) { New-Item -ItemType Directory -Force (Join-Path $root $d) | Out-Null }

# existing local PDFs -> destination (move; multiple destinations = copy then move)
$local = [ordered]@{
    'wacv2025_pesonen.pdf'       = @('PIDNet_teacher_student')
    'segformer_neurips2021.pdf'  = @('SegFormer')
    'dsa_eccv2024.pdf'           = @('DSA')
    'fosp_aaai2024.pdf'          = @('FoSp')
    'mask2former_cvpr2022.pdf'   = @('Mask2Former')
    'multinatsmoke_wacv2026.pdf' = @('SegFormer', 'Mask2Former', 'related')
    'cosw_neurips2024.pdf'       = @('related')
    'transbvm_aaai2022.pdf'      = @('related')
    'fcrfnet_faia2026.pdf'       = @('related')
}
foreach ($f in $local.Keys) {
    $src = Join-Path $ref $f
    if (-not (Test-Path $src)) { "MISSING local $f"; continue }
    foreach ($d in $local[$f]) { Copy-Item $src (Join-Path $root "$d\$f") -Force }
    Remove-Item $src
    "moved  $f -> $($local[$f] -join ', ')"
}

# downloads: name, folder, url
$dl = @(
    @('pidnet_cvpr2023.pdf', 'PIDNet_teacher_student', 'https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.pdf'),
    @('wacv2025_pesonen_supplemental.pdf', 'PIDNet_teacher_student', 'https://openaccess.thecvf.com/content/WACV2025/supplemental/Pesonen_Detecting_Wildfires_on_WACV_2025_supplemental.pdf'),
    @('segment_anything_iccv2023.pdf', 'PIDNet_teacher_student', 'https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf'),
    @('boxsnake_iccv2023.pdf', 'PIDNet_teacher_student', 'https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_BoxSnake_Polygonal_Instance_Segmentation_with_Box_Supervision_ICCV_2023_paper.pdf'),
    @('gated_scnn_iccv2019.pdf', 'PIDNet_teacher_student', 'https://openaccess.thecvf.com/content_ICCV_2019/papers/Takikawa_Gated-SCNN_Gated_Shape_CNNs_for_Semantic_Segmentation_ICCV_2019_paper.pdf'),
    @('trivialaugment_iccv2021.pdf', 'PIDNet_teacher_student', 'https://openaccess.thecvf.com/content/ICCV2021/papers/Muller_TrivialAugment_Tuning-Free_Yet_State-of-the-Art_Data_Augmentation_ICCV_2021_paper.pdf'),
    @('swin_transformer_iccv2021.pdf', 'Mask2Former', 'https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf'),
    @('boreal_forest_fire_scidata2025.pdf', 'related', 'https://europepmc.org/articles/PMC12350734?pdf=render'),
    @('raita_hakola_yolov5_boreal_isprs2023.pdf', 'related', 'https://isprs-archives.copernicus.org/articles/XLVIII-1-W2-2023/1771/2023/isprs-archives-XLVIII-1-W2-2023-1771-2023.pdf'),
    @('figlib_smokeynet_2021.pdf', 'related', 'https://arxiv.org/pdf/2112.08598'),
    @('svls_ipmi2021.pdf', 'related', 'https://arxiv.org/pdf/2104.05788'),
    @('adele_cvpr2022.pdf', 'related', 'https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Adaptive_Early-Learning_Correction_for_Segmentation_From_Noisy_Annotations_CVPR_2022_paper.pdf'),
    @('spatial_correction_noisy_seg_2023.pdf', 'related', 'https://arxiv.org/pdf/2308.02498'),
    @('simple_does_it_cvpr2017.pdf', 'related', 'https://openaccess.thecvf.com/content_cvpr_2017/papers/Khoreva_Simple_Does_It_CVPR_2017_paper.pdf'),
    @('tversky_loss_2017.pdf', 'related', 'https://arxiv.org/pdf/1706.05721'),
    @('focal_tversky_loss_2019.pdf', 'related', 'https://arxiv.org/pdf/1810.07842'),
    @('boundary_loss_midl2019.pdf', 'related', 'https://arxiv.org/pdf/1812.07032')
)
foreach ($x in $dl) {
    $dst = Join-Path $root "$($x[1])\$($x[0])"
    if ((Test-Path $dst) -and ([Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($dst)[0..3]) -eq '%PDF')) { "exists $($x[0])"; continue }
    try {
        Invoke-WebRequest -UserAgent $ua $x[2] -OutFile $dst -TimeoutSec 180
        $h = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($dst)[0..3])
        if ($h -eq '%PDF') { "ok     {0} ({1:N1} MB) -> {2}" -f $x[0], ((Get-Item $dst).Length / 1MB), $x[1] }
        else { Remove-Item $dst; "FAIL   $($x[0]): not a PDF (got '$h') from $($x[2])" }
    } catch { if (Test-Path $dst) { Remove-Item $dst }; "FAIL   $($x[0]): $($_.Exception.Message)" }
}
