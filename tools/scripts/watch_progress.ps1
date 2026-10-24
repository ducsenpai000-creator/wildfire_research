# Live, read-only view of the experiment queue: follows the log of whichever job is running and prints a
# summary line (epochs done, ETA, GPU) every minute. Closing this window does NOT affect training.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\scripts\watch_progress.ps1
$root = 'D:\wildfire_research'
$Host.UI.RawUI.WindowTitle = 'Wildfire training progress (close = stop watching only)'
$registry = Get-Content "$root\configs\experiments_registry.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$exp = ''; $log = ''; $pos = 0L; $lastSummary = [datetime]::MinValue

function Read-New([string]$path, [long]$from) {
    $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
    try {
        if ($fs.Length -lt $from) { $from = 0 }
        $fs.Seek($from, 'Begin') | Out-Null
        $text = (New-Object System.IO.StreamReader($fs)).ReadToEnd()
        return @($text, $fs.Length)
    } finally { $fs.Close() }
}

function Write-Summary([string]$e) {
    $gpu = (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null) -join ''
    $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                 Where-Object { $_.CommandLine -match 'queue_runner' }).Count -gt 0
    $state = if ($running) { 'RUNNING' } elseif (Test-Path "$root\reports\QUEUE_PAUSE") { 'PAUSED' } else { 'STOPPED' }
    $line = "[{0:HH:mm}] queue {1} | GPU {2}" -f (Get-Date), $state, $gpu
    $csv = "$root\experiments\$e\epochs.csv"
    if ($e -and (Test-Path $csv)) {
        $rows = @(Import-Csv $csv)
        $total = if ($registry.$e) { [int]$registry.$e.epochs } else { 0 }
        $line += " | $e epoch $($rows.Count)/$total"
        if ($rows.Count -gt 0 -and $total -gt 0) {
            $avg = ($rows | Select-Object -Last 5 | Measure-Object epoch_sec -Average).Average
            $best = ($rows | Measure-Object val_iou -Maximum).Maximum
            $line += " | best val mIoU {0:N4} | ~{1:N0} min/epoch | job ETA {2:HH:mm dd/MM}" -f $best, ($avg / 60),
                     (Get-Date).AddSeconds($avg * ($total - $rows.Count))
        }
    }
    Write-Host $line -ForegroundColor Cyan
}

while ($true) {
    try {
        $q = Get-Content "$root\reports\queue_status.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        $cur = [string]$q.current_exp
    } catch { $cur = $exp }
    if ($cur -ne $exp) {
        $exp = $cur; $log = ''
        Write-Host ("`n===== job: {0} =====" -f $(if ($exp) { $exp } else { '(none)' })) -ForegroundColor Yellow
        $lastSummary = [datetime]::MinValue
    }
    if ($exp) {
        $newest = Get-ChildItem "$root\experiments\$exp\*.log" -ErrorAction SilentlyContinue |
                  Sort-Object LastWriteTime | Select-Object -Last 1
        if ($newest -and $newest.FullName -ne $log) {
            $log = $newest.FullName
            $pos = [math]::Max(0L, $newest.Length - 1500)   # show the last few lines first
        }
        if ($log) {
            $r = Read-New $log $pos
            if ($r[0]) { Write-Host -NoNewline $r[0] }
            $pos = $r[1]
        }
    }
    if (((Get-Date) - $lastSummary).TotalSeconds -ge 60) { Write-Summary $exp; $lastSummary = Get-Date }
    Start-Sleep -Seconds 5
}
