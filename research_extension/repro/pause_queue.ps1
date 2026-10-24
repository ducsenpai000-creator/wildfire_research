# Pause the unattended experiment queue safely.
#   powershell -ExecutionPolicy Bypass -File pause_queue.ps1 [-Now]
# 1) creates reports\QUEUE_PAUSE so the queue runner neither retries nor starts new jobs;
# 2) waits until the running PIDNet/HF job has just written its per-epoch state checkpoint (unless -Now;
#    FoSp checkpoints every 1000 updates via its own runner, so -Now is used for it);
# 3) stops the training process tree. Resume later with resume_queue.ps1.
param([switch]$Now)
$root = 'D:\wildfire_research'
New-Item -ItemType File -Force "$root\reports\QUEUE_PAUSE" | Out-Null
$all = Get-CimInstance Win32_Process
$train = $all | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_repro|train_hf|run_fosp' }
if (-not $train) { "no training process running; queue will stop at the next job boundary"; exit 0 }
$cmd = ($train | Select-Object -First 1).CommandLine
$exp = ([regex]::Match($cmd, '--exp\s+(\S+)')).Groups[1].Value
$glob = "$root\experiments\$exp\checkpoints\*_state.pt"
if ($cmd -match 'run_fosp') {
    # FoSp (mmseg IterBasedRunner, auto_resume): checkpoint = work_dir\latest.pth, written every 3000 updates
    $exp = 'FoSp_P_seed2077'
    $glob = "$root\experiments\$exp\latest.pth"
}
if (-not $Now -and $exp) {
    $state = Get-ChildItem $glob -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($state -and ((Get-Date) - $state.LastWriteTime).TotalMinutes -lt 3) {
        # a checkpoint was written moments ago: stopping now loses < 3 min instead of waiting a whole epoch
        "checkpoint of $exp saved at $($state.LastWriteTime) (< 3 min ago); stopping now"
    } elseif ($state) {
        $t0 = $state.LastWriteTime
        "waiting for the next epoch checkpoint of $exp (last saved $t0) ..."
        while ((Get-Item $state.FullName).LastWriteTime -eq $t0) { Start-Sleep -Seconds 5 }
        Start-Sleep -Seconds 30
    } else {
        # job just started: wait for its first epoch checkpoint instead of discarding the epoch in progress
        "waiting for the first epoch checkpoint of $exp ..."
        while (-not (Get-ChildItem $glob -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 5 }
        Start-Sleep -Seconds 30
    }
}
$all = Get-CimInstance Win32_Process
$ids = @($all | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_repro|train_hf|run_fosp' } | ForEach-Object { $_.ProcessId })
$frontier = $ids
while ($frontier.Count) { $frontier = @($all | Where-Object { $frontier -contains $_.ParentProcessId } | ForEach-Object { $_.ProcessId }); $ids += $frontier }
foreach ($id in ($ids | Select-Object -Unique)) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
"[{0}] queue PAUSED by pause_queue.ps1 (experiment {1})" -f (Get-Date -Format o), $exp | Tee-Object -FilePath "$root\reports\queue_pause.log" -Append
