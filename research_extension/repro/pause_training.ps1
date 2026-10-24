# Safely pause a running experiment: wait until the per-epoch full-state checkpoint (<exp>_state.pt)
# has just been rewritten, then stop the auto-resume wrapper FIRST (so it does not restart training)
# and then the training process tree. Nothing of the finished epochs is lost.
#   powershell -ExecutionPolicy Bypass -File pause_training.ps1 -Exp E0IN_P_pidnet_s_seed42 [-Now] [-DryRun]
param(
    [Parameter(Mandatory = $true)][string]$Exp,
    [switch]$Now,     # do not wait for the epoch boundary (loses the current partial epoch)
    [switch]$DryRun   # only show what would be stopped
)
$outDir = "D:\wildfire_research\experiments\$Exp"
$state = "$outDir\checkpoints\${Exp}_state.pt"
if (-not (Test-Path $state)) { throw "no state checkpoint yet: $state" }

function Get-RunProcs {
    $all = Get-CimInstance Win32_Process
    $wrappers = $all | Where-Object { $_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'run_with_resume' -and $_.CommandLine -match [regex]::Escape($Exp) }
    $trainers = $all | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_repro' -and $_.CommandLine -match [regex]::Escape($Exp) }
    $tree = @()
    $frontier = @($trainers | ForEach-Object { $_.ProcessId })
    while ($frontier.Count) {
        $tree += $frontier
        $frontier = @($all | Where-Object { $frontier -contains $_.ParentProcessId } | ForEach-Object { $_.ProcessId })
    }
    [pscustomobject]@{ Wrappers = @($wrappers | ForEach-Object { $_.ProcessId }); Tree = @($tree | Select-Object -Unique) }
}

$p = Get-RunProcs
"wrapper PIDs: $($p.Wrappers -join ', ')   training tree PIDs: $($p.Tree -join ', ')"
if (-not $p.Tree.Count) { "no running training process for $Exp"; exit 0 }
if ($DryRun) { "dry run: nothing stopped"; exit 0 }

if (-not $Now) {
    $t0 = (Get-Item $state).LastWriteTime
    "waiting for the next epoch checkpoint (last saved $t0) ..."
    while ((Get-Item $state).LastWriteTime -eq $t0) { Start-Sleep -Seconds 5 }
    Start-Sleep -Seconds 10   # let the write finish (state is saved right after _last.pt)
}
$p = Get-RunProcs
foreach ($id in $p.Wrappers) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
foreach ($id in $p.Tree) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
$rows = (Import-Csv "$outDir\epochs.csv").Count
"[{0}] PAUSED by pause_training.ps1 after {1} completed epochs (state saved {2})" -f (Get-Date -Format o), $rows, (Get-Item $state).LastWriteTime |
    Tee-Object -FilePath "$outDir\attempts.log" -Append
