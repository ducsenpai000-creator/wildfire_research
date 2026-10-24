# Run train_repro.py; if it crashes after saving state, resume (max $MaxRetries times). All attempts are logged.
# Runs at BelowNormal priority so the desktop/VS Code stays responsive.
param(
    [Parameter(Mandatory = $true)][string]$Exp,
    [string[]]$TrainArgs = @(),
    [string]$TrainArgString = '',   # alternative for Start-Process -File (space-separated)
    [int]$MaxRetries = 3,
    [switch]$Resume
)
if ($TrainArgString) { $TrainArgs = $TrainArgString -split '\s+' | Where-Object { $_ } }
if (-not $TrainArgs) { throw 'no training arguments given' }
$py = 'D:\wildfire_research\envs\wfseg_py38\Scripts\python.exe'
$script = 'D:\wildfire_research\research_extension\repro\train_repro.py'
$outDir = "D:\wildfire_research\experiments\$Exp"
New-Item -ItemType Directory -Force $outDir | Out-Null
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'   # inherited by python and its DataLoader workers
$attempt = 0
$extra = @()
if ($Resume) { $extra = @('--resume') }
while ($true) {
    $log = Join-Path $outDir ("train_{0}_attempt{1}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'), $attempt)
    "[{0}] attempt {1} start args: {2} {3} log={4}" -f (Get-Date -Format o), $attempt, ($TrainArgs -join ' '), ($extra -join ' '), $log | Add-Content (Join-Path $outDir 'attempts.log')
    & $py -W ignore $script --exp $Exp @TrainArgs @extra *> $log
    $code = $LASTEXITCODE
    "[{0}] attempt {1} exit code {2}" -f (Get-Date -Format o), $attempt, $code | Add-Content (Join-Path $outDir 'attempts.log')
    if ($code -eq 0) { exit 0 }
    $attempt++
    if ($attempt -gt $MaxRetries -or -not (Test-Path "$outDir\checkpoints\${Exp}_state.pt")) { exit $code }
    $extra = @('--resume')
    Start-Sleep -Seconds 30
}
