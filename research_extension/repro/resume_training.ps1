# Resume a paused/crashed experiment from its last full-state checkpoint, detached from VS Code,
# using the exact training arguments recorded in the first line of attempts.log.
#   powershell -ExecutionPolicy Bypass -File resume_training.ps1 -Exp E0IN_P_pidnet_s_seed42
param([Parameter(Mandatory = $true)][string]$Exp)
$outDir = "D:\wildfire_research\experiments\$Exp"
if (-not (Test-Path "$outDir\checkpoints\${Exp}_state.pt")) { throw "no state checkpoint for $Exp" }
$running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train_repro' -and $_.CommandLine -match [regex]::Escape($Exp) }
if ($running) { throw "$Exp is already running (PID $($running.ProcessId -join ', '))" }
$first = Get-Content "$outDir\attempts.log" | Where-Object { $_ -match 'attempt 0 start args: (--protocol .*?) +(--resume +)?log=' } | Select-Object -First 1
if (-not $first) { throw "could not recover training arguments from attempts.log" }
$ta = ([regex]::Match($first, 'start args: (--protocol .*?) +(--resume +)?log=')).Groups[1].Value.Trim()
$argList = "-NoProfile -ExecutionPolicy Bypass -File D:\wildfire_research\research_extension\repro\run_with_resume.ps1 -Exp $Exp -Resume -TrainArgString `"$ta`""
$p = Start-Process powershell.exe -ArgumentList $argList -WindowStyle Hidden -PassThru
"resumed $Exp (wrapper PID $($p.Id)); args: $ta"
