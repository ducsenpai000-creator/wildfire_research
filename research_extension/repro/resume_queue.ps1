# Resume the unattended experiment queue (detached from VS Code, BelowNormal priority).
#   powershell -ExecutionPolicy Bypass -File resume_queue.ps1
$root = 'D:\wildfire_research'
$running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'queue_runner\.py' }
if ($running) { "queue runner already running (PID $($running.ProcessId -join ', '))"; exit 0 }
Remove-Item "$root\reports\QUEUE_PAUSE" -ErrorAction SilentlyContinue
$p = Start-Process "$root\envs\docs\Scripts\python.exe" -ArgumentList '-I', "$root\tools\scripts\queue_runner.py" `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$root\reports\queue_runner_stdout.log" -RedirectStandardError "$root\reports\queue_runner_stderr.log"
$p.PriorityClass = 'BelowNormal'
"queue runner started (PID $($p.Id))"
