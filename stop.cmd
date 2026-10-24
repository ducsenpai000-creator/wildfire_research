@echo off
rem Safe stop: waits until the running job saves its next epoch checkpoint (up to ~12 min), then stops.
rem Wait for the "queue PAUSED" line before shutting the computer down.
rem   stop.cmd now   -> stop immediately (loses only the epoch in progress)
if /i "%1"=="now" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0research_extension\repro\pause_queue.ps1" -Now
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0research_extension\repro\pause_queue.ps1"
)
echo.
echo Da dung an toan. Co the tat may.
