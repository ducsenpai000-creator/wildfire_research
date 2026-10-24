@echo off
rem Start / continue the experiment queue (resumes from the last saved epoch), then show live progress.
rem Closing the progress view (Ctrl+C) does NOT stop training; use stop.cmd for that.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0research_extension\repro\resume_queue.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\scripts\watch_progress.ps1"
