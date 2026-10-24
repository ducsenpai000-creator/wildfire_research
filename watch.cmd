@echo off
rem Live progress view only (read-only; Ctrl+C or closing it does not affect training).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\scripts\watch_progress.ps1"
