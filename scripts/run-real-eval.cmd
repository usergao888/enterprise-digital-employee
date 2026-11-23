@echo off
rem Run the real-model eval without changing the PowerShell execution policy.
rem Usage: scripts\run-real-eval.cmd [-DryRun] [-BaseUrl "..."] [-Model "..."]
setlocal
set "SCRIPT=%~dp0run-real-eval.ps1"
if not exist "%SCRIPT%" (
    echo [ERROR] missing "%SCRIPT%"
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %ERRORLEVEL%
