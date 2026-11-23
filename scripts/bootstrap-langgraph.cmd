@echo off
rem Phase 0 LangGraph bootstrap for Windows.
rem Uses a process-scoped ExecutionPolicy Bypass so an unsigned .ps1 can run
rem without changing machine/user policy.
rem
rem Usage:
rem   scripts\bootstrap-langgraph.cmd
rem   scripts\bootstrap-langgraph.cmd -DryRun
rem   scripts\bootstrap-langgraph.cmd -IndexUrl "https://pypi.org/simple/"
setlocal
set "SCRIPT=%~dp0bootstrap-langgraph.ps1"
if not exist "%SCRIPT%" (
    echo [ERROR] missing "%SCRIPT%"
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %ERRORLEVEL%
