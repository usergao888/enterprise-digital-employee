#requires -Version 5.1
<#
.SYNOPSIS
    Phase 0 bootstrap: create a repo-local .venv and verify the real LangGraph API.

.DESCRIPTION
    This script does NOT modify de_agent source code. It only:
      1. creates enterprise-agent/.venv
      2. installs numpy + langgraph + langgraph-checkpoint-sqlite
      3. runs scripts/langgraph_smoke.py
      4. runs the existing unittest suite as a baseline

    After it passes, send the smoke output (versions + API behavior) back so the
    Phase 0 migration patch can be written against the exact installed versions.

.PARAMETER PythonVersion
    Version passed to the py launcher, default 3.11.

.PARAMETER VenvDir
    Venv directory relative to the repo root, default .venv.

.PARAMETER IndexUrl
    Optional pip index URL. Empty means "use pip's configured index".

.PARAMETER Recreate
    Delete and recreate the venv.

.PARAMETER SkipTests
    Skip the unittest baseline (smoke test still runs).

.PARAMETER DryRun
    Print commands without creating the venv or installing anything.

.EXAMPLE
    .\scripts\bootstrap-langgraph.ps1

.EXAMPLE
    .\scripts\bootstrap-langgraph.ps1 -DryRun

.EXAMPLE
    .\scripts\bootstrap-langgraph.ps1 -IndexUrl "https://pypi.org/simple/"
#>
[CmdletBinding()]
param(
    [string]$PythonVersion = "3.11",
    [string]$VenvDir = ".venv",
    [string]$IndexUrl = "",
    [switch]$Recreate,
    [switch]$SkipTests,
    [switch]$DryRun,
    [switch]$NoProxy
)

$ErrorActionPreference = "Stop"

if ($NoProxy) {
    Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
    Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
    Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
    Write-Host "proxy env cleared for this run (-NoProxy)" -ForegroundColor Yellow
}
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "== enterprise-agent / LangGraph bootstrap ==" -ForegroundColor Cyan
Write-Host "repo: $root"
if ($DryRun) { Write-Host "mode: dry-run (no files changed)" -ForegroundColor Yellow }

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $display = "$Exe $($Arguments -join ' ')"
    if ($DryRun) {
        Write-Host "[dry-run] $display" -ForegroundColor DarkGray
        return
    }
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "command failed (exit $LASTEXITCODE): $display"
    }
}

# ---- resolve a usable Python (py launcher first, then python) --------------
$candidates = @()
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    $candidates += , @($pyLauncher.Source, @("-$PythonVersion"))
}
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    $candidates += , @($pythonCmd.Source, @())
}

$selected = $null
$versionText = ""
foreach ($cand in $candidates) {
    $exe = $cand[0]
    $pre = $cand[1]
    $probe = & $exe @pre -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($LASTEXITCODE -eq 0 -and $probe) {
        $selected = $cand
        $versionText = ($probe | Select-Object -First 1).Trim()
        break
    }
}
if (-not $selected) {
    throw "No usable Python found. Install Python 3.10+ or pass -PythonVersion."
}

$pyExe = $selected[0]
$pyPre = $selected[1]
Write-Host "python: $versionText ($pyExe $($pyPre -join ' '))"

$parts = $versionText.Split(".")
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 10)) {
    throw "Python >= 3.10 required, got $versionText"
}

# ---- venv ----------------------------------------------------------------
$venvPath = Join-Path $root $VenvDir
if (-not $venvPath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "VenvDir must stay inside the repo: $venvPath"
}
$venvPy = Join-Path $venvPath "Scripts\python.exe"

if ($Recreate -and (Test-Path -LiteralPath $venvPath)) {
    Write-Host "removing $venvPath" -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host "[dry-run] Remove-Item -LiteralPath $venvPath -Recurse -Force" -ForegroundColor DarkGray
    }
    else {
        Remove-Item -LiteralPath $venvPath -Recurse -Force
    }
}

if (-not (Test-Path -LiteralPath $venvPy)) {
    if (Test-Path -LiteralPath $venvPath) {
        # Incomplete venv: remove it before recreating.
        Write-Host "removing incomplete venv: $venvPath" -ForegroundColor Yellow
        if (-not $DryRun) {
            Remove-Item -LiteralPath $venvPath -Recurse -Force
        }
    }
    Invoke-Checked $pyExe ($pyPre + @("-m", "venv", $venvPath))
}
else {
    Write-Host "venv exists: $venvPath"
}

# ---- dependencies --------------------------------------------------------
$requirements = Join-Path $root "scripts\requirements-langgraph.txt"
$installArgs = @("-m", "pip", "install", "-r", $requirements)
if ($IndexUrl) {
    $installArgs += @("--index-url", $IndexUrl)
}
Invoke-Checked $venvPy $installArgs

# ---- smoke test ----------------------------------------------------------
Invoke-Checked $venvPy @("scripts\langgraph_smoke.py")

# ---- baseline regression -------------------------------------------------
if (-not $SkipTests) {
    Invoke-Checked $venvPy @("-m", "unittest", "discover", "-s", "tests", "-t", ".")
}

Write-Host ""
Write-Host "Phase 0 environment check finished." -ForegroundColor Green
Write-Host "Next: copy the version lines + smoke output into the migration issue/PR,"
Write-Host "then write the real-LangGraph adapter against those exact versions."
