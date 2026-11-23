#requires -Version 5.1
<#
.SYNOPSIS
    Run the evaluation twice (15-case eval + 30-task taskeval) against a real
    OpenAI-compatible model, then keep the summaries under the chosen index dir.

.DESCRIPTION
    The API key is read from an environment variable; it is never written to
    disk or printed. Set it in the current terminal only:

        $env:DE_LLM_API_KEY = "sk-..."

    DeepSeek is the default; Qwen/Ollama/vLLM only need a different base URL and
    model names. See docs/real-model-rerun.md.

.PARAMETER BaseUrl
    OpenAI-compatible base URL, e.g. https://api.deepseek.com/v1

.PARAMETER Model
    Strong model used for plan/review.

.PARAMETER SmallModel
    Cheap model used for execute. For a first run, set both to the same
    tool-capable model if the strong model does not support function calling.

.PARAMETER IndexDir
    Separate index/report dir so the offline reports are not overwritten.

.PARAMETER DryRun
    Print the commands/environment without calling the API.
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = "https://api.deepseek.com/v1",
    [string]$Model = "deepseek-chat",
    [string]$SmallModel = "deepseek-chat",
    [string]$ApiKeyEnv = "DE_LLM_API_KEY",
    [string]$EnvFile = ".env",
    [string]$EmbedBackend = "hashing",
    [string]$IndexDir = "var/index/real",
    [int]$Limit = 0,
    [switch]$IncludeAblation,
    [switch]$SkipTaskEval,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Optional .env (gitignored). Only fills variables that are not already set;
# values are never printed. Real terminals should prefer the environment.
$envPath = Join-Path $root $EnvFile
if (Test-Path -LiteralPath $envPath) {
    Write-Host "env file: $envPath (values not printed)" -ForegroundColor DarkGray
    Get-Content -LiteralPath $envPath -Encoding UTF8 | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
            $name = $matches[1]
            $value = $matches[2].Trim().Trim('"').Trim("'")
            if (-not [Environment]::GetEnvironmentVariable($name)) {
                [Environment]::SetEnvironmentVariable($name, $value, "Process")
            }
        }
    }
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    $python = $venvPython
}
else {
    $python = (Get-Command python -ErrorAction Stop).Source
}

$key = [Environment]::GetEnvironmentVariable($ApiKeyEnv)
if (-not $DryRun -and [string]::IsNullOrWhiteSpace($key)) {
    throw "环境变量 $ApiKeyEnv 为空。请先在当前窗口设置，例如：`$env:$ApiKeyEnv='sk-...'（不要写进仓库）"
}

$env:DE_LLM_BACKEND = "openai_compat"
$env:DE_LLM_BASE_URL = $BaseUrl
$env:DE_LLM_API_KEY = $key
$env:DE_LLM_MODEL = $Model
$env:DE_SMALL_MODEL = $SmallModel
$env:DE_LLM_TEMPERATURE = "0"
$env:DE_LLM_TIMEOUT_S = "90"
$env:DE_LLM_MAX_TOKENS = "1024"
$env:DE_EMBED_BACKEND = $EmbedBackend
$env:DE_INDEX_DIR = $IndexDir
$env:DE_GRAPH_RUNTIME = "auto"
$env:DE_ENABLE_TRACING = "true"
$env:DE_ENABLE_CONTEXT_BUDGET = "true"
$env:DE_ENABLE_SKILLS = "true"
$env:DE_PERMISSION_MODE = "auto"
$env:DE_REQUIRE_CONFIRMATION = "false"
$env:DE_MAX_ITERATIONS = "6"

$outDir = Join-Path $root $IndexDir
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$evalArgs = @("-m", "de_agent", "eval", "--json", "--progress")
if ($Limit -gt 0) { $evalArgs += @("--limit", "$Limit") }
if (-not $IncludeAblation) { $evalArgs += "--skip-ablation" }
$taskArgs = @("-m", "de_agent", "taskeval", "--json", "--progress")
if ($Limit -gt 0) { $taskArgs += @("--limit", "$Limit") }

Write-Host "== real-model rerun ==" -ForegroundColor Cyan
Write-Host "python:   $python"
Write-Host "base_url: $BaseUrl"
Write-Host "model:    $Model / small=$SmallModel"
Write-Host "key:      from `$$ApiKeyEnv (not printed)"
Write-Host "index:    $outDir"
if ($Limit -gt 0) { Write-Host "limit:    first $Limit cases" -ForegroundColor Yellow }
Write-Host "ablation: $(if ($IncludeAblation) { 'included' } else { 'skipped (faster/cheaper)' })"

if ($DryRun) {
    Write-Host "[dry-run] $python -m de_agent ask '年假有几天？' --json" -ForegroundColor DarkGray
    Write-Host "[dry-run] $python $($evalArgs -join ' ') > $outDir\eval_summary.json" -ForegroundColor DarkGray
    if (-not $SkipTaskEval) {
        Write-Host "[dry-run] $python $($taskArgs -join ' ') > $outDir\task_eval_summary.json" -ForegroundColor DarkGray
    }
    Write-Host "dry-run only; no API call was made." -ForegroundColor Yellow
    return
}

# 1) smoke: catch 401/402/tool-call incompatibility before spending on cases.
& $python -m de_agent ask "年假有几天？" --json | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "smoke ask failed (exit $LASTEXITCODE). Check key/balance/base_url/model."
}
Write-Host "smoke: OK" -ForegroundColor Green
Write-Host "running cases; progress lines below (Ctrl+C only if you want to abort)..." -ForegroundColor DarkGray

# 2) original eval (ablation skipped by default for real-model runs)
& $python @evalArgs |
    Out-File -FilePath (Join-Path $outDir "eval_summary.json") -Encoding utf8
if ($LASTEXITCODE -ne 0) {
    throw "eval failed (exit $LASTEXITCODE)"
}

# 3) task-level 30-case eval
if (-not $SkipTaskEval) {
    & $python @taskArgs |
        Out-File -FilePath (Join-Path $outDir "task_eval_summary.json") -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        throw "taskeval failed (exit $LASTEXITCODE)"
    }
}

Write-Host ""
Write-Host "done. reports:" -ForegroundColor Green
Write-Host "  $outDir\eval_report.md"
if (-not $SkipTaskEval) {
    Write-Host "  $outDir\task_eval_report.md"
    Write-Host "  $outDir\task_eval_summary.json"
}
Write-Host ""
Write-Host "Compare with the FakeLLM baseline before updating README:"
Write-Host "  FakeLLM taskeval: 20/30, tool_missing=6, generation=4"
