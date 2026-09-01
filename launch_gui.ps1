$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "This launcher needs uv. Install it from https://docs.astral.sh/uv/ and try again."
    exit 1
}

$env:UV_CACHE_DIR = Join-Path $projectRoot ".uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot ".uv-python"
uv python install 3.12
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv sync --python 3.12
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& (Join-Path $projectRoot ".venv\Scripts\python.exe") -m buy_vs_rent.web_server
exit $LASTEXITCODE

