# One-time setup of the project on Windows with an NVIDIA GPU (G3 on Frederik's PC; docs/tasks/g3_finetuning.md).
# Only Frederik's Windows PC needs this (see frederik/README.md); Macs, Linux and the HPC use "uv sync".
#
# uv.lock covers only macOS and Linux, so "uv sync" and "uv run" do not work on Windows. This script builds a separate
# environment in .venv with the SAME package versions as uv.lock, without changing uv.lock or pyproject.toml:
# 1. Every package version in uv.lock becomes a constraint (uv export --frozen, with the lock's macOS/Linux markers
#    removed), and uv resolves the project's dependencies for Windows under these constraints. Packages that only
#    Windows needs (e.g. colorama) are not in uv.lock and get their newest compatible version.
# 2. torch and torchvision come from PyTorch's own server as CUDA 12.1 builds (--torch-backend cu121): the versions
#    of uv.lock (torch 2.2.2, torchvision 0.17.2), but with CUDA (the Windows torch on PyPI has none).
# 3. psutil is added: on Windows it measures the memory of nnU-Net's worker processes (src/segreview/memory.py).
# 4. It checks that torch can use the GPU (one small 3D convolution in float16; no training), runs the tests, and
#    downloads the TotalSegmentator weights and the 2 scans of the G3 smoke test.
# It does NOT run the smoke test or any training. The installed versions and the results of the checks are saved in
# results/g3_smoke_gpu/windows_env.txt and results/g3_smoke_gpu/setup_check.txt. It can be run again.
#
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File frederik\windows_setup.ps1
# Afterwards, run every script with the environment's Python instead of "uv run":
#   .venv\Scripts\python.exe scripts\<script>.py --config configs\<config>.yaml
#   .venv\Scripts\python.exe -m pytest

$ErrorActionPreference = "Continue"      # native programs report errors by their exit code, checked after each step
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = ".venv\Scripts\python.exe"
$results = Join-Path (Get-Location) "results\g3_smoke_gpu"
$report = Join-Path $results "setup_check.txt"
$utf8 = New-Object Text.UTF8Encoding $false
New-Item -ItemType Directory -Force $results | Out-Null

function Stop-OnError([string]$step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$step failed (exit code $LASTEXITCODE)." -ForegroundColor Red
        exit 1
    }
}

function Add-Report([string[]]$lines) {
    [IO.File]::AppendAllText($report, (($lines -join "`n") + "`n"), $utf8)
}

function Get-PackageKey([string]$name) {
    # Package names compare in lower case with "-", "_" and "." treated alike (PEP 503).
    return ($name.Trim().ToLower() -replace '[-_.]+', '-')
}

# ---- 0. Programs that must be there ------------------------------------------------------------------------------
foreach ($program in "uv", "nvidia-smi") {
    if (-not (Get-Command $program -ErrorAction SilentlyContinue)) {
        Write-Host "$program was not found." -ForegroundColor Red
        exit 1
    }
}
$commit = "unknown (git is not on PATH)"
if (Get-Command git -ErrorAction SilentlyContinue) {
    $commit = (git rev-parse --short HEAD) + " on branch " + (git rev-parse --abbrev-ref HEAD)
    if (git status --porcelain) { $commit += " (with uncommitted changes)" }
} else {
    Write-Host ("Note: git is not on PATH, so machine.json cannot record the git commit. Install Git for Windows " +
                "(winget install --id Git.Git -e) and open a new terminal.") -ForegroundColor Yellow
}
[IO.File]::WriteAllText($report, ("# Windows setup check (frederik/windows_setup.ps1), " +
    (Get-Date -Format "yyyy-MM-dd HH:mm") + "`n# Commit: $commit`n"), $utf8)
$gpu = nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv
Stop-OnError "nvidia-smi"
$gpu | Write-Host
Add-Report (@("", "## GPU (nvidia-smi; memory.used and utilization are from other programs at this moment)") + $gpu)

# ---- 1. The versions of uv.lock as constraints ---------------------------------------------------------------------
$lock = uv export --frozen --no-hashes --no-emit-project --format requirements-txt
Stop-OnError "uv export"
$pins = @($lock | Where-Object { $_ -match '^[A-Za-z0-9]' } | ForEach-Object { ($_ -split ';')[0].Trim() })
$constraints = Join-Path $env:TEMP "segreview-uv-lock-constraints.txt"
[IO.File]::WriteAllLines($constraints, [string[]]$pins, $utf8)
Write-Host "$($pins.Count) package versions from uv.lock are used as constraints."

# ---- 2.-3. The environment ------------------------------------------------------------------------------------------
if (-not (Test-Path $python)) {
    uv venv .venv --python 3.12
    Stop-OnError "Creating .venv"
}
uv pip install --python $python --torch-backend cu121 --constraints $constraints --editable . --group dev psutil
Stop-OnError "Installing the packages"

$freeze = @(uv pip freeze --python $python | Where-Object { $_ -match '^[A-Za-z0-9][^ ]*==' })   # not the project
Stop-OnError "uv pip freeze"
$locked = @{}
foreach ($pin in $pins) {
    $name, $version = $pin -split '==', 2
    $locked[(Get-PackageKey $name)] = $version
}
$differs = @(); $windowsOnly = @()
foreach ($line in $freeze) {
    $name, $version = $line -split '==', 2
    $key = Get-PackageKey $name
    if (-not $locked.ContainsKey($key)) { $windowsOnly += $line }
    elseif (($version -split '\+')[0] -ne $locked[$key]) { $differs += "$line (uv.lock: $($locked[$key]))" }
}
# (In PowerShell "," binds more strongly than "+", so each joined string is in parentheses.)
[IO.File]::WriteAllLines((Join-Path $results "windows_env.txt"), [string[]](@(
    ("# Packages of the Windows environment (.venv), made by frederik/windows_setup.ps1 on " +
     (Get-Date -Format "yyyy-MM-dd") + "."),
    "# Same versions as uv.lock; torch and torchvision as CUDA 12.1 builds; packages not in uv.lock are Windows-only.") +
    $freeze), $utf8)
$differText = if ($differs.Count) { $differs -join ", " } else { "none" }
$windowsOnlyText = if ($windowsOnly.Count) { $windowsOnly -join ", " } else { "none" }
$summary = @("", "## Environment (.venv, see windows_env.txt)",
             "$($freeze.Count) packages installed, $($pins.Count) versions pinned by uv.lock.",
             "Versions that differ from uv.lock: $differText",
             "Packages not in uv.lock (Windows only): $windowsOnlyText")
$summary | Write-Host
Add-Report $summary
if ($differs.Count) {
    Write-Host "Some versions differ from uv.lock (see above)." -ForegroundColor Red
    exit 1
}

# ---- 4a. torch and the GPU (no training) ---------------------------------------------------------------------------
$gpuCheck = @'
import torch

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
if not torch.cuda.is_available():
    raise SystemExit("torch does not see the GPU")
props = torch.cuda.get_device_properties(0)
print(f"GPU {props.name}, compute capability {props.major}.{props.minor}, {props.total_memory / 2**30:.1f} GiB")
# One small 3D convolution in float16, as nnU-Net trains (mixed precision). A GPU that torch 2.2.2 cannot use
# (e.g. the RTX 50 series, Blackwell) fails here with "no kernel image is available for execution on the device".
conv = torch.nn.Conv3d(1, 8, 3, padding=1).cuda()
x = torch.randn(2, 1, 64, 64, 64, device="cuda")
with torch.autocast("cuda", dtype=torch.float16):
    y = conv(x)
torch.cuda.synchronize()
print(f"3D convolution on the GPU in float16: OK, output {tuple(y.shape)}")
'@
$checkFile = Join-Path $env:TEMP "segreview-gpu-check.py"
[IO.File]::WriteAllText($checkFile, $gpuCheck, $utf8)
$out = @(& $python $checkFile 2>&1 | ForEach-Object { "$_" })
$gpuStatus = $LASTEXITCODE
$out | Write-Host
Add-Report (@("", "## torch and the GPU") + $out)
if ($gpuStatus -ne 0) {
    Write-Host "torch cannot use the GPU (see above)." -ForegroundColor Red
    exit 1
}

# ---- 4b. Tests -----------------------------------------------------------------------------------------------------
Write-Host "Running the tests (a few minutes) ..."
& $python -m pytest -q -rs 2>&1 | ForEach-Object { "$_" } | Tee-Object -Variable testLines | Write-Host
$testStatus = $LASTEXITCODE
Add-Report (@("", "## pytest (.venv\Scripts\python.exe -m pytest -q -rs)") + $testLines)

# ---- 4c. Weights and the 2 scans of the smoke test ----------------------------------------------------------------
& $python scripts\download_model.py --config configs\kits100.yaml
Stop-OnError "Downloading the weights"
$cases = (& $python -c "import yaml; print(' '.join(yaml.safe_load(open('configs/g3_smoke_gpu.yaml'))['cases']))") -split ' '
Stop-OnError "Reading the scans of the smoke test"
& $python scripts\download_data.py --config configs\kits100.yaml --cases $cases
Stop-OnError "Downloading the scans of the smoke test"

Write-Host ""
if ($testStatus -ne 0) {
    Write-Host "Setup finished, but the tests FAILED (see $report). Do not push before they pass." -ForegroundColor Red
    exit 1
}
Write-Host "Setup finished; nothing was trained. Next, only after the go-ahead (close games and other GPU programs first):"
Write-Host "  .venv\Scripts\python.exe scripts\g3_smoke_test.py --config configs\g3_smoke_gpu.yaml       # 6 mm, batch 2 and 13"
Write-Host "  .venv\Scripts\python.exe scripts\g3_smoke_test.py --config configs\g3_smoke_gpu_3mm.yaml   # 3 mm, batch 2"
