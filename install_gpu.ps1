# ============================================================
# GPU Setup Script for KTP OCR Project
# Installs CUDA-enabled PyTorch (EasyOCR uses PyTorch)
# Works with the currently active Python or project .venv automatically
# ============================================================

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  GPU SETUP - Installing CUDA Components" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# Resolve Python interpreter preference order:
# 1) .\.venv\Scripts\python.exe (project venv)
# 2) $env:VIRTUAL_ENV\Scripts\python.exe (active venv)
# 3) (Get-Command python).Source (current shell python)

$pythonPath = $null
$projectVenvPy = Join-Path (Get-Location) ".\.venv\Scripts\python.exe"
if (Test-Path $projectVenvPy) {
    $pythonPath = $projectVenvPy
}
elseif ($env:VIRTUAL_ENV) {
    $candidate = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    if (Test-Path $candidate) { $pythonPath = $candidate }
}
if (-not $pythonPath) {
    try { $pythonPath = (Get-Command python -ErrorAction Stop).Source } catch {}
}

if (-not $pythonPath -or -not (Test-Path $pythonPath)) {
    Write-Host "ERROR: Could not find a Python interpreter. Activate your venv or install Python 3.11." -ForegroundColor Red
    Write-Host "Hint: .\\.venv\\Scripts\\Activate.ps1" -ForegroundColor Yellow
    exit 1
}

# Helper to run pip within the same interpreter
function Pip {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & $pythonPath -m pip @Args
}

Write-Host "[1/4] Checking current environment..." -ForegroundColor Yellow
& $pythonPath -c "import sys; print('Python:', sys.version); import site, os; print('Interpreter:', sys.executable)"
& $pythonPath -c "import sys; print(sys.version); import torch; print(torch.__version__, getattr(torch.version, 'cuda', None), torch.cuda.is_available())"
& $pythonPath -c "
try:
 import easyocr; print(f'Current EasyOCR: {easyocr.__version__}')
except Exception as e:
 print('EasyOCR not installed or error:', e)
"

Write-Host "`n[2/4] Uninstalling CPU-only packages..." -ForegroundColor Yellow
Write-Host "  Removing PyTorch CPU..." -ForegroundColor Gray
Pip uninstall -y torch torchvision torchaudio 2>$null

Write-Host "`n[3/4] Installing PyTorch GPU (CUDA 12.1)..." -ForegroundColor Yellow
Write-Host "  This will download ~2-3 GB, please wait..." -ForegroundColor Gray
# Pin to versions compatible with requirements.txt
Pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: PyTorch GPU installation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[4/4] Verifying GPU installation..." -ForegroundColor Yellow
& $pythonPath scripts\diagnostics\check_cuda.py

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Green
Write-Host "  1. Check the diagnostic output above" -ForegroundColor White
Write-Host "  2. If all components show OK, run OCR with GPU (PowerShell):" -ForegroundColor White
Write-Host "     python run_ocr.py --mode folder --input \"datasets\sample_ocr_ktp_axa\" --output \"outputs\results.fullgpu.json\" --yolo-device cuda --donut-device cuda --easyocr-use-gpu" -ForegroundColor Cyan
Write-Host ""
