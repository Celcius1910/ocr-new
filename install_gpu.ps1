# ============================================================
# GPU Setup Script for KTP OCR Project
# Installs CUDA-enabled PyTorch and PaddlePaddle
# ============================================================

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  GPU SETUP - Installing CUDA Components" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

$venvPath = "c:\OCR AXA\afi-ocr-ktp-code\.venv_system"
$pipPath = "$venvPath\Scripts\pip.exe"
$pythonPath = "$venvPath\Scripts\python.exe"

# Check if venv exists
if (-not (Test-Path $pipPath)) {
    Write-Host "ERROR: Virtual environment not found at $venvPath" -ForegroundColor Red
    Write-Host "Please ensure .venv_system exists" -ForegroundColor Red
    exit 1
}

Write-Host "[1/5] Checking current environment..." -ForegroundColor Yellow
& $pythonPath -c "import torch; print(f'Current PyTorch: {torch.__version__}')"
& $pythonPath -c "import paddle; print(f'Current Paddle: {paddle.__version__}')"

Write-Host "`n[2/5] Uninstalling CPU-only packages..." -ForegroundColor Yellow
Write-Host "  Removing PyTorch CPU..." -ForegroundColor Gray
& $pipPath uninstall -y torch torchvision torchaudio 2>$null

Write-Host "  Removing PaddlePaddle CPU..." -ForegroundColor Gray
& $pipPath uninstall -y paddlepaddle 2>$null

Write-Host "`n[3/5] Installing PyTorch GPU (CUDA 12.1)..." -ForegroundColor Yellow
Write-Host "  This will download ~2-3 GB, please wait..." -ForegroundColor Gray
& $pipPath install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: PyTorch GPU installation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[4/5] Installing PaddlePaddle GPU (CUDA 12.1)..." -ForegroundColor Yellow
Write-Host "  This will download ~500 MB, please wait..." -ForegroundColor Gray
& $pipPath install paddlepaddle-gpu==2.6.1.post121 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nWARNING: PaddlePaddle GPU installation had issues." -ForegroundColor Yellow
    Write-Host "Trying alternative installation method..." -ForegroundColor Yellow
    & $pipPath install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
}

Write-Host "`n[5/5] Verifying GPU installation..." -ForegroundColor Yellow
& $pythonPath scripts\diagnostics\check_cuda.py

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Green
Write-Host "  1. Check the diagnostic output above" -ForegroundColor White
Write-Host "  2. If all components show OK, run OCR with GPU:" -ForegroundColor White
Write-Host "`n     # Fastest (no fallback):" -ForegroundColor Gray
Write-Host '     & ".\.venv_system\Scripts\python.exe" run_ocr.py --mode folder \' -ForegroundColor Cyan
Write-Host '        --input "datasets\sample_ocr_ktp_axa" \' -ForegroundColor Cyan
Write-Host '        --output "outputs\results.fullgpu.json" \' -ForegroundColor Cyan
Write-Host '        --yolo-device cuda --donut-device cuda' -ForegroundColor Cyan
Write-Host "`n     # With fallback (slower but extracts agama/pekerjaan):" -ForegroundColor Gray
Write-Host '     & ".\.venv_system\Scripts\python.exe" run_ocr.py --mode folder \' -ForegroundColor Cyan
Write-Host '        --input "datasets\sample_ocr_ktp_axa" \' -ForegroundColor Cyan
Write-Host '        --output "outputs\results.fullgpu.fallback.json" \' -ForegroundColor Cyan
Write-Host '        --yolo-device cuda --donut-device cuda \' -ForegroundColor Cyan
Write-Host '        --paddle-use-gpu --enable-fallback' -ForegroundColor Cyan
Write-Host ""
