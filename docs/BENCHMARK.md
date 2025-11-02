# KTP OCR - GPU vs CPU Benchmark Results

Date: November 2, 2025
GPU: NVIDIA GeForce RTX 4070 SUPER
Dataset: 7 KTP images (sample_ocr_ktp_axa)

## Configuration

- PyTorch: 2.5.1+cu121 (CUDA 12.1)
- YOLO: Ultralytics 8.3.206
- Donut: VisionEncoderDecoderModel
- PaddleOCR: Disabled (import conflict)

## Benchmark Results

### CPU Mode

```
Command: --yolo-device cpu --donut-device cpu
Total time: 30.21 seconds
Per image: ~4.32 seconds
Device: Intel CPU (oneDNN v3.6.2)
```

### GPU Mode

```
Command: --yolo-device cuda --donut-device cuda
Total time: 14.39 seconds
Per image: ~2.06 seconds
Device: NVIDIA RTX 4070 SUPER (cuda:0)
```

### Performance Comparison

- **Speedup: 2.1x faster**
- **Time saved: 15.82 seconds (52.4% faster)**
- **Throughput:**
  - CPU: 0.23 images/second
  - GPU: 0.49 images/second

## Notes

- First run showed 16x speedup (4.49s total) due to warm cache
- This benchmark includes model loading overhead for fair comparison
- PaddleOCR header extraction disabled (import conflict with paddlepaddle-gpu 2.6.2)
- Both runs used same dataset and parameters
- GPU memory usage: ~1.7 GB VRAM

## Recommendations

**For Production:**

- Use GPU mode (--yolo-device cuda --donut-device cuda)
- 2x faster processing
- Lower CPU usage for parallel tasks
- Better scalability for high-volume processing

**For Development:**

- CPU mode acceptable for testing/debugging
- GPU mode recommended for batch processing
- Consider warm-up run for latency-sensitive applications

## Command Examples

```powershell
# Fastest production mode (GPU, no fallback)
python run_ocr.py --mode folder `
    --input "datasets\sample_ocr_ktp_axa" `
    --output "results.json" `
    --yolo-device cuda --donut-device cuda

# CPU mode (fallback/compatibility)
python run_ocr.py --mode folder `
    --input "datasets\sample_ocr_ktp_axa" `
    --output "results.json" `
    --yolo-device cpu --donut-device cpu
```
