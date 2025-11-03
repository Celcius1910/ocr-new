# KTP OCR - Benchmark Results (CPU vs GPU vs Hybrid)

**Updated:** November 3, 2025  
**GPU:** NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)  
**Dataset:** 7 KTP images (sample_ocr_ktp_axa)  
**Environment:** Windows 11, Python 3.11.9, PyTorch 2.4.1+cu121, CUDA 12.1

---

## Ringkasan Hasil

| Mode                   | Config                                   | Avg (s)  | Speedup  | Provinsi  | Kota | Use Case                     |
| ---------------------- | ---------------------------------------- | -------- | -------- | --------- | ---- | ---------------------------- |
| **🏆 Hybrid Accurate** | YOLO half + Donut FP32 + beam 4 + inline | **2.26** | **5.0×** | 5/7 (71%) | 7/7  | **Rekomendasi produksi**     |
| ⚡ GPU Fast            | YOLO half + Donut FP16 + beam 2 + inline | **2.03** | **5.6×** | 5/7 (71%) | 7/7  | Batch besar, prioritas speed |
| 🎯 GPU Normal          | Subprocess EasyOCR                       | 4.55     | 2.5×     | 6/7 (86%) | 7/7  | Akurasi provinsi maksimal    |
| 💻 CPU Fast            | Inline EasyOCR + beam 2                  | 4.11     | 2.7×     | 5/7 (71%) | 7/7  | Tanpa GPU, cepat             |
| CPU Normal             | Subprocess EasyOCR + beam 4              | 11.30    | 1.0×     | 6/7 (86%) | 7/7  | Baseline (paling lambat)     |

**Catatan Speedup:** Relatif terhadap CPU Normal (11.30s baseline)

---

## Detail Benchmark

### 🏆 Hybrid Accurate (REKOMENDASI)

**Config:**

- YOLO: CUDA FP16 (`--yolo-half`)
- Donut: CUDA FP32 full precision
- EasyOCR: Inline reuse (`--easyocr-inline`)
- Beam width: 4 (default, akurat)
- Max length: 512 (default)

**Hasil:**

```
Avg time: 2.257 seconds/image
Min: 1.64s | Max: 2.75s
Provinsi: 5/7 (71.4%)
Kota: 7/7 (100%)
```

**Command:**

```powershell
python run_ocr.py --mode folder --input datasets\sample_ocr_ktp_axa --output outputs\results.json --yolo-device cuda --donut-device cuda --easyocr-inline --yolo-half --donut-beams 4
```

**Kenapa Rekomendasi?**

- ✅ Balance optimal: 2× lebih cepat dari GPU Normal, akurasi Donut maksimal
- ✅ Inline EasyOCR: Eliminasi subprocess overhead (~2s saving)
- ✅ YOLO FP16: Speed boost tanpa menurunkan deteksi
- ✅ Donut FP32 + beam 4: Kualitas OCR terjaga

---

### ⚡ GPU Fast

**Config:**

- YOLO: CUDA FP16
- Donut: CUDA FP16 (`--donut-fp16`)
- EasyOCR: Inline
- Beam width: 2 (cepat)
- Max length: 448

**Hasil:**

```
Avg time: 2.034 seconds/image
Min: 1.49s | Max: 2.47s
Provinsi: 5/7 (71.4%)
Kota: 7/7 (100%)
```

**Command:**

```powershell
python run_ocr.py --mode folder --input datasets\sample_ocr_ktp_axa --output outputs\results.json --yolo-device cuda --donut-device cuda --easyocr-inline --yolo-half --donut-fp16 --donut-beams 2 --donut-max-length 448
```

**Use Case:** Batch processing ratusan/ribuan gambar, provinsi bukan field kritis

---

### 🎯 GPU Normal (Akurasi Maksimal)

**Config:**

- YOLO: CUDA FP32
- Donut: CUDA FP32
- EasyOCR: Subprocess + GPU (`--easyocr-use-gpu`)
- Beam width: 4
- Max length: 512

**Hasil:**

```
Avg time: 4.553 seconds/image
Min: 4.31s | Max: 5.29s
Provinsi: 6/7 (85.7%)
Kota: 7/7 (100%)
```

**Command:**

```powershell
python run_ocr.py --mode folder --input datasets\sample_ocr_ktp_axa --output outputs\results.json --yolo-device cuda --donut-device cuda --easyocr-use-gpu
```

**Use Case:** Prioritas akurasi provinsi maksimal, waktu proses tidak masalah

---

### 💻 CPU Fast

**Config:**

- YOLO: CPU
- Donut: CPU
- EasyOCR: Inline (CPU)
- Beam width: 2
- Max length: 448

**Hasil:**

```
Avg time: 4.111 seconds/image
Min: 3.62s | Max: 4.60s
Provinsi: 5/7 (71.4%)
Kota: 7/7 (100%)
```

**Command:**

```powershell
python run_ocr.py --mode folder --input datasets\sample_ocr_ktp_axa --output outputs\results.json --yolo-device cpu --donut-device cpu --easyocr-inline --donut-beams 2 --donut-max-length 448
```

**Use Case:** Sistem tanpa GPU, batch kecil-menengah (<100 gambar/hari)

**Catatan:** CPU Fast (4.1s) hampir sama cepat dengan GPU Normal (4.5s) karena inline EasyOCR!

---

### CPU Normal (Baseline)

**Config:**

- YOLO: CPU
- Donut: CPU
- EasyOCR: Subprocess (CPU)
- Beam width: 4
- Max length: 512

**Hasil:**

```
Avg time: 11.304 seconds/image
Min: 8.39s | Max: 13.69s
Provinsi: 6/7 (85.7%)
Kota: 7/7 (100%)
```

**Command:**

```powershell
python run_ocr.py --mode folder --input datasets\sample_ocr_ktp_axa --output outputs\results.json --yolo-device cpu --donut-device cpu
```

**Use Case:** Fallback untuk kompatibilitas, single image processing

---

## Analisis Performa

### Faktor Speedup

| Optimasi           | Dampak            | Implementasi                                         |
| ------------------ | ----------------- | ---------------------------------------------------- |
| **Inline EasyOCR** | ~2.0-2.5× faster  | `--easyocr-inline` (hindari subprocess overhead)     |
| **YOLO FP16**      | ~1.1-1.2× faster  | `--yolo-half` (CUDA only)                            |
| **Donut FP16**     | ~1.1-1.2× faster  | `--donut-fp16` (CUDA only, minor accuracy trade-off) |
| **Beam 2 vs 4**    | ~1.2-1.3× faster  | `--donut-beams 2` (akurasi turun sedikit)            |
| **Max length 448** | ~1.05-1.1× faster | `--donut-max-length 448`                             |

### Trade-off Akurasi

**Provinsi 5/7 vs 6/7:**

- Perbedaan: 1 gambar (71% vs 86%)
- Gambar yang hilang: biasanya low-quality/blur
- Kota tetap sempurna: 7/7 di semua mode
- **Trade-off acceptable** untuk speedup 2-5×

**Rekomendasi:**

- Jika provinsi kritis: GPU Normal (6/7)
- Jika kota yang kritis: Semua mode (7/7)
- Produksi umum: Hybrid Accurate (5/7 cukup, 2× faster)

---

## Memory & Resource

| Mode        | VRAM    | CPU Usage   | Model Loading  |
| ----------- | ------- | ----------- | -------------- |
| GPU (semua) | ~1.7 GB | Low         | ~2-3s overhead |
| CPU (semua) | 0 GB    | High (100%) | ~2-3s overhead |

**Catatan:**

- First run: +2-3s overhead (model loading)
- Subsequent runs: Stable performance
- GPU memory: Constant ~1.7 GB untuk YOLO + Donut
- Inline EasyOCR: +500MB RAM (CPU/GPU shared)

---

## Rekomendasi Per Skenario

### 1. Produksi API (High Volume)

**Mode:** Hybrid Accurate  
**Alasan:** Balance terbaik speed & akurasi, throughput ~27 gambar/menit

### 2. Batch Processing Nightly (Ratusan Gambar)

**Mode:** GPU Fast  
**Alasan:** Tercepat, ~30 gambar/menit, provinsi bukan blocker

### 3. Interactive Upload (Single Image)

**Mode:** Hybrid Accurate  
**Alasan:** Response <3s acceptable, akurasi bagus

### 4. Server Tanpa GPU

**Mode:** CPU Fast  
**Alasan:** ~15 gambar/menit, tidak perlu investasi GPU

### 5. Research/QA (Akurasi Maksimal)

**Mode:** GPU Normal  
**Alasan:** Provinsi 6/7, waktu tidak masalah

---

## Hardware Requirements

### Minimum (CPU Mode)

- Processor: Intel i5 / AMD Ryzen 5
- RAM: 8 GB
- Storage: 5 GB
- Performance: ~4-11 s/image

### Recommended (GPU Mode)

- Processor: Intel i7 / AMD Ryzen 7
- RAM: 16 GB
- GPU: NVIDIA RTX 3060+ (4GB+ VRAM)
- Storage: 10 GB
- Performance: ~2-5 s/image

### Tested Configuration

- GPU: NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)
- Driver: 581.57 (CUDA 13.0)
- PyTorch: 2.4.1+cu121
- Performance: 2.03-4.55 s/image (tergantung mode)

---

## Tips Optimasi

1. **Pre-load models** di startup untuk avoid first-run overhead
2. **Batch processing** lebih efisien dari single-image repeated calls
3. **Inline EasyOCR** wajib untuk produksi (2× speedup gratis)
4. **Monitor VRAM** jika parallel processing (1.7GB per instance)
5. **CPU Fast** sangat kompetitif untuk deployment tanpa GPU

---

**Kesimpulan:** Hybrid Accurate mode adalah sweet spot terbaik untuk mayoritas use case produksi.
