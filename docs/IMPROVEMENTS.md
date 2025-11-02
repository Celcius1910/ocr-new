# OCR KTP Enhancement Summary

## Overview

Sistem OCR KTP menggunakan YOLO untuk deteksi kartu, Donut untuk body text extraction, dan PaddleOCR khusus untuk header (Provinsi/Kota).

## Latest Improvements (November 2025)

### 1. Image Preprocessing ✅

- **CLAHE**: Adaptive contrast enhancement
- **Sharpening**: Kernel-based edge enhancement
- **Denoising**: Fast NlMeans untuk reduce noise
- **Conditional**: Hanya apply jika YOLO confidence < 0.85

### 2. Orientation Detection ✅

- Auto-detect vertical KTP (aspect ratio < 0.8)
- Auto-rotate 90° untuk fix orientation

### 3. Enhanced Crop Margin ✅

### 4. Header OCR & Parsing (NEW) ✅

Mengatasi keterbatasan Donut dalam membaca header dengan menambahkan PaddleOCR untuk strip atas (≈30% tinggi crop) dan parsing yang lebih cerdas:

- Normalisasi token: memisahkan kata yang menempel (KOTAMEDAN → KOTA MEDAN), perbaikan typo umum (PHUVINSI → PROVINSI, JAWA TIMUF → JAWA TIMUR, SUMATERA UTA → SUMATERA UTARA)
- Regex dengan lookahead agar tidak over-capture (berhenti di NIK/NAMA/TEMPAT/TTL/KELAMIN/ALAMAT)
- Canonical province mapping dengan fuzzy matching (SequenceMatcher ≥ 0.75) + alias umum
- Kota cleaner untuk buang noise di ekor (nama orang/huruf lepas) dan tetap menjaga kota dua kata yang valid (JAKARTA SELATAN, TANGERANG SELATAN, KEPULAUAN SERIBU)

Hasil (dataset 19 gambar):

- header_found: 16/19
- provinsi: 11/19
- kota: 13/19

- Added 10px margin around YOLO bounding box
- Prevents text cutoff at edges

### 5. Enhanced Field Parsing ✅

**Significant improvements:**

| Field            | Before       | After         | Improvement  |
| ---------------- | ------------ | ------------- | ------------ |
| **kel_desa**     | 1/17 (5.9%)  | 5/17 (29.4%)  | **+400%** 🎉 |
| **kecamatan**    | 1/17 (5.9%)  | 2/17 (11.8%)  | **+100%**    |
| **agama**        | 0/17 (0.0%)  | 1/17 (5.9%)   | **NEW**      |
| **tempat_lahir** | 2/17 (11.8%) | 10/17 (58.8%) | **+400%**    |

**Total improvement: +12 fields detected (+35% overall)**

### Enhanced Parsing Features:

- **30+ kelurahan database**: TELAGAMURNI, MULYOREJO, NGABETAN, JAJAR, etc.
- **30+ kecamatan database**: CIKARANG BARAT, DEPOK, KEBAYORAN BARU, etc.
- **Fuzzy religion matching**: Handles "KATHOLIK" → "KATOLIK", etc.
- **Smart occupation keywords**: 20+ pekerjaan types (karyawan swasta, PNS, dll)

### 6. Confidence Scoring ✅

**Multi-level confidence:**

- `yolo_confidence`: YOLO bounding box detection confidence
- `ocr_avg_token_prob`: Average token probability from Donut model
- `overall_confidence`: Geometric mean of YOLO × OCR
- `field_confidence`: Per-field validation (regex, format, heuristics)

## Known Limitations

### `pekerjaan` Field

**Status**: ❌ **0% detection rate**

**Root Cause**: Donut model (`donut-ktp-v3`) was **NOT trained** to extract pekerjaan field.

**Evidence**: Analysis of 17 images shows 0 occurrences of occupation keywords in raw OCR output.

**Solutions**:

- **Short-term**: Use business logic/defaults
- **Long-term**: Retrain Donut model with pekerjaan-labeled dataset

## File Structure

### Core Files

- `run_ocr.py` - Main entry point (CLI: file, folder, camera), termasuk merge hasil header
- `process_ktp_json.py` - Field parsing & confidence computation
- `tools/diagnostics/show_headers.py` - Diagnostic: tampilkan OCR header dan hasil parsing per file

### Model Files

- `best.pt` - YOLO v8 trained for KTP card detection
- `donut-ktp-v3/` - Donut vision-encoder-decoder model for OCR

### Results

- `outputs/results_latest.json` - Latest OCR results (contoh)
- `outputs/results.debug.json` - Hasil batch terbaru untuk debugging/diagnostik (contoh)

## Usage

```powershell
# Single file
python run_ocr.py --mode file --input path\to\ktp.jpg --output result.json

# Batch folder
python run_ocr.py --mode folder --input datasets\sample_ocr_ktp_axa --output results.json

# Camera (interactive)
python run_ocr.py --mode camera
```

## Performance

**Test dataset**: 19 images (sample_ocr_ktp_axa)
**Processing time**: ~53 seconds (CPU mode, ~3s per image)
**Success rate**: 17/19 (89.5%)
**Header**: header_found 16/19, provinsi 11/19, kota 13/19

## Next Steps / Future Improvements

1. **Expand kecamatan database** - Add more known kecamatan names
2. **Retrain Donut model** - Include pekerjaan field in training data
3. **GPU optimization** - Enable CUDA for faster processing (~10x speedup)
4. **Post-processing validation** - Cross-check NIK format with provinsi/kota codes
5. **Province/city dictionaries** - Tambah alias/typo umum agar semakin robust

---

**Last Updated**: November 1, 2025
**Version**: 2.1 (Header OCR + Enhanced Parsing)
