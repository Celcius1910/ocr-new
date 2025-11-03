# Wilayah Lookup (Kelurahan/Kecamatan)

This module provides robust kelurahan/kecamatan extraction using a complete Indonesia administrative CSV dataset and fuzzy matching.

- Module: `wilayah_lookup.py`
- Dataset path: `wilayah_administrasi_indonesia/csv/`
  - `provinces.csv`: id;name
  - `regencies.csv`: id;province_id;name (KAB./KOTA)
  - `districts.csv`: id;regency_id;name (kecamatan)
  - `villages.csv`: id;district_id;name (kelurahan/desa)
- Delimiter: semicolon (`;`)
- Encoding: UTF-8

## How it works

1. Text normalization (uppercase, strip KAB./KOTA prefixes, remove punctuation)
2. Filter candidates by detected Kota/Kabupaten (word-level intersection)
3. N-gram tokenization (1–4 word chunks) over address/body text
4. Fuzzy matching (difflib.SequenceMatcher) with threshold 0.70
5. Return best match per entity (kelurahan, kecamatan)

Integration point:

- Triggered in `run_ocr.py` AFTER kota is populated from header OCR
- Only runs when `kel_desa` or `kecamatan` are still missing

## Quick usage (standalone)

```python
from wilayah_lookup import fuzzy_match_kelurahan, fuzzy_match_kecamatan

kota = "TANGERANG"
alamat = "CURUG SANGERENG KELAPA DUA GADING SERPONG"

kel, ks = fuzzy_match_kelurahan(alamat, kota, threshold=0.70)
kec, cs = fuzzy_match_kecamatan(alamat, kota, threshold=0.70)
print(kel, ks, kec, cs)
```

## Update the CSV dataset

1. Prepare new CSV files with the same schema and delimiter `;`
2. Replace files in `wilayah_administrasi_indonesia/csv/`
3. Ensure UTF-8 encoding (no BOM)
4. No code changes required; the module lazy-loads and caches the dataset

Validation tip (optional):

```powershell
# Quick sanity check for first rows
Get-Content wilayah_administrasi_indonesia/csv/provinces.csv -Head 5
Get-Content wilayah_administrasi_indonesia/csv/regencies.csv -Head 5
Get-Content wilayah_administrasi_indonesia/csv/districts.csv -Head 5
Get-Content wilayah_administrasi_indonesia/csv/villages.csv -Head 5
```

## Tuning and notes

- Default threshold: 0.70 (tuned for OCR noise). Increase to reduce false positives.
- For ambiguous kota names (e.g., TANGERANG), regency filter uses word-level intersection.
- Performance: CSVs are loaded once and cached in memory; subsequent calls are fast.

### Configuration via `config.py`

- Pipeline utama (`run_ocr.py`) mengambil nilai threshold fuzzy dari `config.py`:
  - `FUZZY_WILAYAH_THRESHOLD` (default 0.70)
- Walau fungsi `fuzzy_match_*` memiliki parameter `threshold` (default internal 0.75), call-site di `run_ocr.py` selalu mengirim nilai dari `config.py` agar konsisten.
- Untuk tuning global, ubah angka di `config.py` lalu jalankan ulang pipeline.

## Troubleshooting

- None returned? Check that `kota` was detected and passed to the matcher.
- Wrong match? Try raising threshold to 0.75–0.80.
- Mixed entities in the same line (e.g., multiple kelurahan words): n-gram search resolves this by scoring each chunk separately.
