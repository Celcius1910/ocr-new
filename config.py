"""Centralized configuration for OCR thresholds and heuristics.

Adjust values here to tune behavior across the pipeline without changing multiple files.
"""

# Field-level acceptance thresholds applied to computed confidences (0.0–1.0)
# A value is the minimum confidence required for a field to be kept in the final output.
FIELD_THRESHOLDS: dict[str, float] = {
    "nik": 0.90,
    "nama": 0.70,
    "provinsi": 0.80,
    "kota": 0.70,
    "tempat_lahir": 0.60,
    "tanggal_lahir": 0.70,
    "jenis_kelamin": 0.80,
    "alamat": 0.50,
    "rt_rw": 0.80,
    "kel_desa": 0.55,
    "kecamatan": 0.55,
    "agama": 0.70,
    "status_perkawinan": 0.70,
    "pekerjaan": 0.70,
    "kewarganegaraan": 0.90,
    "berlaku_hingga": 0.80,
}

# EasyOCR thresholds
EASY_HEADER_MIN_CONF: float = 0.30  # min conf for header tokens (PROVINSI/KOTA line)
EASY_BODY_MIN_CONF: float = (
    0.40  # min conf for body tokens (kelurahan/kecamatan fallback)
)

# Header crop configuration (relative to detected card crop)
HEADER_CROP_RATIO: float = 0.40  # top portion height ratio for header OCR

# Body ROI (lower-left) where kelurahan/kecamatan often appear (ratios of crop dims)
BODY_ROI_Y_START: float = 0.40
BODY_ROI_Y_END: float = 0.95
BODY_ROI_X_START: float = 0.00
BODY_ROI_X_END: float = 0.75

# Fuzzy matching threshold for wilayah lookup (difflib-based matching in wilayah_lookup)
FUZZY_WILAYAH_THRESHOLD: float = 0.70
