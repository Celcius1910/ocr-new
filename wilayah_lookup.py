"""
Wilayah Lookup Module
Load and match kelurahan/kecamatan from CSV dataset using fuzzy matching
"""

import csv
import re
from pathlib import Path
from difflib import SequenceMatcher

# Cache untuk performa
_VILLAGES = None
_DISTRICTS = None
_REGENCIES = None
_PROVINCES = None


def _normalize_name(name):
    """Normalize nama wilayah untuk matching"""
    if not name:
        return ""
    # Uppercase dan hilangkan prefix KAB./KOTA
    name = name.upper()
    name = re.sub(r"^(KAB\.|KOTA)\s+", "", name)
    name = re.sub(r"^(KABUPATEN|KOTA)\s+", "", name)
    # Bersihkan karakter non-alphanumeric
    name = re.sub(r"[^A-Z\s]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def load_csv_data():
    """Load semua CSV data ke memory (lazy load)"""
    global _VILLAGES, _DISTRICTS, _REGENCIES, _PROVINCES

    if _VILLAGES is not None:
        return  # Already loaded

    base_path = Path(__file__).parent / "wilayah_administrasi_indonesia" / "csv"

    # Load provinces
    _PROVINCES = {}
    with open(base_path / "provinces.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            prov_id = row["id"]
            prov_name = row["name"].strip('"')
            _PROVINCES[prov_id] = _normalize_name(prov_name)

    # Load regencies (kota/kabupaten)
    _REGENCIES = {}
    with open(base_path / "regencies.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            reg_id = row["id"]
            reg_name = row["name"].strip('"')
            prov_id = row["province_id"]
            _REGENCIES[reg_id] = {
                "name": _normalize_name(reg_name),
                "province_id": prov_id,
            }

    # Load districts (kecamatan)
    _DISTRICTS = {}
    with open(base_path / "districts.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            dist_id = row["id"]
            dist_name = row["name"].strip('"')
            reg_id = row["regency_id"]
            _DISTRICTS[dist_id] = {
                "name": _normalize_name(dist_name),
                "regency_id": reg_id,
            }

    # Load villages (kelurahan/desa)
    _VILLAGES = {}
    with open(base_path / "villages.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            vill_id = row["id"]
            vill_name = row["name"].strip('"')
            dist_id = row["district_id"]
            _VILLAGES[vill_id] = {
                "name": _normalize_name(vill_name),
                "district_id": dist_id,
            }


def _generate_ngrams(text, min_n=2, max_n=4):
    """Generate word n-grams from text"""
    words = text.split()
    ngrams = []
    for n in range(min_n, min(max_n + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i : i + n]))
    # Also add individual words for single-word matches
    ngrams.extend(words)
    return ngrams


def fuzzy_match_kelurahan(text, kota_name=None, threshold=0.75):
    """
    Fuzzy match kelurahan/desa dari OCR text
    Uses n-gram tokenization to handle multi-entity text

    Args:
        text: OCR text yang mau di-match
        kota_name: Nama kota/kabupaten untuk filter (opsional)
        threshold: minimum similarity ratio (0-1)

    Returns:
        tuple: (kelurahan_name, confidence) atau (None, 0)
    """
    load_csv_data()

    if not text:
        return None, 0

    text_norm = _normalize_name(text)
    if len(text_norm) < 3:
        return None, 0

    # Filter villages by kota if provided
    candidates = []
    if kota_name:
        kota_norm = _normalize_name(kota_name)
        # Find regency_id yang match (flexible: partial match OK)
        target_reg_ids = []
        for reg_id, reg_data in _REGENCIES.items():
            reg_name = reg_data["name"]
            # Match if either name contains the other (handles KAB/KOTA prefix)
            if kota_norm in reg_name or reg_name in kota_norm:
                target_reg_ids.append(reg_id)
            # Also try word-level match for multi-word names
            kota_words = set(kota_norm.split())
            reg_words = set(reg_name.split())
            if kota_words & reg_words:  # Intersection
                target_reg_ids.append(reg_id)
        # Filter villages by regency
        for vill_id, vill_data in _VILLAGES.items():
            dist_id = vill_data["district_id"]
            if dist_id in _DISTRICTS:
                reg_id = _DISTRICTS[dist_id]["regency_id"]
                if reg_id in target_reg_ids:
                    candidates.append(vill_data["name"])
    else:
        # Semua villages (slower)
        candidates = [v["name"] for v in _VILLAGES.values()]

    # Generate n-grams from input text
    ngrams = _generate_ngrams(text_norm)

    # Fuzzy match against all ngrams
    best_match = None
    best_score = 0

    for ngram in ngrams:
        for candidate in candidates:
            # Try substring match first (faster)
            if candidate in ngram or ngram in candidate:
                score = min(1.0, len(candidate) / max(len(ngram), 1))
                if score > best_score:
                    best_match = candidate
                    best_score = score
            else:
                # Full fuzzy match
                ratio = SequenceMatcher(None, ngram, candidate).ratio()
                if ratio > best_score:
                    best_match = candidate
                    best_score = ratio

    if best_score >= threshold:
        return best_match, best_score

    return None, 0


def fuzzy_match_kecamatan(text, kota_name=None, threshold=0.75):
    """
    Fuzzy match kecamatan dari OCR text
    Uses n-gram tokenization to handle multi-entity text

    Args:
        text: OCR text yang mau di-match
        kota_name: Nama kota/kabupaten untuk filter (opsional)
        threshold: minimum similarity ratio (0-1)

    Returns:
        tuple: (kecamatan_name, confidence) atau (None, 0)
    """
    load_csv_data()

    if not text:
        return None, 0

    text_norm = _normalize_name(text)
    if len(text_norm) < 3:
        return None, 0

    # Filter districts by kota if provided
    candidates = []
    if kota_name:
        kota_norm = _normalize_name(kota_name)
        # Find regency_id yang match (flexible: partial match OK)
        target_reg_ids = []
        for reg_id, reg_data in _REGENCIES.items():
            reg_name = reg_data["name"]
            # Match if either name contains the other (handles KAB/KOTA prefix)
            if kota_norm in reg_name or reg_name in kota_norm:
                target_reg_ids.append(reg_id)
            # Also try word-level match for multi-word names
            kota_words = set(kota_norm.split())
            reg_words = set(reg_name.split())
            if kota_words & reg_words:  # Intersection
                target_reg_ids.append(reg_id)
        # Filter districts by regency
        for dist_id, dist_data in _DISTRICTS.items():
            if dist_data["regency_id"] in target_reg_ids:
                candidates.append(dist_data["name"])
    else:
        # Semua districts (slower)
        candidates = [d["name"] for d in _DISTRICTS.values()]

    # Generate n-grams from input text
    ngrams = _generate_ngrams(text_norm)

    # Fuzzy match against all ngrams
    best_match = None
    best_score = 0

    for ngram in ngrams:
        for candidate in candidates:
            # Try substring match first
            if candidate in ngram or ngram in candidate:
                score = min(1.0, len(candidate) / max(len(ngram), 1))
                if score > best_score:
                    best_match = candidate
                    best_score = score
            else:
                # Full fuzzy match
                ratio = SequenceMatcher(None, ngram, candidate).ratio()
                if ratio > best_score:
                    best_match = candidate
                    best_score = ratio

    if best_score >= threshold:
        return best_match, best_score

    return None, 0


# Test function
if __name__ == "__main__":
    # Test cases dari sample OCR
    test_cases = [
        ("CURUG SANGERENG", "TANGERANG"),
        ("KELAPA DUA", "TANGERANG"),
        ("SEI PUTIH TENGAH", "MEDAN"),
        ("MEDAN PETISAH", "MEDAN"),
        ("BATU SELICIN", "BATAM"),
        ("LUBUK BAJA", "BATAM"),
        ("KEDUNGDORO", "SURABAYA"),
        ("TEGALSARI", "SURABAYA"),
        ("TELAGAMURNI", "BEKASI"),
        ("CIKARANG BARAT", "BEKASI"),
    ]

    print("Testing Wilayah Lookup...")
    print("=" * 80)

    for text, kota in test_cases:
        kel_match, kel_score = fuzzy_match_kelurahan(text, kota, threshold=0.70)
        kec_match, kec_score = fuzzy_match_kecamatan(text, kota, threshold=0.70)

        print(f"\nInput: '{text}' in {kota}")
        print(f"  Kelurahan: {kel_match} (score: {kel_score:.2f})")
        print(f"  Kecamatan: {kec_match} (score: {kec_score:.2f})")
