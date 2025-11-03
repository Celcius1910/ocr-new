"""
Fuzzy Location Matcher
Robust location extraction menggunakan fuzzy string matching
"""

from rapidfuzz import fuzz, process
from indonesia_locations import (
    PROVINSI_INDONESIA,
    KOTA_KABUPATEN_INDONESIA,
    PROVINSI_ALIASES,
)
import re


def normalize_text(text):
    """Normalize text untuk matching yang lebih baik"""
    if not text:
        return ""
    # Uppercase dan clean non-alphanumeric
    text = text.upper()
    text = re.sub(r"[^A-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_concatenated_words(text):
    """
    Split kata-kata yang nempel (OCR error) dengan mencari pattern provinsi/kota
    Contoh: "PBOVINSHJAWABARAT" -> ["JAWA BARAT"]
    """
    if not text:
        return []

    candidates = [text]

    # Check untuk pattern provinsi yang umum
    common_provinsi_words = [
        "JAWA",
        "BARAT",
        "TIMUR",
        "TENGAH",
        "SELATAN",
        "UTARA",
        "SUMATERA",
        "KALIMANTAN",
        "SULAWESI",
        "PAPUA",
        "MALUKU",
        "BALI",
        "ACEH",
        "RIAU",
        "BENGKULU",
        "LAMPUNG",
        "BANTEN",
        "YOGYAKARTA",
        "GORONTALO",
        "JAMBI",
    ]

    # Cari pattern kata provinsi di dalam concatenated text
    for word in common_provinsi_words:
        if word in text:
            # Extract dari posisi word sampai akhir atau sampai ketemu keyword lain
            idx = text.find(word)
            # Ambil 2-3 kata setelah word
            remaining = text[idx:].split()[:3]
            if len(remaining) >= 1:
                candidates.append(" ".join(remaining))

    return candidates


def extract_ngrams(text, max_n=4):
    """
    Extract n-grams dari text untuk matching
    Contoh: "PROVINSI JAWA BARAT" -> ["PROVINSI", "JAWA", "BARAT", "PROVINSI JAWA", "JAWA BARAT", ...]
    """
    words = text.split()
    ngrams = []

    # Single words
    ngrams.extend(words)

    # Bigrams, trigrams, etc
    for n in range(2, min(max_n + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            ngrams.append(ngram)

    return ngrams


def fuzzy_match_provinsi(text, threshold=75):
    """
    Fuzzy match text dengan daftar provinsi Indonesia

    Args:
        text: OCR text yang mau di-match
        threshold: minimum similarity score (0-100)

    Returns:
        tuple: (matched_provinsi, confidence_score) atau (None, 0) jika tidak ada match
    """
    if not text:
        return None, 0

    normalized = normalize_text(text)

    # Check aliases dulu (exact match dengan alias)
    for alias, actual in PROVINSI_ALIASES.items():
        if alias in normalized:
            return actual, 100

    # Extract n-grams dari text
    ngrams = extract_ngrams(normalized, max_n=4)

    # Tambahkan split concatenated words
    split_candidates = split_concatenated_words(normalized)
    for candidate in split_candidates:
        ngrams.extend(extract_ngrams(candidate, max_n=4))

    # Cari best match dari semua n-grams
    best_match = None
    best_score = 0

    for ngram in ngrams:
        # Skip jika terlalu pendek
        if len(ngram) < 3:
            continue

        result = process.extractOne(ngram, PROVINSI_INDONESIA, scorer=fuzz.ratio)

        if result and result[1] > best_score:
            best_match = result[0]
            best_score = result[1]

    # Return hanya jika score >= threshold
    if best_score >= threshold:
        return best_match, best_score

    return None, 0


def fuzzy_match_kota(text, threshold=75):
    """
    Fuzzy match text dengan daftar kota/kabupaten Indonesia

    Args:
        text: OCR text yang mau di-match
        threshold: minimum similarity score (0-100)

    Returns:
        tuple: (matched_kota, confidence_score, kota_type) atau (None, 0, None)
        kota_type bisa "KOTA" atau "KABUPATEN"
    """
    if not text:
        return None, 0, None

    normalized = normalize_text(text)

    # Extract n-grams
    ngrams = extract_ngrams(normalized, max_n=4)

    # Cari best match
    best_match = None
    best_score = 0

    for ngram in ngrams:
        # Skip jika terlalu pendek
        if len(ngram) < 3:
            continue

        result = process.extractOne(ngram, KOTA_KABUPATEN_INDONESIA, scorer=fuzz.ratio)

        if result and result[1] > best_score:
            best_match = result[0]
            best_score = result[1]

    if best_score >= threshold:
        # Tentukan type (KOTA atau KABUPATEN)
        # Default ke KOTA, kecuali nama mengandung pattern kabupaten
        kota_type = "KOTA"

        # List kota yang memang kota (bukan kabupaten)
        kota_names = {
            "MEDAN",
            "JAKARTA",
            "BANDUNG",
            "SURABAYA",
            "SEMARANG",
            "MAKASSAR",
            "PALEMBANG",
            "TANGERANG",
            "BEKASI",
            "DEPOK",
            "BATAM",
            "PADANG",
            "MALANG",
            "BOGOR",
            "YOGYAKARTA",
            "DENPASAR",
            "MANADO",
            "BALIKPAPAN",
            # ... bisa ditambah
        }

        # Jika match dengan kabupaten pattern atau tidak ada di kota_names
        if any(
            word in best_match.split()
            for word in ["SELATAN", "UTARA", "BARAT", "TIMUR", "TENGAH"]
        ):
            # Check apakah ini kabupaten atau kota administratif
            if best_match not in kota_names:
                kota_type = "KABUPATEN"

        return best_match, best_score, kota_type

    return None, 0, None


def extract_location_from_ocr(ocr_text, provinsi_threshold=75, kota_threshold=75):
    """
    Extract provinsi dan kota dari raw OCR text menggunakan fuzzy matching

    Args:
        ocr_text: Full OCR text (bisa dari rec_texts yang di-join atau raw text)
        provinsi_threshold: minimum score untuk provinsi match
        kota_threshold: minimum score untuk kota match

    Returns:
        dict: {
            'provinsi': str atau None,
            'provinsi_confidence': float,
            'kota': str atau None,
            'kota_confidence': float,
            'kota_type': 'KOTA' atau 'KABUPATEN' atau None
        }
    """
    result = {
        "provinsi": None,
        "provinsi_confidence": 0,
        "kota": None,
        "kota_confidence": 0,
        "kota_type": None,
    }

    if not ocr_text:
        return result

    # Match provinsi
    provinsi, prov_score = fuzzy_match_provinsi(ocr_text, threshold=provinsi_threshold)
    if provinsi:
        result["provinsi"] = provinsi
        result["provinsi_confidence"] = prov_score

    # Match kota
    kota, kota_score, kota_type = fuzzy_match_kota(ocr_text, threshold=kota_threshold)
    if kota:
        result["kota"] = kota
        result["kota_confidence"] = kota_score
        result["kota_type"] = kota_type

    return result


# Test function
if __name__ == "__main__":
    # Test cases
    test_cases = [
        "PROVINSI JAWA BARAT KABUPATEN BEKASI",
        "PBOVINSHJAWABARAT KABUPATEN BEKASI",
        "PROVINSI SUMATERA UTARA KOTA MEDAN",
        "KOTA SURABAYA",
        "JAKARTA BARAT",
        "PROVINS BANTEN KABUPATEN TANGERANG",
        "KEPULAUANRIAU KOTA BATAM",
    ]

    print("Testing Fuzzy Location Matching...")
    print("=" * 80)

    for test in test_cases:
        result = extract_location_from_ocr(test)
        print(f"\nInput: {test}")
        print(
            f"  Provinsi: {result['provinsi']} (confidence: {result['provinsi_confidence']:.1f})"
        )
        print(
            f"  Kota: {result['kota_type']} {result['kota']} (confidence: {result['kota_confidence']:.1f})"
        )
