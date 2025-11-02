import cv2
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import json
import time
from datetime import datetime
import re
import argparse
import os

"""process_ktp_json

Utilities for KTP OCR parsing and image processing.

Important usage note:
- Importing this module will NOT open the camera or start any long-running
    operations. The camera loop and model-loading helper are guarded behind
    `if __name__ == '__main__'` and exposed via `main()` / `_camera_loop()`.

- Callers (for example `run_ocr.py`) should load models and pass `processor`,
    `model`, `yolo`, and `device` into `process_image` / `process_image_folder`.

This prevents side effects on import and makes the functions suitable for
unit testing and reuse in other scripts.
"""


def parse_ktp_text(text):
    """Parse raw OCR text into structured KTP fields"""
    # Initialize fields with default None
    fields = {
        "nik": None,
        "nama": None,
        "provinsi": None,
        "kota": None,
        "tempat_lahir": None,
        "tanggal_lahir": None,
        "jenis_kelamin": None,
        "alamat": None,
        "rt_rw": None,
        "kel_desa": None,
        "kecamatan": None,
        "agama": None,
        "status_perkawinan": None,
        "pekerjaan": None,
        "kewarganegaraan": None,
        "berlaku_hingga": None,
    }

    # Split text into lines and join with spaces for better searching
    lines = [line.strip() for line in text.split("\n")]
    text_single = " ".join(lines)
    # Common excluded words that aren't names
    # Common excluded words that aren't names
    excluded_words = {
        "LAKI",
        "PEREMPUAN",
        "KAWIN",
        "BELUM",
        "CERAI",
        "ISLAM",
        "KRISTEN",
        "KATOLIK",
        "HINDU",
        "BUDDHA",
        "KECAMATAN",
        "KELURAHAN",
        "DESA",
        "PROVINSI",
        "KABUPATEN",
        "KOTA",
        "JAKARTA",
        "BANDUNG",
        "BEKASI",
        "BOGOR",
        "DEPOK",
        "TANGERANG",
        "SEUMUR",
        "HIDUP",
        "INDONESIA",
        "TELAGAMURNI",
        "CIKARANG",
        "BARAT",
        "TIMUR",
        "SELATAN",
        "UTARA",
        "RT",
        "RW",
        "PERUM",
        "BLOK",
        "WNI",
        "WNA",
        "JALAN",
        "KOMPLEK",
        "KELAPA",
        "DUA",
        "SATU",
        "TIGA",
        "EMPAT",
        "LIMA",
    }

    # Expanded city list for better tempat_lahir detection
    known_cities = [
        "JAKARTA",
        "BANDUNG",
        "BEKASI",
        "DEPOK",
        "BOGOR",
        "TANGERANG",
        "SURABAYA",
        "MEDAN",
        "SEMARANG",
        "MAKASSAR",
        "PALEMBANG",
        "DENPASAR",
        "YOGYAKARTA",
        "MALANG",
        "SOLO",
        "SURAKARTA",
        "PADANG",
        "PEKANBARU",
        "BANJARMASIN",
        "PONTIANAK",
        "SAMARINDA",
        "MANADO",
        "BATAM",
        "JAMBI",
        "BALIKPAPAN",
        "SUKABUMI",
        "CIREBON",
        "SERANG",
        "CILEGON",
        "LUMAJANG",
        "MORO",
    ]

    known_provinces = [
        "DKI JAKARTA",
        "JAWA BARAT",
        "JAWA TENGAH",
        "JAWA TIMUR",
        "BANTEN",
        "DAERAH ISTIMEWA YOGYAKARTA",
        "BALI",
        "NUSA TENGGARA BARAT",
        "NUSA TENGGARA TIMUR",
        "ACEH",
        "SUMATERA UTARA",
        "SUMATERA BARAT",
        "RIAU",
        "KEPULAUAN RIAU",
        "JAMBI",
        "SUMATERA SELATAN",
        "BENGKULU",
        "LAMPUNG",
        "KEPULAUAN BANGKA BELITUNG",
        "KALIMANTAN BARAT",
        "KALIMANTAN TENGAH",
        "KALIMANTAN SELATAN",
        "KALIMANTAN TIMUR",
        "KALIMANTAN UTARA",
        "SULAWESI UTARA",
        "SULAWESI TENGAH",
        "SULAWESI SELATAN",
        "SULAWESI TENGGARA",
        "GORONTALO",
        "SULAWESI BARAT",
        "MALUKU",
        "MALUKU UTARA",
        "PAPUA",
        "PAPUA BARAT",
    ]

    # Find NIK first as anchor point
    nik_match = re.search(r"\b(\d{16})\b", text_single)
    if nik_match:
        fields["nik"] = nik_match.group(1)
        # Try to find name after NIK
        nik_pos = text_single.find(nik_match.group(1))
        text_after_nik = text_single[nik_pos + 16 :]
        name_match = re.search(
            r"\s*([A-Z][A-Z\s]+?)(?=\s+(?:WNI|WNA|LAKI|PEREMPUAN|ISLAM|KRISTEN|KATOLIK|HINDU|BUDDHA))",
            text_after_nik,
        )
        if name_match:
            candidate_name = name_match.group(1).strip()
            if not any(word in candidate_name.split() for word in excluded_words):
                fields["nama"] = candidate_name

    # If name not found after NIK, try other methods
    if not fields["nama"]:
        # Look for sequences of capital letters that could be names
        for line in lines:
            name_match = re.search(r"([A-Z][A-Z\s]{2,}[A-Z])", line)
            if name_match:
                candidate_name = name_match.group(1).strip()
                if not any(word in candidate_name.split() for word in excluded_words):
                    fields["nama"] = candidate_name
                    break

    # Gender
    if "laki-laki" in text_single.lower() or "laki laki" in text_single.lower():
        fields["jenis_kelamin"] = "LAKI-LAKI"
    elif "perempuan" in text_single.lower():
        fields["jenis_kelamin"] = "PEREMPUAN"

    # Birth info - look for date patterns
    date_pattern = r"\b(\d{2}[-/]\d{2}[-/]\d{4}|\d{2}\s+[A-Za-z]+\s+\d{4})\b"
    date_match = re.search(date_pattern, text_single)
    if date_match:
        fields["tanggal_lahir"] = date_match.group(1).strip()

    # Look for birth place near birth date or from known cities
    if fields["tanggal_lahir"]:
        birth_date_pos = text_single.find(fields["tanggal_lahir"])
        if birth_date_pos != -1:
            text_around_date = text_single[
                max(0, birth_date_pos - 30) : birth_date_pos
                + len(fields["tanggal_lahir"])
                + 30
            ]
            for city in known_cities:
                if city in text_around_date.upper():
                    fields["tempat_lahir"] = city
                    break

    # If still not found, search entire text for known cities
    if not fields["tempat_lahir"]:
        for city in known_cities:
            if city in text_single.upper():
                fields["tempat_lahir"] = city
                break

    # Address components - Position-based extraction
    # More tolerant RT/RW pattern (allows extra spaces)
    rt_rw_pattern = r"\b(?:RT|RW|RT/RW)?\s*:?\s*(\d{3}|\d{2})\s*[/\-]\s*(\d{3}|\d{2})\b"
    rt_rw_match = re.search(rt_rw_pattern, text_single, re.IGNORECASE)
    if rt_rw_match:
        fields["rt_rw"] = f"{rt_rw_match.group(1)}/{rt_rw_match.group(2)}"

    # NEW: Position-based alamat extraction (no hardcoded keywords)
    # Strategy: Extract text between RT/RW and kelurahan/kecamatan markers
    if fields["rt_rw"]:
        rt_pos = text_single.find(fields["rt_rw"])
        if rt_pos != -1:
            # Get text after RT/RW
            text_after_rt = text_single[rt_pos + len(fields["rt_rw"]) :].strip()

            # Find end boundary: kelurahan/kecamatan or known non-address keywords
            end_markers = [
                "KELURAHAN",
                "DESA",
                "KECAMATAN",
                "AGAMA",
                "PEKERJAAN",
                "STATUS",
                "BERLAKU",
                "KAWIN",
                "BELUM KAWIN",
                "CERAI",
                "ISLAM",
                "KRISTEN",
                "KATOLIK",
                "HINDU",
                "BUDDHA",
            ]

            end_pos = len(text_after_rt)
            for marker in end_markers:
                pos = text_after_rt.upper().find(marker)
                if pos != -1 and pos < end_pos:
                    end_pos = pos

            # Extract address chunk
            address_candidate = text_after_rt[:end_pos].strip()

            # Clean: remove trailing RT/RW-like patterns that got captured
            address_candidate = re.sub(r"\s+\d{2,3}/\d{2,3}$", "", address_candidate)

            # Validate: must have reasonable length (10-100 chars) and mixed content
            if 10 <= len(address_candidate) <= 100:
                # Must contain letters and at least some numbers or special chars
                has_letters = bool(re.search(r"[A-Z]", address_candidate.upper()))
                has_content = bool(re.search(r"[A-Z0-9]", address_candidate.upper()))

                if has_letters and has_content:
                    fields["alamat"] = address_candidate.strip()

    # Fallback: heuristic extraction if position-based failed
    if not fields["alamat"] and fields.get("rt_rw"):
        rt_pos = text_single.find(fields["rt_rw"])
        if rt_pos != -1:
            # Take next 30-80 chars after RT/RW as candidate
            text_after = text_single[
                rt_pos + len(fields["rt_rw"]) : rt_pos + len(fields["rt_rw"]) + 80
            ].strip()

            # Split by multiple spaces (likely field boundaries)
            tokens = re.split(r"\s{2,}", text_after)
            if tokens:
                candidate = tokens[0].strip()

                # Validate: reasonable length, has letters, not a known field value
                excluded_patterns = [
                    r"^(KAWIN|BELUM|CERAI|ISLAM|KRISTEN|KATOLIK|WNI|WNA|LAKI|PEREMPUAN)$",
                    r"^\d{16}$",  # NIK
                    r"^\d{2}[-/]\d{2}[-/]\d{4}$",  # date
                ]

                is_excluded = any(
                    re.match(pat, candidate.upper()) for pat in excluded_patterns
                )

                if 10 <= len(candidate) <= 100 and not is_excluded:
                    has_letters = bool(re.search(r"[A-Z]", candidate.upper()))
                    if has_letters:
                        fields["alamat"] = candidate.strip()

    # NEW Fallback 2: No RT/RW found - search after known field markers
    if not fields["alamat"]:
        # Try to find address-like text after known field markers (date, name, etc)
        # Look for sequences with street patterns or reasonable length
        text_upper = text_single.upper()

        # Find position after jenis_kelamin, status_perkawinan, or kewarganegaraan
        search_start = 0
        for marker in ["LAKI-LAKI", "PEREMPUAN", "KAWIN", "BELUM KAWIN", "WNI", "WNA"]:
            pos = text_upper.find(marker)
            if pos != -1:
                search_start = max(search_start, pos + len(marker))

        if search_start > 0:
            text_after_markers = text_single[search_start:].strip()

            # Split by multiple spaces and take reasonable chunks
            chunks = re.split(r"\s{2,}", text_after_markers)

            for chunk in chunks:
                chunk = chunk.strip()
                # Look for address-like patterns
                excluded = [
                    r"^(KAWIN|BELUM|CERAI|ISLAM|KRISTEN|KATOLIK|SEUMUR|HIDUP)$",
                    r"^\d{16}$",
                    r"^\d{2}[-/]\d{2}[-/]\d{4}$",
                ]

                is_excluded = any(re.match(pat, chunk.upper()) for pat in excluded)

                if 15 <= len(chunk) <= 100 and not is_excluded:
                    has_letters = bool(re.search(r"[A-Z]", chunk.upper()))
                    has_numbers = bool(re.search(r"\d", chunk))

                    # Likely an address if has both letters and numbers/slashes
                    if has_letters and (has_numbers or "/" in chunk):
                        fields["alamat"] = chunk.strip()
                        break

    # Post-processing cleanup: remove common prefixes from alamat
    if fields.get("alamat"):
        alamat_clean = fields["alamat"]

        # Remove "SEUMUR HIDUP" prefix if present
        alamat_clean = re.sub(
            r"^SEUMUR\s+HIDUP\s+", "", alamat_clean, flags=re.IGNORECASE
        )

        # Remove date prefix (DD-MM-YYYY or DD/MM/YYYY)
        alamat_clean = re.sub(r"^\d{2}[-/]\d{2}[-/]\d{4}\s+", "", alamat_clean)

        # Remove RT/RW pattern prefix (e.g., "003/020" or "001 / 010")
        alamat_clean = re.sub(r"^\d{2,3}\s*[/\-]\s*\d{2,3}\s+", "", alamat_clean)

        # Remove standalone RT/RW separator remnants like "/-"
        alamat_clean = re.sub(r"^[/\-]+\s*", "", alamat_clean)

        fields["alamat"] = alamat_clean.strip()

    # Kelurahan & Kecamatan - enhanced extraction
    # List of common kelurahan/desa names
    known_kelurahan = [
        "TELAGAMURNI",
        "MULYOREJO",
        "NGABETAN",
        "JAJAR",
        "CAURTUNGGAL",
        "CIKARANG BARAT",
        "CIKARANG TIMUR",
        "CIKARANG UTARA",
        "CIKARANG SELATAN",
        "KEBAYORAN",
        "SENAYAN",
        "MELAWAI",
        "PETOGOGAN",
        "KUNINGAN",
        "MENTENG",
        "TANAH ABANG",
        "GAMBIR",
        "SETIABUDI",
        "TEBET",
        "MATRAMAN",
        "JATINEGARA",
        "CAKUNG",
        "PULO GADUNG",
        "KRAMAT",
        "SENEN",
        "JOHAR BARU",
        "KEMAYORAN",
        "SAWAH BESAR",
        "TAMAN SARI",
    ]

    # List of common kecamatan names
    known_kecamatan = [
        "CIKARANG BARAT",
        "CIKARANG TIMUR",
        "CIKARANG UTARA",
        "CIKARANG SELATAN",
        "KEBAYORAN BARU",
        "KEBAYORAN LAMA",
        "SETIABUDI",
        "TANAH ABANG",
        "MENTENG",
        "GAMBIR",
        "SENEN",
        "JOHAR BARU",
        "KEMAYORAN",
        "SAWAH BESAR",
        "TAMAN SARI",
        "TAMBORA",
        "CENGKARENG",
        "GROGOL PETAMBURAN",
        "KEBON JERUK",
        "KEMBANGAN",
        "PALMERAH",
        "DEPOK",
        "SUKMAJAYA",
        "PANCORAN MAS",
        "BEJI",
        "CILODONG",
        "LIMO",
        "CINERE",
        "CIPAYUNG",
        "BOJONG SARI",
        "TAPOS",
        "CIMANGGIS",
    ]

    # First look for explicit mentions with "KELURAHAN", "DESA", or "KECAMATAN"
    for line in lines:
        line_upper = line.upper()
        if "KELURAHAN" in line_upper or "DESA" in line_upper:
            extracted = line_upper.replace("KELURAHAN", "").replace("DESA", "").strip()
            # Clean up: remove common non-location words
            extracted = re.sub(r"\b(RT|RW|BLOK|JL|JALAN|NO)\b.*", "", extracted).strip()
            if extracted and len(extracted) > 3:
                fields["kel_desa"] = extracted
        elif "KECAMATAN" in line_upper:
            extracted = line_upper.replace("KECAMATAN", "").strip()
            extracted = re.sub(r"\b(RT|RW|BLOK|JL|JALAN|NO)\b.*", "", extracted).strip()
            if extracted and len(extracted) > 3:
                fields["kecamatan"] = extracted

    # If not found with explicit keywords, try to match known kelurahan names
    if not fields["kel_desa"]:
        text_upper = text_single.upper()
        for kel in known_kelurahan:
            if kel in text_upper:
                fields["kel_desa"] = kel
                break

    # Try to match known kecamatan names
    if not fields["kecamatan"]:
        text_upper = text_single.upper()
        for kec in known_kecamatan:
            if kec in text_upper:
                fields["kecamatan"] = kec
                break

    # Religion - enhanced with variations and fuzzy matching
    # Try word boundary matching first for clean extraction
    religion_patterns = [
        ("ISLAM", r"\b(?:ISLAM|MOSLEM|MUSLIM)\b"),
        ("KRISTEN", r"\b(?:KRISTEN|PROTESTAN|CHRISTIAN)\b"),
        ("KATOLIK", r"\b(?:KATOLIK|KATHOLIK|CATHOLIC)\b"),
        ("HINDU", r"\b(?:HINDU)\b"),
        ("BUDDHA", r"\b(?:BUDDHA|BUDHA|BUDDHIST)\b"),
        ("KONGHUCU", r"\b(?:KONGHUCU|KHONGHUCU|CONFUCIUS)\b"),
    ]

    text_upper = text_single.upper()
    for religion_name, pattern in religion_patterns:
        if re.search(pattern, text_upper):
            fields["agama"] = religion_name
            break

    # Fallback: search in raw text without word boundaries (for OCR errors)
    if not fields["agama"]:
        text_lower = text_single.lower()
        simple_patterns = {
            "ISLAM": ["islam", "moslem", "muslim"],
            "KRISTEN": ["kristen", "protestan"],
            "KATOLIK": ["katolik", "katholik", "catholi"],
            "HINDU": ["hindu"],
            "BUDDHA": ["buddha", "budha"],
            "KONGHUCU": ["konghucu", "khonghucu"],
        }
        for religion, patterns in simple_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    fields["agama"] = religion
                    break
            if fields["agama"]:
                break

    # Pekerjaan - Position-based extraction (like alamat)
    # Strategy: Find position markers and extract the text between them
    # Common markers after pekerjaan: KEWARGANEGARAAN, BERLAKU HINGGA
    text_upper = text_single.upper()

    # Method 1: Look for "PEKERJAAN" label explicitly
    pekerjaan_match = re.search(
        r"\bPEKERJAAN\s*[:;]?\s*([A-Z/\s\-]+?)(?=\s*(?:KEWARGANEGARAAN|BERLAKU|WNI|SEUMUR|$))",
        text_upper,
    )
    if pekerjaan_match:
        pekerjaan_candidate = pekerjaan_match.group(1).strip()
        if 3 <= len(pekerjaan_candidate) <= 50:  # Reasonable length for occupation
            fields["pekerjaan"] = pekerjaan_candidate

    # Method 2: Position-based after status_perkawinan/agama
    # Typical layout: ...AGAMA <religion> PEKERJAAN <job> KEWARGANEGARAAN...
    if not fields["pekerjaan"]:
        # Find position after AGAMA or STATUS PERKAWINAN markers
        markers = ["AGAMA", "KAWIN", "BELUM KAWIN", "CERAI"]
        for marker in markers:
            marker_pos = text_upper.find(marker)
            if marker_pos != -1:
                # Extract text after marker (skip the marker itself and any religion/status)
                after_marker = text_upper[marker_pos + len(marker) :].strip()
                # Skip known religion/status values
                skip_terms = [
                    "ISLAM",
                    "KRISTEN",
                    "KATOLIK",
                    "HINDU",
                    "BUDDHA",
                    "KONGHUCU",
                    "BELUM KAWIN",
                    "KAWIN",
                    "CERAI",
                    "LAKI-LAKI",
                    "PEREMPUAN",
                ]
                for skip in skip_terms:
                    if after_marker.startswith(skip):
                        after_marker = after_marker[len(skip) :].strip()

                # Now extract until end marker
                end_match = re.search(
                    r"^([A-Z/\s\-]+?)(?=\s*(?:KEWARGANEGARAAN|WNI|BERLAKU|SEUMUR|$))",
                    after_marker,
                )
                if end_match:
                    pekerjaan_candidate = end_match.group(1).strip()
                    # Validate: reasonable length and not a known other field
                    if 3 <= len(pekerjaan_candidate) <= 50:
                        # Make sure it's not actually kewarganegaraan or other field
                        if not any(
                            x in pekerjaan_candidate
                            for x in ["KEWARGANEGARAAN", "BERLAKU", "SEUMUR", "HIDUP"]
                        ):
                            fields["pekerjaan"] = pekerjaan_candidate
                            break

    # Marital Status
    if "belum kawin" in text_single.lower():
        fields["status_perkawinan"] = "BELUM KAWIN"
    elif "kawin" in text_single.lower():
        fields["status_perkawinan"] = "KAWIN"
    elif "cerai" in text_single.lower():
        fields["status_perkawinan"] = "CERAI"

    # Header parsing: PROVINSI / KABUPATEN / KOTA (moved out of marital-status block)
    def _normalize_region_name(s: str) -> str:
        s = s.upper()
        # Drop common punctuation and extra tokens
        s = s.replace("ADM.", " ").replace(" ADM ", " ")
        s = re.sub(r"[^A-Z\s-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip(" -:")
        return s

    # Alias mapping for province variants and common short forms/typos
    prov_alias = {
        "DAERAH KHUSUS IBUKOTA JAKARTA": "DKI JAKARTA",
        "DKI": "DKI JAKARTA",
        "DI YOGYAKARTA": "DAERAH ISTIMEWA YOGYAKARTA",
        "DIY": "DAERAH ISTIMEWA YOGYAKARTA",
        "KEPRI": "KEPULAUAN RIAU",
        "KEP RIAU": "KEPULAUAN RIAU",
        "BANGKA BELITUNG": "KEPULAUAN BANGKA BELITUNG",
    }

    def _canonical_province(cand: str) -> str | None:
        cand = (cand or "").upper().strip()
        if not cand:
            return None
        # Quick alias pass
        if cand in prov_alias:
            return prov_alias[cand]
        # Common OCR typos fix-ups
        fixes = {
            "JAWABARA": "JAWA BARAT",
            "JAWATENGAK": "JAWA TENGAH",
            "JAWA TIMUF": "JAWA TIMUR",
            "SUMATERA UTA": "SUMATERA UTARA",
            "OAERAH ISTIMEWA YOGYAKARTA": "DAERAH ISTIMEWA YOGYAKARTA",
            "BAN7EN": "BANTEN",
            "BANTFN": "BANTEN",
        }
        if cand in fixes:
            return fixes[cand]
        # Fuzzy match against known province list
        known_set = {
            "DKI JAKARTA",
            "JAWA BARAT",
            "JAWA TENGAH",
            "JAWA TIMUR",
            "BANTEN",
            "DAERAH ISTIMEWA YOGYAKARTA",
            "BALI",
            "NUSA TENGGARA BARAT",
            "NUSA TENGGARA TIMUR",
            "ACEH",
            "SUMATERA UTARA",
            "SUMATERA BARAT",
            "RIAU",
            "KEPULAUAN RIAU",
            "JAMBI",
            "SUMATERA SELATAN",
            "BENGKULU",
            "LAMPUNG",
            "KEPULAUAN BANGKA BELITUNG",
            "KALIMANTAN BARAT",
            "KALIMANTAN TENGAH",
            "KALIMANTAN SELATAN",
            "KALIMANTAN TIMUR",
            "KALIMANTAN UTARA",
            "SULAWESI UTARA",
            "SULAWESI TENGAH",
            "SULAWESI SELATAN",
            "SULAWESI TENGGARA",
            "GORONTALO",
            "SULAWESI BARAT",
            "MALUKU",
            "MALUKU UTARA",
            "PAPUA",
            "PAPUA BARAT",
        }
        try:
            import difflib

            cand_ns = cand.replace(" ", "")
            best = None
            best_s = 0.0
            for p in known_set:
                s = difflib.SequenceMatcher(None, cand_ns, p.replace(" ", "")).ratio()
                if s > best_s:
                    best_s, best = s, p
            if best_s >= 0.75:
                return best
        except Exception:
            pass
        return None

    def _clean_kota(name: str) -> str:
        """Post-process extracted kota/kabupaten to remove trailing noise like names or stray letters,
        while preserving valid multi-word regions (e.g., JAKARTA SELATAN, TANGERANG SELATAN, KEPULAUAN SERIBU).
        """
        if not name:
            return name
        tokens = [t for t in name.split() if t]
        if not tokens:
            return name

        # Remove trailing single-letter tokens or tokens with punctuation remnants
        while tokens and (len(tokens[-1]) <= 2 or not tokens[-1].isalpha()):
            tokens.pop()
        if not tokens:
            return None

        # Preserve specific valid multi-word regions
        keep_n = 1
        if len(tokens) >= 2:
            t0, t1 = tokens[0], tokens[1]
            dir_words = {"SELATAN", "UTARA", "TIMUR", "BARAT", "PUSAT", "TENGAH"}
            if t0 == "JAKARTA" and t1 in dir_words:
                keep_n = 2
            elif t0 == "TANGERANG" and t1 == "SELATAN":
                keep_n = 2
            elif t0 == "KEPULAUAN" and t1 == "SERIBU":
                keep_n = 2

        # Truncate beyond the kept tokens
        tokens = tokens[:keep_n]

        return " ".join(tokens) if tokens else None

    for i, line in enumerate(lines):
        up = line.upper()
        # Province: handle PROVINSI/PROPINSI line or when name is on next line
        if ("PROVINSI" in up or "PROPINSI" in up) and not fields["provinsi"]:
            after = re.split(r"PROVINSI|PROPINSI", up, maxsplit=1)[-1].strip(" :-.")
            if not after and i + 1 < len(lines):
                after = lines[i + 1].upper().strip(" :-. ")
            # Stop at KABUPATEN/KOTA if on same line
            after = re.split(r"\bKABUPATEN\b|\bKOTA\b", after)[0].strip()
            cand = _normalize_region_name(after)
            if cand:
                # Map aliases and typos, then fuzzy-canon if needed
                cand = prov_alias.get(cand, cand)
                canon = _canonical_province(cand)
                if canon:
                    fields["provinsi"] = canon
                elif cand in known_provinces or len(cand) >= 3:
                    fields["provinsi"] = cand

        # Kota/Kabupaten: handle same-line and next-line layouts
        if ("KABUPATEN" in up or "KOTA" in up) and not fields["kota"]:
            if "KABUPATEN" in up:
                name = up.split("KABUPATEN", 1)[1].strip(" :-. ")
            else:
                name = up.split("KOTA", 1)[1].strip(" :-. ")
            if not name and i + 1 < len(lines):
                name = lines[i + 1].upper().strip(" :-. ")
            # Remove trailing noise tokens (stop at common non-city keywords)
            name = re.split(
                r"\bPROVINSI\b|\bPROPINSI\b|\bKECAMATAN\b|\bKELURAHAN\b|\bDESA\b|\bNIK\b|\bNAMA\b|\bTEMPAT\b|\bTTL\b",
                name or "",
            )[0].strip()
            name = _normalize_region_name(name)
            if name:
                cleaned = _clean_kota(name)
                if cleaned:
                    fields["kota"] = cleaned

    # Fallback: if provinsi still empty, scan text for any known province tokens
    if not fields["provinsi"]:
        full_up = text_single.upper()
        # Exact contains first
        for prov in known_provinces:
            if prov in full_up:
                fields["provinsi"] = prov
                break
        # Fuzzy fallback near header keywords
        if not fields["provinsi"]:
            # Try to grab a short slice around any 'PROVINSI/PROPINSI' mention
            m = re.search(r"(PROVINSI|PROPINSI)\s+([A-Z\s-]{3,40})", full_up)
            if m:
                cand = m.group(2).strip(" -:.")
                # Stop at next common keyword
                cand = re.split(
                    r"\bKABUPATEN\b|\bKOTA\b|\bNIK\b|\bNAMA\b|\bTTL\b|\bLAHIR\b", cand
                )[0].strip()
                canon = _canonical_province(cand)
                if canon:
                    fields["provinsi"] = canon
    if "wni" in text_single.lower():
        fields["kewarganegaraan"] = "WNI"
    elif "wna" in text_single.lower():
        fields["kewarganegaraan"] = "WNA"

    # Validity
    if "seumur hidup" in text_single.lower():
        fields["berlaku_hingga"] = "SEUMUR HIDUP"

    return fields


def compute_field_confidence(fields):
    """
    Compute confidence score (0.0 to 1.0) for each parsed field.
    Uses heuristics: regex validation, format checks, completeness.
    """
    confidences = {}

    # NIK: 16 digits
    if fields.get("nik"):
        nik = fields["nik"]
        if re.fullmatch(r"\d{16}", nik):
            confidences["nik"] = 1.0
        elif re.fullmatch(r"\d+", nik):
            confidences["nik"] = 0.5  # numeric but wrong length
        else:
            confidences["nik"] = 0.2
    else:
        confidences["nik"] = 0.0

    # Nama: at least 2 words, all caps letters
    if fields.get("nama"):
        nama = fields["nama"]
        words = nama.split()
        if len(words) >= 2 and all(w.isupper() and w.isalpha() for w in words):
            confidences["nama"] = 1.0
        elif len(words) >= 2:
            confidences["nama"] = 0.7
        elif len(words) == 1 and len(nama) > 2:
            confidences["nama"] = 0.4
        else:
            confidences["nama"] = 0.2
    else:
        confidences["nama"] = 0.0

    # Tempat lahir: known city or caps word
    if fields.get("tempat_lahir"):
        tempat = fields["tempat_lahir"]
        known_cities = [
            "JAKARTA",
            "BANDUNG",
            "BEKASI",
            "DEPOK",
            "BOGOR",
            "TANGERANG",
            "SURABAYA",
            "MEDAN",
        ]
        if tempat.upper() in known_cities:
            confidences["tempat_lahir"] = 1.0
        elif tempat.isupper() and len(tempat) > 2:
            confidences["tempat_lahir"] = 0.7
        else:
            confidences["tempat_lahir"] = 0.4
    else:
        confidences["tempat_lahir"] = 0.0

    # Tanggal lahir: DD-MM-YYYY or DD/MM/YYYY
    if fields.get("tanggal_lahir"):
        tgl = fields["tanggal_lahir"]
        if re.fullmatch(r"\d{2}[-/]\d{2}[-/]\d{4}", tgl):
            confidences["tanggal_lahir"] = 1.0
        elif re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", tgl):
            confidences["tanggal_lahir"] = 0.7
        else:
            confidences["tanggal_lahir"] = 0.3
    else:
        confidences["tanggal_lahir"] = 0.0

    # Jenis kelamin: must be LAKI-LAKI or PEREMPUAN
    if fields.get("jenis_kelamin"):
        jk = fields["jenis_kelamin"]
        if jk in ["LAKI-LAKI", "PEREMPUAN"]:
            confidences["jenis_kelamin"] = 1.0
        else:
            confidences["jenis_kelamin"] = 0.3
    else:
        confidences["jenis_kelamin"] = 0.0

    # Alamat: position and length-based confidence (no keyword dependency)
    if fields.get("alamat"):
        alamat = fields["alamat"]
        length = len(alamat)

        # Scoring based on characteristics
        score = 0.0

        # Length scoring (sweet spot: 15-60 chars)
        if 15 <= length <= 60:
            score = 1.0
        elif 10 <= length < 15 or 60 < length <= 80:
            score = 0.7
        elif length < 10:
            score = 0.3
        else:  # > 80 (likely overcapture)
            score = 0.4

        # Boost if has typical address patterns (optional indicators, not required)
        optional_indicators = [
            "JL",
            "JALAN",
            "GG",
            "GANG",
            "BLOK",
            "PERUM",
            "KOMP",
            "NO",
            "RT",
        ]
        if any(ind in alamat.upper() for ind in optional_indicators):
            score = min(1.0, score + 0.2)

        # Boost if has numbers (street numbers, block numbers)
        if re.search(r"\d", alamat):
            score = min(1.0, score + 0.1)

        confidences["alamat"] = round(score, 2)
    else:
        confidences["alamat"] = 0.0

    # RT/RW: pattern XXX/XXX or XX/XX
    if fields.get("rt_rw"):
        rt_rw = fields["rt_rw"]
        if re.fullmatch(r"\d{2,3}/\d{2,3}", rt_rw):
            confidences["rt_rw"] = 1.0
        else:
            confidences["rt_rw"] = 0.3
    else:
        confidences["rt_rw"] = 0.0

    # Kel/Desa: caps word
    if fields.get("kel_desa"):
        kel = fields["kel_desa"]
        if kel.isupper() and len(kel) > 2:
            confidences["kel_desa"] = 0.9
        else:
            confidences["kel_desa"] = 0.5
    else:
        confidences["kel_desa"] = 0.0

    # Kecamatan: caps words
    if fields.get("kecamatan"):
        kec = fields["kecamatan"]
        if kec.isupper() and len(kec) > 2:
            confidences["kecamatan"] = 0.9
        else:
            confidences["kecamatan"] = 0.5
    else:
        confidences["kecamatan"] = 0.0

    # Provinsi: should match known list or be uppercase multi-word
    if fields.get("provinsi"):
        prov = fields["provinsi"].upper()
        known_prov = {
            "DKI JAKARTA",
            "JAWA BARAT",
            "JAWA TENGAH",
            "JAWA TIMUR",
            "BANTEN",
            "DAERAH ISTIMEWA YOGYAKARTA",
            "BALI",
            "NUSA TENGGARA BARAT",
            "NUSA TENGGARA TIMUR",
            "ACEH",
            "SUMATERA UTARA",
            "SUMATERA BARAT",
            "RIAU",
            "KEPULAUAN RIAU",
            "JAMBI",
            "SUMATERA SELATAN",
            "BENGKULU",
            "LAMPUNG",
            "KEPULAUAN BANGKA BELITUNG",
            "KALIMANTAN BARAT",
            "KALIMANTAN TENGAH",
            "KALIMANTAN SELATAN",
            "KALIMANTAN TIMUR",
            "KALIMANTAN UTARA",
            "SULAWESI UTARA",
            "SULAWESI TENGAH",
            "SULAWESI SELATAN",
            "SULAWESI TENGGARA",
            "GORONTALO",
            "SULAWESI BARAT",
            "MALUKU",
            "MALUKU UTARA",
            "PAPUA",
            "PAPUA BARAT",
        }
        if prov in known_prov:
            confidences["provinsi"] = 1.0
        elif prov.isupper() and len(prov) >= 3:
            confidences["provinsi"] = 0.7
        else:
            confidences["provinsi"] = 0.4
    else:
        confidences["provinsi"] = 0.0

    # Kota/Kabupaten: uppercase region name
    if fields.get("kota"):
        k = fields["kota"]
        if k.isupper() and len(k) >= 3:
            confidences["kota"] = 0.9
        else:
            confidences["kota"] = 0.5
    else:
        confidences["kota"] = 0.0

    # Agama: known religions
    if fields.get("agama"):
        agama = fields["agama"]
        known_religions = ["ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA", "KONGHUCU"]
        if agama.upper() in known_religions:
            confidences["agama"] = 1.0
        else:
            confidences["agama"] = 0.4
    else:
        confidences["agama"] = 0.0

    # Status perkawinan: known values
    if fields.get("status_perkawinan"):
        status = fields["status_perkawinan"]
        if status in ["BELUM KAWIN", "KAWIN", "CERAI"]:
            confidences["status_perkawinan"] = 1.0
        else:
            confidences["status_perkawinan"] = 0.4
    else:
        confidences["status_perkawinan"] = 0.0

    # Pekerjaan: if present (usually missing in parsing)
    if fields.get("pekerjaan"):
        confidences["pekerjaan"] = 0.7  # basic heuristic
    else:
        confidences["pekerjaan"] = 0.0

    # Kewarganegaraan: WNI or WNA
    if fields.get("kewarganegaraan"):
        kwn = fields["kewarganegaraan"]
        if kwn in ["WNI", "WNA"]:
            confidences["kewarganegaraan"] = 1.0
        else:
            confidences["kewarganegaraan"] = 0.4
    else:
        confidences["kewarganegaraan"] = 0.0

    # Berlaku hingga: SEUMUR HIDUP or date
    if fields.get("berlaku_hingga"):
        berlaku = fields["berlaku_hingga"]
        if berlaku == "SEUMUR HIDUP":
            confidences["berlaku_hingga"] = 1.0
        elif re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", berlaku):
            confidences["berlaku_hingga"] = 0.9
        else:
            confidences["berlaku_hingga"] = 0.4
    else:
        confidences["berlaku_hingga"] = 0.0

    return confidences


def create_response(success, fields=None, error=None, metadata=None):
    """Create a standardized API-like response"""
    response = {
        "status": "success" if success else "error",
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
        "data": fields if success else None,
        "error": error if not success else None,
    }
    return response


def process_image(image_path, processor, model, yolo, device):
    """Process a single image file and return OCR results"""
    t0 = time.time()

    # Read image
    if isinstance(image_path, str):
        frame = cv2.imread(image_path)
        if frame is None:
            return create_response(False, error=f"Failed to read image: {image_path}")
    else:
        frame = image_path

    try:
        # Run YOLO on frame
        results = yolo(frame, verbose=False)

        # Find largest box
        largest = None
        largest_area = 0
        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res, "boxes") else []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest = (x1, y1, x2, y2)

        if largest is None:
            return create_response(False, error="No card detected in image")

        x1, y1, x2, y2 = largest
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return create_response(False, error="Invalid crop region")

        # Convert to PIL RGB
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Prepare input
        pixel_values = processor(pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # Generate with task prompt
        task_prompt = "<s_ktp>"
        outputs = model.generate(
            pixel_values,
            max_length=512,
            num_beams=4,
            decoder_start_token_id=processor.tokenizer.bos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        # Decode
        decoded_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Parse into structured fields
        ktp_fields = parse_ktp_text(decoded_text)

        # Compute confidences and default thresholds (simple baseline here)
        field_confidences = compute_field_confidence(ktp_fields)
        default_thresholds = {
            "nik": 0.90,
            "nama": 0.70,
            "provinsi": 0.80,
            "kota": 0.70,
            "tempat_lahir": 0.60,
            "tanggal_lahir": 0.70,
            "jenis_kelamin": 0.80,
            "alamat": 0.60,
            "rt_rw": 0.80,
            "kel_desa": 0.60,
            "kecamatan": 0.60,
            "agama": 0.70,
            "status_perkawinan": 0.70,
            "pekerjaan": 0.70,
            "kewarganegaraan": 0.90,
            "berlaku_hingga": 0.80,
        }

        def _apply_thresholds(fields: dict, conf: dict, thr: dict) -> dict:
            out = {}
            for k, v in (fields or {}).items():
                if v in (None, ""):
                    out[k] = None
                    continue
                c = float(conf.get(k, 0.0))
                m = float(thr.get(k, 0.60))
                out[k] = v if c >= m else None
            return out

        processed_fields = _apply_thresholds(
            ktp_fields, field_confidences, default_thresholds
        )

        # Create success response with metadata
        processing_time = time.time() - t0
        metadata = {
            "processing_time": f"{processing_time:.2f}s",
            "image_size": f"{frame.shape[1]}x{frame.shape[0]}",
            "crop_size": f"{crop.shape[1]}x{crop.shape[0]}",
            "raw_ocr_text": decoded_text,
            "field_confidence": field_confidences,
            "field_thresholds": default_thresholds,
        }

        return create_response(True, fields=processed_fields, metadata=metadata)

    except Exception as e:
        return create_response(False, error=str(e))


def process_image_folder(folder_path, processor, model, yolo, device):
    """Process all images in a folder and return results"""
    results = []
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    # Get list of image files
    image_files = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(folder_path, file))

    if not image_files:
        return create_response(False, error=f"No image files found in {folder_path}")

    # Process each image
    t0 = time.time()
    for image_path in image_files:
        result = process_image(image_path, processor, model, yolo, device)
        result["metadata"]["file_name"] = os.path.basename(image_path)
        results.append(result)

    total_time = time.time() - t0

    # Calculate per-image timing statistics
    successful_results = [r for r in results if r["status"] == "success"]
    if successful_results:
        processing_times = []
        for r in successful_results:
            if "processing_time" in r["metadata"]:
                # Extract numeric value from "X.XXs" format
                time_str = r["metadata"]["processing_time"]
                time_val = float(time_str.rstrip("s"))
                processing_times.append(time_val)

        avg_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0
        )
        min_time = min(processing_times) if processing_times else 0
        max_time = max(processing_times) if processing_times else 0
    else:
        avg_time = min_time = max_time = 0

    # Create summary response
    metadata = {
        "total_images": len(image_files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "total_processing_time": f"{total_time:.2f}s",
        "avg_time_per_image": f"{avg_time:.2f}s",
        "min_time": f"{min_time:.2f}s",
        "max_time": f"{max_time:.2f}s",
    }

    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata,
        "results": results,
    }


# Config
MODEL_DIR = "models/donut-ktp-v3"
YOLO_MODEL = "models/best.pt"
CAMERA_INDEX = 0  # change to your camera index or set to None to use DroidCam URL
DROIDCAM_URL = "http://192.168.1.3:4747/video"


def _camera_loop(processor, model, yolo, device):
    """Internal: simple camera loop used when running this module directly."""
    # Open camera
    if CAMERA_INDEX is not None:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(DROIDCAM_URL)

    if not cap.isOpened():
        print(
            json.dumps(create_response(False, error="Failed to open camera"), indent=2)
        )
        return

    print(
        "Camera opened. Press 'c' to capture and OCR the largest detected card, 'q' to quit."
    )

    last_response = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print(
                json.dumps(
                    create_response(False, error="Failed to grab frame"), indent=2
                )
            )
            break

        # Resize preview for display
        disp = cv2.resize(frame, (960, 640))

        # If we have a last response, show a preview of the JSON
        if last_response:
            # Draw a semi-transparent overlay
            overlay = disp.copy()
            cv2.rectangle(overlay, (0, 0), (400, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, disp, 0.5, 0, disp)

            # Show brief status
            status_text = (
                "✅ Success" if last_response["status"] == "success" else "❌ Error"
            )
            cv2.putText(
                disp,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if last_response["status"] == "success" else (0, 0, 255),
                2,
            )
            if last_response["status"] == "success" and last_response["data"]["nik"]:
                cv2.putText(
                    disp,
                    f"NIK: {last_response['data']['nik']}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        cv2.imshow("Preview (c=capture, q=quit)", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            print("\nCapturing frame and running detection/OCR...")
            t0 = time.time()

            last_response = process_image(frame, processor, model, yolo, device)
            if last_response["status"] == "success":
                last_response["metadata"][
                    "processing_time"
                ] = f"{time.time() - t0:.2f}s"
            print(json.dumps(last_response, indent=2))

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Run camera mode when this file is executed directly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    print("Loading Donut processor and model...")
    processor = DonutProcessor.from_pretrained(MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
    model.to(device)

    print("Loading YOLO model...")
    yolo = YOLO(YOLO_MODEL)

    _camera_loop(processor, model, yolo, device)


if __name__ == "__main__":
    main()
