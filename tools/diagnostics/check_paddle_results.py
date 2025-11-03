"""
Diagnostic: summarize OCR results JSON (legacy filename)

Note: Despite the name, this script is generic and does not use PaddleOCR.
It summarizes header_ocr_text and provinsi/kota fields from results JSON.
"""

import json

with open("results_latest.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results", [])
print(f"Total files: {len(results)}")

# Count header OCR detections
header_found = sum(1 for r in results if r.get("metadata", {}).get("header_ocr_text"))
print(f"Header OCR found: {header_found}/{len(results)}")

# Count provinsi/kota filled
prov_filled = sum(
    1 for r in results if r.get("data") and r.get("data", {}).get("provinsi")
)
kota_filled = sum(1 for r in results if r.get("data") and r.get("data", {}).get("kota"))
print(f"Provinsi filled: {prov_filled}/{len(results)}")
print(f"Kota filled: {kota_filled}/{len(results)}")

# Show some examples
print("\n" + "=" * 80)
print("Sample provinsi/kota extractions:")
print("=" * 80)
for i, r in enumerate(results[:5], 1):
    fname = r.get("metadata", {}).get("file_name", "unknown")
    header = r.get("metadata", {}).get("header_ocr_text", "N/A")
    data = r.get("data") or {}
    prov = data.get("provinsi", "None")
    kota = data.get("kota", "None")
    print(f"\n{i}. {fname}")
    print(f"   Header OCR: {header}")
    print(f"   Provinsi: {prov}")
    print(f"   Kota: {kota}")
