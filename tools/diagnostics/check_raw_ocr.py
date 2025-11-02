import json

with open("results_latest.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total results: {len(data['results'])}")
print("\n" + "=" * 80)
print("CHECKING FOR PROVINSI/KOTA IN FULL CARD OCR TEXT:")
print("=" * 80)

for i, r in enumerate(data["results"][:5], 1):
    meta = r.get("metadata", {})
    raw_text = meta.get("raw_ocr_text", "")
    filename = meta.get("file_name", "unknown")

    print(f"\n{i}. {filename}")
    print(f"   Full OCR text (first 300 chars):")
    print(f"   {raw_text[:300]}")

    # Check for PROVINSI or KOTA keywords
    has_provinsi = "PROVINSI" in raw_text.upper() or "PROPINSI" in raw_text.upper()
    has_kota = "KOTA" in raw_text.upper() or "KABUPATEN" in raw_text.upper()
    print(f"   Contains PROVINSI: {has_provinsi}")
    print(f"   Contains KOTA/KABUPATEN: {has_kota}")
    print("-" * 80)
