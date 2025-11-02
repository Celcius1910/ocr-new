import json
import sys
import os

# Determine results file to read
default_candidates = [
    "outputs/results_latest.json",
    "outputs/results.debug.json",
    "results_latest.json",
    "results.debug.json",
]

results_path = None
if len(sys.argv) > 1 and sys.argv[1]:
    results_path = sys.argv[1]
else:
    for cand in default_candidates:
        if os.path.exists(cand):
            results_path = cand
            break

if not results_path:
    print(
        "No results JSON found. Provide a path as argument or create outputs/results.debug.json."
    )
    sys.exit(1)

with open(results_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print(f"HEADER OCR TEXT dari semua file (from {os.path.basename(results_path)}):")
print("=" * 80)

for i, r in enumerate(data.get("results", []), 1):
    if r.get("status") == "success":
        fname = r.get("metadata", {}).get("file_name", "unknown")
        header_full = r.get("metadata", {}).get("header_ocr_text", "None")
        prov_line = r.get("metadata", {}).get("header_provinsi_line", "None")
        kota_line = r.get("metadata", {}).get("header_kota_line", "None")

        print(f"\n{i}. {fname}")
        print(f"   Full header OCR: {header_full}")
        print(f"   Parsed provinsi line:         {prov_line}")
        print(f"   Parsed kota line:             {kota_line}")
        print("-" * 80)
