import json

with open("outputs/test_easyocr_v2.json", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results", [data])
for r in results:
    file_name = r.get("metadata", {}).get("file_name", "N/A")
    provinsi = r.get("data", {}).get("header_provinsi_line", "NULL")
    kota = r.get("data", {}).get("header_kota_line", "NULL")
    header_text = r.get("metadata", {}).get("header_ocr_text", "NULL")[:80]
    print(f"{file_name:40}")
    print(f"  Header: {header_text}...")
    print(f"  Provinsi: {provinsi}")
    print(f"  Kota: {kota}")
    print()
