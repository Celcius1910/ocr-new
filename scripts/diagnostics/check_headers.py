import json

with open("outputs/test_easyocr_final.json", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results", [data])
success_provinsi = 0
success_kota = 0
for r in results:
    file_name = r.get("metadata", {}).get("file_name", "N/A")
    # Check di data section, bukan metadata
    provinsi = r.get("data", {}).get("provinsi", "NULL")
    kota = r.get("data", {}).get("kota", "NULL") or r.get("data", {}).get(
        "kota_kabupaten", "NULL"
    )

    if provinsi and provinsi != "NULL":
        success_provinsi += 1
    if kota and kota != "NULL":
        success_kota += 1

    print(f"{file_name:40} -> Provinsi: {provinsi or 'NULL':30} Kota: {kota or 'NULL'}")

print(f"\n✅ Provinsi detected: {success_provinsi}/{len(results)} images")
print(f"✅ Kota detected: {success_kota}/{len(results)} images")
