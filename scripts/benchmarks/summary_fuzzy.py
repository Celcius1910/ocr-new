import json

with open("outputs/test_fuzzy_matching.json", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results", [data])
success_provinsi = 0
success_kota = 0

print("=" * 90)
print("HASIL FUZZY MATCHING - Location Extraction")
print("=" * 90)

for r in results:
    file_name = r.get("metadata", {}).get("file_name", "N/A")
    provinsi = r.get("data", {}).get("provinsi", None)
    kota = r.get("data", {}).get("kota", None) or r.get("data", {}).get(
        "kota_kabupaten", None
    )
    header_text = r.get("metadata", {}).get("header_ocr_text", "")[:60]

    if provinsi:
        success_provinsi += 1
    if kota:
        success_kota += 1

    print(f"\n📄 {file_name}")
    print(f"   Header OCR: {header_text}...")
    print(f"   ✅ Provinsi: {provinsi or '❌ NULL'}")
    print(f"   ✅ Kota: {kota or '❌ NULL'}")

print("\n" + "=" * 90)
print(f"📊 SUMMARY:")
print(
    f"   Provinsi detected: {success_provinsi}/{len(results)} images ({success_provinsi/len(results)*100:.1f}%)"
)
print(
    f"   Kota detected: {success_kota}/{len(results)} images ({success_kota/len(results)*100:.1f}%)"
)
print("=" * 90)

if success_provinsi >= len(results) * 0.85 and success_kota >= len(results) * 0.85:
    print("🎉 EXCELLENT! Fuzzy matching bekerja dengan sangat baik!")
elif success_provinsi >= len(results) * 0.7 and success_kota >= len(results) * 0.7:
    print("✅ GOOD! Fuzzy matching bekerja dengan baik!")
else:
    print("⚠️  Needs improvement")
