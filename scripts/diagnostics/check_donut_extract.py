import json

with open("outputs/test_gpu.json", encoding="utf-8") as f:
    d = json.load(f)

r = d["results"][6]
print("File:", r["metadata"]["file_name"])
print("Provinsi extracted:", r["data"].get("provinsi", "NULL"))
print("Kota extracted:", r["data"].get("kota_kabupaten", "NULL"))
print(
    "\nRaw text parsed:",
    r["data"].get("raw_text", "NULL")[:500] if r["data"].get("raw_text") else "NULL",
)
