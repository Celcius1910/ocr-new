import json

with open("outputs/results_latest.json", encoding="utf-8") as f:
    r = json.load(f)

print("\nAlamat extraction results:")
print("-" * 80)
for i, res in enumerate(r["results"]):
    fname = res["metadata"]["file_name"]
    alamat = res["data"].get("alamat")
    conf = res["metadata"]["field_confidence"].get("alamat", 0)
    print(f"{i+1}. {fname}")
    print(f"   Alamat: {alamat}")
    print(f"   Confidence: {conf}")
    print()
