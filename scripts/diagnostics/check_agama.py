import json

with open("outputs/results_latest.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nAgama extraction results:")
print("=" * 80)

for i, result in enumerate(data["results"], 1):
    print(f"{i}. {result['metadata']['file_name']}")
    raw_text = result["metadata"]["raw_ocr_text"]
    print(f"   Raw OCR (first 100 chars): {raw_text[:100]}...")
    print(f"   Agama: {result['data'].get('agama', 'NULL')}")
    print(f"   Confidence: {result['metadata']['field_confidence'].get('agama', 0.0)}")
    print()
