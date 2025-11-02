#!/usr/bin/env python3
"""Helper script to check pekerjaan extraction results"""
import json
import sys


def check_pekerjaan(json_file="outputs/results.position_based.json"):
    """Display pekerjaan extraction results"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n{'='*80}")
    print(f"PEKERJAAN EXTRACTION RESULTS")
    print(f"{'='*80}\n")

    for i, result in enumerate(data["results"], 1):
        fname = result["metadata"]["file_name"]
        pekerjaan = result["data"].get("pekerjaan")
        confidence = result["metadata"]["field_confidence"].get("pekerjaan", 0.0)
        raw_text = result["metadata"].get("raw_ocr_text", "")

        print(f"{i}. {fname}")
        print(f"   Pekerjaan: {pekerjaan}")
        print(f"   Confidence: {confidence}")
        print(f"   Raw OCR: {raw_text[:100]}...")
        print()


if __name__ == "__main__":
    json_file = (
        sys.argv[1] if len(sys.argv) > 1 else "outputs/results.position_based.json"
    )
    check_pekerjaan(json_file)
