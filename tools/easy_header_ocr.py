#!/usr/bin/env python3
"""
Helper script: Jalankan EasyOCR untuk header detection (kompatibel dengan PyTorch).
Output JSON ke stdout dengan keys: rec_texts, rec_scores
"""
import argparse
import json
import sys
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path ke gambar header crop")
    ap.add_argument("--lang", default="id", help="Language code (id, en, etc)")
    ap.add_argument("--use-gpu", action="store_true", help="Use GPU for OCR")
    args = ap.parse_args()

    try:
        import easyocr

        # EasyOCR menggunakan PyTorch backend, jadi GPU setting otomatis compatible
        gpu = args.use_gpu

        # Inisialisasi reader dengan bahasa Indonesia dan English
        # EasyOCR 'id' = Indonesian, 'en' = English
        langs = ["id", "en"] if args.lang == "id" else ["en"]
        reader = easyocr.Reader(langs, gpu=gpu)

        img_path = args.image

        # EasyOCR bisa langsung baca dari path atau numpy array
        # readtext returns list of (bbox, text, confidence)
        results = reader.readtext(img_path)

        rec_texts = []
        rec_scores = []

        for bbox, text, conf in results:
            rec_texts.append(text)
            rec_scores.append(float(conf))

        output = {
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
            "debug": {
                "num_detections": len(results),
                "gpu_used": gpu,
                "languages": langs,
            },
        }

        sys.stdout.write(json.dumps(output))
        sys.stdout.flush()

    except Exception as e:
        error_output = {"rec_texts": [], "rec_scores": [], "debug": {"error": str(e)}}
        sys.stdout.write(json.dumps(error_output))
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
