"""Debug script to inspect PaddleOCR output for separate regions"""

import cv2
from paddleocr import PaddleOCR
import re

# Initialize PaddleOCR
paddle_ocr = PaddleOCR(use_textline_orientation=True, lang="id")

# Load test image
img_path = "datasets/sample_ocr_ktp_axa/Positive - Fotocopyan KTP.jpg"
img = cv2.imread(img_path)

# Manual crop to simulated card detection (you'd use YOLO result here)
# For this debug, using approximate card region
h, w = img.shape[:2]
crop = img[10:452, 11:726]  # Based on YOLO box from previous run

h_total = crop.shape[0]

# Region 1: Provinsi (0-18%)
provinsi_h = max(70, int(h_total * 0.18))
provinsi_crop = crop[0:provinsi_h, :]

# Region 2: Kota (18-36%)
kota_start = provinsi_h
kota_end = max(140, int(h_total * 0.36))
kota_crop = crop[kota_start:kota_end, :]

print(f"Crop total height: {h_total}px")
print(f"Provinsi region: 0-{provinsi_h}px ({provinsi_h/h_total*100:.1f}%)")
print(
    f"Kota region: {kota_start}-{kota_end}px ({(kota_end-kota_start)/h_total*100:.1f}%)"
)
print()

# OCR on provinsi region
print("=" * 80)
print("PROVINSI REGION OCR:")
print("=" * 80)
prov_result = paddle_ocr.predict(provinsi_crop)
if prov_result and len(prov_result) > 0:
    result_dict = prov_result[0]
    if isinstance(result_dict, dict):
        rec_texts = result_dict.get("rec_texts", [])
        rec_scores = result_dict.get("rec_scores", [])
        for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
            print(f"{i+1}. Text: '{text}' | Score: {score:.3f}")
print()

# OCR on kota region
print("=" * 80)
print("KOTA REGION OCR:")
print("=" * 80)
kota_result = paddle_ocr.predict(kota_crop)
if kota_result and len(kota_result) > 0:
    result_dict = kota_result[0]
    if isinstance(result_dict, dict):
        rec_texts = result_dict.get("rec_texts", [])
        rec_scores = result_dict.get("rec_scores", [])
        for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
            print(f"{i+1}. Text: '{text}' | Score: {score:.3f}")
print()

# Save cropped regions for visual inspection
cv2.imwrite("debug_provinsi_region.jpg", provinsi_crop)
cv2.imwrite("debug_kota_region.jpg", kota_crop)
print("Saved debug images: debug_provinsi_region.jpg, debug_kota_region.jpg")
