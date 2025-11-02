"""Debug PaddleOCR output to see what's being detected"""

import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Load models
yolo = YOLO("best.pt")
paddle_ocr = PaddleOCR(use_textline_orientation=True, lang="id")

# Load image
img_path = "datasets/sample_ocr_ktp_axa/Positive - Fotocopyan KTP.jpg"
frame = cv2.imread(img_path)

# Detect KTP
results = yolo(frame, verbose=False)
boxes = results[0].boxes
largest = max(
    boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
)
x1, y1, x2, y2 = map(int, largest.xyxy[0])

margin = 10
x1 = max(0, x1 - margin)
y1 = max(0, y1 - margin)
x2 = min(frame.shape[1], x2 + margin)
y2 = min(frame.shape[0], y2 + margin)

crop = frame[y1:y2, x1:x2]
h_total = crop.shape[0]
header_h = max(120, int(h_total * 0.30))
header_crop = crop[0:header_h, :]

print(f"Header crop size: {header_crop.shape}")

# Run PaddleOCR
result = paddle_ocr.predict(header_crop)

print(f"\n{'='*80}")
print("PaddleOCR rec_texts:")
print(f"{'='*80}")

if result and len(result) > 0:
    result_dict = result[0]
    if isinstance(result_dict, dict):
        rec_texts = result_dict.get("rec_texts", [])
        rec_scores = result_dict.get("rec_scores", [])

        for i, (text, score) in enumerate(zip(rec_texts, rec_scores), 1):
            print(f"{i}. Text: '{text}' | Score: {score:.3f}")

        print(f"\n{'='*80}")
        print("How we're joining them:")
        print(f"{'='*80}")

        header_lines = []
        for text, score in zip(rec_texts, rec_scores):
            if score > 0.5:
                header_lines.append(text.upper())

        print(f"header_lines: {header_lines}")
        print(f"joined: {' '.join(header_lines)}")
