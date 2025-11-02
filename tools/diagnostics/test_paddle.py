"""Test PaddleOCR on header region"""

import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Load models
print("Loading models...")
yolo = YOLO("best.pt")
paddle_ocr = PaddleOCR(use_textline_orientation=True, lang="id")

# Load image
img_path = "datasets/sample_ocr_ktp_axa/Positive Clear KTP.jpg"
frame = cv2.imread(img_path)

# Detect KTP with YOLO
results = yolo(frame, verbose=False)
boxes = results[0].boxes
largest = max(
    boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
)
x1, y1, x2, y2 = map(int, largest.xyxy[0])

# Add margin
margin = 10
x1 = max(0, x1 - margin)
y1 = max(0, y1 - margin)
x2 = min(frame.shape[1], x2 + margin)
y2 = min(frame.shape[0], y2 + margin)

crop = frame[y1:y2, x1:x2]
print(f"Card crop size: {crop.shape}")

# Extract header (top 20%)
h_total = crop.shape[0]
header_h = int(h_total * 0.20)
header_crop = crop[0:header_h, :]
print(f"Header crop size: {header_crop.shape}")

# Save header crop for inspection
cv2.imwrite("debug_header_paddle.jpg", header_crop)
print("Saved debug_header_paddle.jpg")

# Run PaddleOCR
print("\nRunning PaddleOCR...")
result = paddle_ocr.predict(header_crop)
print(f"\nPaddleOCR result type: {type(result)}")
print(f"Result length: {len(result) if result else 0}")

if result:
    print(f"\nFirst item type: {type(result[0])}")
    print(f"\nFull result structure:")
    print(result)

    # Try to extract text
    if isinstance(result[0], dict):
        print(f"\nDict keys: {result[0].keys()}")
        rec_text = result[0].get("rec_text", [])
        print(f"\nrec_text type: {type(rec_text)}")
        print(f"rec_text content: {rec_text}")
