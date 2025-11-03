import cv2
import sys

# Read the image
img_path = "datasets/sample_ocr_ktp_axa/Positive.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]

# Extract header crop (top 50%)
header_h = max(int(h * 0.5), 180)
header_crop = img[:header_h, :]

# Save the header crop
output_path = "outputs/debug/temp_header_crop.png"
cv2.imwrite(output_path, header_crop)
print(f"Header crop saved to: {output_path}")
print(f"Original size: {w}x{h}, Header crop size: {w}x{header_h}")
