import cv2
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import numpy as np
import time

# Config
CAMERA_INDEX = 0  # change to your camera index or set to None to use DroidCam URL
DROIDCAM_URL = "http://192.168.1.3:4747/video"
MODEL_DIR = "donut-ktp-v3"
YOLO_MODEL = "best.pt"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models
print("Loading Donut processor and model...")
processor = DonutProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
model.to(device)

print("Loading YOLO model...")
yolo = YOLO(YOLO_MODEL)

# Open camera
if CAMERA_INDEX is not None:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(DROIDCAM_URL)

if not cap.isOpened():
    print("⚠️ Unable to open camera. Exiting.")
    exit()

print(
    "Camera opened. Press 'c' to capture and OCR the largest detected card, 'q' to quit."
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize preview for display
    disp = cv2.resize(frame, (960, 640))

    # (No per-frame detection to keep preview smooth on CPU)
    # Detection/OCR will run when user presses 'c'

    cv2.imshow("Preview", disp)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        # Capture high-res frame for OCR
        h, w = frame.shape[:2]
        print("Capturing frame and running detection/OCR...")
        t0 = time.time()
        # Run YOLO on original frame to get accurate coordinates
        results = yolo(frame, verbose=False)
        # find largest box
        largest = None
        largest_area = 0
        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res, "boxes") else []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest = (x1, y1, x2, y2)

        if largest is None:
            print("No card detected. Try again.")
            continue

        x1, y1, x2, y2 = largest
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            print("Crop is empty, aborting")
            continue

        # Convert to PIL RGB
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Prepare input
        pixel_values = processor(pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # Generate
        outputs = model.generate(pixel_values, max_length=512, num_beams=4)
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        t1 = time.time()
        print(f"OCR result (took {(t1-t0):.2f}s):\n{decoded}")

        # Show crop
        cv2.imshow("Crop", cv2.resize(crop, (640, 400)))


cap.release()
cv2.destroyAllWindows()
