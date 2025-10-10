import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
from transformers import DonutProcessor, DonutForConditionalGeneration  # Donut classes

# ---------- Config ----------
YOLO_MODEL_PATH = "best_ktp_yolov8.pt"  # ganti ke path model YOLO kamu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DONUT_MODEL_NAME = "naver-clova-ix/donut-base"  # bisa ganti ke model finetuned jika ada
# ----------------------------

# load models
yolo = YOLO(YOLO_MODEL_PATH)
processor = DonutProcessor.from_pretrained(DONUT_MODEL_NAME)
donut = DonutForConditionalGeneration.from_pretrained(DONUT_MODEL_NAME).to(DEVICE)

def detect_ktp_yolo(image_bgr, conf_thresh=0.3):
    """Run YOLO, return list of detections as (x1,y1,x2,y2,score)"""
    results = yolo(image_bgr)  # ultralytics returns a Results object or list
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()  # (N,4)
        scores = r.boxes.conf.cpu().numpy()
        for (box, score) in zip(boxes, scores):
            if score >= conf_thresh:
                x1,y1,x2,y2 = map(int, box)
                dets.append((x1,y1,x2,y2,float(score)))
    return dets

def crop_and_warp_ktp(image_bgr, box, margin=0.05):
    """Crop with margin, then warp using minAreaRect to deskew and crop tight."""
    x1,y1,x2,y2,score = box
    h, w = image_bgr.shape[:2]
    # expand box by margin (relative)
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    rx1, ry1 = max(0, x1-dx), max(0, y1-dy)
    rx2, ry2 = min(w-1, x2+dx), min(h-1, y2+dy)
    crop = image_bgr[ry1:ry2, rx1:rx2].copy()

    # Convert to grayscale and threshold to find largest contour (KTP border)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert if necessary (we want KTP edges visible)
    if np.mean(th) > 127:
        th = 255 - th

    # find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        # fallback: return axis-aligned crop resized
        return cv2.resize(crop, (960,600))

    # pick largest contour by area
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
    box_pts = cv2.boxPoints(rect)  # 4x2
    # shift box points back to original image coords (relative to rx1,ry1)
    box_pts[:,0] = box_pts[:,0] + rx1
    box_pts[:,1] = box_pts[:,1] + ry1

    # compute perspective transform to get top-down rectangle
    # Order points: use helper to order clockwise tl,tr,br,bl
    def order_points(pts):
        pts = pts.reshape(4,2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype="float32")

    pts_src = order_points(box_pts)
    # compute width and height of new image
    widthA = np.linalg.norm(pts_src[2] - pts_src[3])
    widthB = np.linalg.norm(pts_src[1] - pts_src[0])
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(pts_src[1] - pts_src[2])
    heightB = np.linalg.norm(pts_src[0] - pts_src[3])
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0,0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts_src, dst)
    warped = cv2.warpPerspective(image_bgr, M, (maxWidth, maxHeight))
    # Normalize size to Donut expected size (Donut will handle resizing internally via processor)
    return warped

def preprocess_for_donut(image_bgr):
    """Basic cleanups: convert to RGB PIL, maybe enhance contrast/denoise."""
    # Convert to RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    # Optionally: apply CLAHE or slight sharpening in OpenCV (skipped for clarity)
    return pil

def run_donut_ocr(pil_image, task_prompt="ocr"):
    """
    Run Donut to generate text. For vanilla donut, you typically send a prompt
    describing the task like '<s_doc> ... <s_task>...'.
    We'll use processor to prepare inputs.
    """
    # For Donut, many pretrained models expect task_prompt or prompt tokens.
    # Processor will convert image->pixel_values and add required tokens.
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(DEVICE)
    # generate
    generated_ids = donut.generate(pixel_values, max_length=1024, num_beams=4)
    generated_str = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_str

# -----------------------
# Example usage:
# -----------------------
if __name__ == "__main__":
    img_path = "example_ktp_photo.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    dets = detect_ktp_yolo(img, conf_thresh=0.25)
    if not dets:
        print("No KTP detected.")
    else:
        # choose highest-scoring detection
        dets_sorted = sorted(dets, key=lambda x: x[4], reverse=True)
        best = dets_sorted[0]
        warped = crop_and_warp_ktp(img, best)
        pil = preprocess_for_donut(warped)
        # run donut
        ocr_text = run_donut_ocr(pil)
        print("=== Donut raw output ===")
        print(ocr_text)
        # You can parse ocr_text (JSON-like) depending on your Donut fine-tuned task