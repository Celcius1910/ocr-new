import cv2
from ultralytics import YOLO

# DroidCam stream (replace with your phone IP)
droidcam_url = "http://192.168.1.35:4747/video"

# Load trained model
model = YOLO("best.pt")

# Try DroidCam stream first, then fall back to any available local webcam indices
cap = cv2.VideoCapture(droidcam_url)
use_droidcam = True
if not cap.isOpened():
    print(
        "⚠️ Cannot connect to DroidCam stream. Searching for available local webcams..."
    )
    use_droidcam = False
    cap = None
    # try a set of likely indexes; will pick the first that opens
    for idx in range(0, 6):
        try_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if try_cap.isOpened():
            cap = try_cap
            chosen_idx = idx
            break
        else:
            try_cap.release()

    if cap is None or not cap.isOpened():
        print("❌ Cannot open any local webcam indexes (tried 0-5). Exiting.")
        exit()
    else:
        print(f"✅ Connected to local webcam (index {chosen_idx}). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Run YOLO detection
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    title = (
        "Card Detector (YOLO + DroidCam)"
        if use_droidcam
        else "Card Detector (YOLO + Webcam)"
    )
    cv2.imshow(title, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
