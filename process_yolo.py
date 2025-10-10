import cv2
from ultralytics import YOLO

# DroidCam stream (replace with your phone IP)
droidcam_url = "http://192.168.1.3:4747/video"

# Load trained model
model = YOLO("best.pt")

# Open video feed
cap = cv2.VideoCapture(droidcam_url)

if not cap.isOpened():
    print("❌ Cannot connect to DroidCam stream.")
    exit()

print("✅ Connected. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Run YOLO detection
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("Card Detector (YOLO + DroidCam)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
