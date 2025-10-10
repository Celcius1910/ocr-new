import cv2
import numpy as np

# Replace with your DroidCam IP (shown in app)
# Example: http://192.168.0.105:4747/video
droidcam_url = "http://192.168.1.3:4747/video"

# Open the video stream
cap = cv2.VideoCapture(droidcam_url)

if not cap.isOpened():
    print("❌ Cannot connect to DroidCam stream.")
    exit()

print("✅ Connected to DroidCam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    # Resize for better performance
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 150)

    # Find contours (possible cards)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Approximate the contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Check for rectangular shape (4 corners)
        if len(approx) == 4 and cv2.contourArea(cnt) > 1000:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            cv2.putText(frame, "Card detected", (approx[0][0][0], approx[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Card Detector (DroidCam)", frame)

    # Exit when pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
