import cv2
from ultralytics import YOLO
import torch
import time

# Enable optimizations
torch.set_num_threads(4)  # Adjust based on your CPU
print(f"PyTorch using {torch.get_num_threads()} threads")

# Load model in half precision to save memory and potentially improve speed
model = YOLO("best.pt")
model.model.half()  # Convert to FP16
print("Model loaded in FP16 mode")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Cannot open webcam. Exiting.")
    exit()

print("✅ Connected to webcam. Press 'q' to quit.")

# Performance tracking
frame_times = []
detection_times = []
fps_update_interval = 30  # Update FPS every 30 frames
frame_count = 0
start_time = time.time()

try:
    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to get frame from camera")
            break

        # Resize frame to match model input size (640x640)
        frame = cv2.resize(frame, (640, 640))

        try:
            # Run YOLO detection
            detect_start = time.time()
            results = model(frame, verbose=False)
            detect_time = time.time() - detect_start
            detection_times.append(detect_time)

            # Draw results
            annotated_frame = results[0].plot()

            # Calculate and display FPS
            frame_count += 1
            if frame_count % fps_update_interval == 0:
                end_time = time.time()
                fps = fps_update_interval / (end_time - start_time)
                avg_detect_time = (
                    sum(detection_times[-fps_update_interval:]) / fps_update_interval
                )

                # Display performance metrics
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated_frame,
                    f"Detect: {avg_detect_time*1000:.0f}ms",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Reset timing
                start_time = time.time()
                detection_times = []

            # Show results
            cv2.imshow("Card Detector (YOLO - FP16)", annotated_frame)

        except Exception as e:
            print(f"❌ Error during detection: {str(e)}")
            cv2.imshow("Camera Feed (Detection Failed)", frame)

        # Calculate frame time
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)

        # Break loop if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nExiting...")

            # Print final statistics
            avg_frame_time = sum(frame_times) / len(frame_times)
            print(f"\nPerformance Statistics:")
            print(f"Average frame time: {avg_frame_time*1000:.1f}ms")
            print(f"Average FPS: {1/avg_frame_time:.1f}")
            if detection_times:
                avg_detect = sum(detection_times) / len(detection_times)
                print(f"Average detection time: {avg_detect*1000:.1f}ms")
            break

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
