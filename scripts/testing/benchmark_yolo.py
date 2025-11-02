from ultralytics import YOLO
import cv2
import time
import torch


def print_device_info():
    print("\nDevice Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(
        f"Current device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )
    if hasattr(torch, "directml"):
        print("DirectML support: Available")
    else:
        print("DirectML support: Not available")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")


def benchmark_yolo(model_path="best.pt", num_frames=100, imgsz=640):
    print("\nLoading YOLO model...")
    model = YOLO(model_path)
    print(f"Model device: {next(model.parameters()).device}")

    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    # Warm up
    print("\nWarming up model...")
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (imgsz, imgsz))
        _ = model(frame, verbose=False)

    # Benchmark
    print(f"\nRunning benchmark ({num_frames} frames)...")
    times = []
    fps_list = []
    t0 = time.time()

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (imgsz, imgsz))

        t1 = time.time()
        results = model(frame, verbose=False)
        inference_time = (time.time() - t1) * 1000  # ms
        times.append(inference_time)

        # Calculate running FPS
        elapsed = time.time() - t0
        fps = (i + 1) / elapsed
        fps_list.append(fps)

        # Draw results
        annotated_frame = results[0].plot()

        # Show stats
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
            f"Inference: {inference_time:.1f}ms",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO Benchmark", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print statistics
    avg_time = sum(times) / len(times)
    avg_fps = sum(fps_list) / len(fps_list)
    print("\nBenchmark Results:")
    print(f"Average inference time: {avg_time:.1f}ms")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Min inference time: {min(times):.1f}ms")
    print(f"Max inference time: {max(times):.1f}ms")


if __name__ == "__main__":
    print_device_info()
    benchmark_yolo()
