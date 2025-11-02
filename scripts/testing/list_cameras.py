import cv2


def list_cameras():
    """List all available cameras and their resolutions"""
    # Try first 5 camera indices
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Get resolution
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            # Try to get one frame
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i}: {int(width)}x{int(height)} - OK")
            else:
                print(f"Camera {i}: Could not read frame")

            cap.release()
        else:
            print(f"Camera {i}: Not available")


if __name__ == "__main__":
    print("Available cameras:")
    print("-----------------")
    list_cameras()
    print("\nTo use specific camera in process.py/process_yolo.py:")
    print("Change: cv2.VideoCapture(url) to cv2.VideoCapture(index, cv2.CAP_DSHOW)")
    print("Where index is the number of the DroidCam camera from the list above")
