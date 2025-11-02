import cv2
import time
import socket
from urllib.parse import urlparse


def check_port_open(host, port, timeout=2):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        sock.close()
        return False


def test_droidcam(url, timeout=5):
    print(f"\nTesting connection to {url}")

    # Parse URL to check port
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or 4747

    # Check if port is open first
    if host != "127.0.0.1":  # Skip localhost check
        print(f"Checking if port {port} is open on {host}...")
        if not check_port_open(host, port):
            print(f"❌ Port {port} is not open on {host}")
            print("   - Is DroidCam running on your phone?")
            print("   - Are phone and PC on the same network?")
            print("   - Check firewall settings")
            return False
        else:
            print(f"✅ Port {port} is open on {host}")

    # Try to connect and get video
    print("Attempting to open video stream...")
    cap = cv2.VideoCapture(url)
    start = time.time()

    while time.time() - start < timeout:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✅ Success! Got video frame: {w}x{h}")
                print("Showing test frame for 2 seconds...")

                # Show frame with connection info
                text = f"Connected to {url}"
                cv2.putText(
                    frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                cv2.imshow("DroidCam Test", frame)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                cap.release()
                return True
        time.sleep(0.5)

    print("❌ Could not get video stream")
    print("   - Check if URL is correct")
    print("   - Make sure DroidCam video is started")
    cap.release()
    return False


if __name__ == "__main__":
    print("DroidCam Connection Tester")
    print("-------------------------")
    print(
        "This script will try to connect to DroidCam using both WiFi and USB methods."
    )
    print("Make sure:")
    print("1. DroidCam app is running on your phone")
    print("2. You can see an IP address in the app (for WiFi)")
    print("3. USB debugging is enabled (for USB)")
    print("\nPress Enter to start testing, Ctrl+C to quit...")
    input()

    # Try available camera indices with DirectShow
    print("\nTrying available camera indices...")
    success = False

    for idx in [0, 2, 3]:  # indices that were detected
        print(f"\nTrying camera index {idx}")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"✅ Success! Connected to camera {idx} ({w}x{h})")
                print("Showing test frame for 2 seconds...")

                text = f"Camera {idx} ({w}x{h})"
                cv2.putText(
                    frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                cv2.imshow("Camera Test", frame)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                cap.release()
                success = True
                break
                cap.release()

        if not success:
            print("\n❌ Could not connect to DroidCam")
            print("\nTroubleshooting steps:")
            print("1. Make sure DroidCam app is running and showing camera")
            print("2. For WiFi:")
            print("   - Check the IP shown in DroidCam app")
            print("   - Make sure phone and PC are on same network")
            print("   - Try disabling firewall temporarily")
            print("3. For USB:")
            print("   - Enable USB debugging in phone settings")
            print("   - Install/run DroidCam Windows client")
            print("   - Try different USB port/cable")
            print("\nTo try a specific IP, edit wifi_url in this script.")

    print("\nPress Enter to exit...")
