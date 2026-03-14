import sys
import signal
import subprocess
import numpy as np
from ultralytics import YOLO
import time
import cv2
from flask import Flask, Response
from threading import Thread, Lock

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"
LED_PATH = "/sys/class/leds/ACT"
CONFIDENCE = 0.5
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_SIZE = 256       # Smaller = faster inference, less power
INFER_EVERY = 10      # Run YOLO every Nth frame (battery saver)
CAMERA_FPS = 15       # Lower camera FPS = less CPU + power

STREAM_PORT = 5000

# --- FLASK MJPEG STREAM ---
app = Flask(__name__)
latest_frame = None
frame_lock = Lock()


def generate_mjpeg():
    """Yield JPEG frames for the MJPEG stream."""
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            _, jpeg = cv2.imencode('.jpg', latest_frame, [
                                   cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)  # ~30 FPS cap


@app.route('/')
def index():
    return (
        '<html><head><title>Elephant Detection</title></head>'
        '<body style="margin:0;background:#111;display:flex;'
        'justify-content:center;align-items:center;height:100vh">'
        '<img src="/stream" style="max-width:100%;max-height:100vh"/>'
        '</body></html>'
    )


@app.route('/stream')
def stream():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def start_stream_server():
    app.run(host='0.0.0.0', port=STREAM_PORT, threaded=True)

# --- LED CONTROL ---


def setup_led():
    try:
        with open(f"{LED_PATH}/trigger", "w") as f:
            f.write("none")
        print("✅ LED Control: Active")
    except PermissionError:
        print("❌ ERROR: Run with 'sudo'!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ LED Error: {e}")


def set_led(state):
    try:
        with open(f"{LED_PATH}/brightness", "w") as f:
            f.write("1" if state else "0")
    except:
        pass


def restore_led():
    try:
        with open(f"{LED_PATH}/trigger", "w") as f:
            f.write("mmc0")
        print("🔄 LED restored to default.")
    except:
        pass

# --- CONSOLE DISPLAY ---


def draw_console(detections, frame_count, fps):
    print("\033[H\033[J", end="")
    print("=" * 50)
    print("   🐘  ELEPHANT DETECTION  |  Raspberry Pi 5")
    print("=" * 50)
    print(f"  Frame : {frame_count}       FPS: {fps:.1f}")
    print("-" * 50)

    if detections:
        print(f"  🚨 STATUS : ELEPHANT DETECTED  ({len(detections)} object(s))")
        print()
        for i, det in enumerate(detections, 1):
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w // 2
            cy = y1 + h // 2
            print(f"  [{i}] Confidence : {conf:.1%}")
            print(f"      Box        : ({x1}, {y1}) → ({x2}, {y2})")
            print(f"      Size       : {w}px × {h}px")
            print(f"      Center     : ({cx}, {cy})")
            print()
    else:
        print("  ✅ STATUS : Clear — No elephant detected")
        print()

    print("=" * 50)
    print("  Ctrl+C to quit  |  Stream: http://pielephant.local:5000")
    print("=" * 50)

# --- DRAW ON FRAME ---


def draw_on_frame(frame, detections, fps):
    """Draw bounding boxes, labels and FPS on the OpenCV frame."""
    display = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        # Red box when elephant detected
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Label background
        label = f"Elephant {conf:.1%}"
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display, (x1, y1 - lh - baseline - 5),
                      (x1 + lw, y1), (0, 0, 255), -1)

        # Label text
        cv2.putText(display, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # FPS counter top-left
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Status banner at bottom
    if detections:
        banner_color = (0, 0, 200)
        banner_text = f"ELEPHANT DETECTED ({len(detections)})"
    else:
        banner_color = (0, 180, 0)
        banner_text = "Clear - No Elephant"

    cv2.rectangle(display, (0, CAMERA_HEIGHT - 40),
                  (CAMERA_WIDTH, CAMERA_HEIGHT), banner_color, -1)
    cv2.putText(display, banner_text, (10, CAMERA_HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return display

# --- CAMERA via rpicam-vid pipe (threaded) ---


class CameraStream:
    """Threaded camera reader — always has the latest frame ready."""

    def __init__(self):
        cmd = [
            "rpicam-vid",
            "--width",     str(CAMERA_WIDTH),
            "--height",    str(CAMERA_HEIGHT),
            "--framerate", str(CAMERA_FPS),
            "--codec",     "yuv420",
            "--output",    "-",
            "--timeout",   "0",
            "--nopreview"
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.frame = None
        self.stopped = False
        self._lock = Lock()
        self._thread = Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        frame_size = int(CAMERA_WIDTH * CAMERA_HEIGHT * 1.5)
        while not self.stopped:
            raw = self.process.stdout.read(frame_size)
            if len(raw) < frame_size:
                err = self.process.stderr.read().decode(errors='ignore').strip()
                if err:
                    print(f"⚠️  rpicam-vid error:\n{err}")
                self.stopped = True
                break
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                (CAMERA_HEIGHT * 3 // 2, CAMERA_WIDTH)
            )
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            with self._lock:
                self.frame = rgb

    def read(self):
        with self._lock:
            return self.frame

    def release(self):
        self.stopped = True
        self.process.terminate()
        self.process.wait()

# --- MAIN ---


def main():
    setup_led()

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded.")

    print("Starting camera via rpicam-vid...")
    cam = CameraStream()
    time.sleep(2)
    print("✅ Camera streaming.\n")

    # Start MJPEG web stream in background
    stream_thread = Thread(target=start_stream_server, daemon=True)
    stream_thread.start()
    print(f"🌐 Live stream at: http://pielephant.local:{STREAM_PORT}")

    running = True

    def handle_exit(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_exit)

    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    fps_counter = 0
    detections = []
    elephant_detected = False

    try:
        while running:
            if cam.stopped:
                print("❌ Camera stream ended.")
                break

            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            fps_counter += 1

            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()

            # YOLO inference — only every Nth frame
            if frame_count % INFER_EVERY == 0:
                small = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
                results = model(small, conf=CONFIDENCE, verbose=False)

                detections = []
                elephant_detected = False

                # Scale boxes back to original resolution
                sx = CAMERA_WIDTH / YOLO_SIZE
                sy = CAMERA_HEIGHT / YOLO_SIZE

                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        elephant_detected = True
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            detections.append({
                                "confidence": conf,
                                "bbox": (int(x1*sx), int(y1*sy),
                                         int(x2*sx), int(y2*sy))
                            })

            # LED
            set_led(elephant_detected)

            # Console (only on inference frames to reduce spam)
            if frame_count % INFER_EVERY == 0:
                draw_console(detections, frame_count, fps)

            # Update frame for MJPEG web stream
            display_frame = draw_on_frame(frame, detections, fps)
            display_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            with frame_lock:
                global latest_frame
                latest_frame = display_bgr

            # Throttle to prevent CPU overheating / undervoltage shutdown
            time.sleep(0.02)

    finally:
        cam.release()
        set_led(False)
        restore_led()
        print("\n👋 Shutting down. Camera and LED released.")


if __name__ == "__main__":
    main()
