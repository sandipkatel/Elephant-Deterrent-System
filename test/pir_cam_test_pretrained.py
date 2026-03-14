import sys
import signal
import subprocess
import numpy as np
from ultralytics import YOLO
import time
import cv2
import gpiod
import lgpio
from flask import Flask, Response
from threading import Thread, Lock

# --- CONFIGURATION ---
# Uses Ultralytics pretrained model (auto-downloads on first run).
MODEL_PATH = "../yolov8n.pt"
LED_PATH = "/sys/class/leds/ACT"
CONFIDENCE = 0.5
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_SIZE = 256
INFER_EVERY = 10
CAMERA_FPS = 15

# COCO class name used by yolov8n.pt
TARGET_CLASS_NAME = "elephant"

STREAM_PORT = 5000

# PIR Sensor
GPIO_CHIP = "/dev/gpiochip4"
PIR_LINE = 27                  # GPIO pin 27
PIR_COOLDOWN = 5              # Seconds to keep camera on after last motion
PIR_WARMUP = 5                # Seconds to let PIR sensor stabilize

# LDR Sensor (digital output via comparator module DO pin)
LDR_LINE = 17                 # GPIO pin number for LDR DO

# Buzzer (GPIO18 via lgpio PWM)
BUZZER_PIN = 18
BUZZER_DURATION = 0.3         # Kept for API compatibility with caller
BUZZER_INTERVAL = 5.0         # Minimum seconds between beeps when elephant detected

# --- FLASK MJPEG STREAM ---
app = Flask(__name__)
latest_frame = None
frame_lock = Lock()

# Idle placeholder image (shown when camera is off)
idle_img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
cv2.putText(idle_img, "STANDBY - Waiting for motion...", (40, CAMERA_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
cv2.putText(idle_img, "PIR sensor active", (180, CAMERA_HEIGHT // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)


def generate_mjpeg():
    """Yield JPEG frames for the MJPEG stream."""
    while True:
        with frame_lock:
            frame = latest_frame if latest_frame is not None else idle_img
            _, jpeg = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)


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
        print("  LED Control     : Active")
    except PermissionError:
        print("  ERROR: Run with 'sudo'!")
        sys.exit(1)
    except Exception as e:
        print(f"  LED Error       : {e}")


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
        print("  LED restored to default.")
    except:
        pass


# --- CONSOLE DISPLAY ---

def draw_console(state, detections, frame_count, fps, last_motion_ago, ldr_dark=False):
    print("\033[H\033[J", end="")
    print("=" * 55)
    print("   🐘  ELEPHANT DETECTION  |  PIR + LDR + Camera")
    print("=" * 55)

    if state == "standby":
        print("  MODE   : 💤 STANDBY  (camera off, PIR watching)")
        print(f"  MOTION : last {last_motion_ago:.0f}s ago")
        print(f"  LIGHT  : {'🌙 Dark' if ldr_dark else '☀️  Bright'}")
        print("-" * 55)
        print("  ✅ STATUS : Idle — Waiting for motion")
    else:
        print("  MODE   : 📷 ACTIVE   (camera + YOLO running)")
        print(f"  MOTION : last {last_motion_ago:.0f}s ago")
        print(f"  LIGHT  : {'🌙 Dark' if ldr_dark else '☀️  Bright'}")
        print(f"  Frame  : {frame_count:<10}  FPS: {fps:.1f}")
        print(f"  Cooldown: camera off in {max(0, PIR_COOLDOWN - last_motion_ago):.0f}s")
        print("-" * 55)

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
                print(f"      Box        : ({x1}, {y1}) -> ({x2}, {y2})")
                print(f"      Size       : {w}px x {h}px")
                print(f"      Center     : ({cx}, {cy})")
                print()
        else:
            print("  ✅ STATUS : Clear — No elephant detected")

    print()
    print("=" * 55)
    print("  Ctrl+C to quit  |  Stream: http://pielephant.local:5000")
    print("=" * 55)


# --- DRAW ON FRAME ---

def draw_on_frame(frame, detections, fps):
    """Draw bounding boxes, labels and FPS on the OpenCV frame."""
    display = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"Elephant {conf:.1%}"
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display, (x1, y1 - lh - baseline - 5),
                      (x1 + lw, y1), (0, 0, 255), -1)
        cv2.putText(display, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
                    print(f"  rpicam-vid error: {err}")
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


# --- PIR SENSOR (threaded) ---

class PIRSensor:
    """Reads the PIR sensor in a background thread."""

    def __init__(self):
        self._motion = False
        self._last_motion_time = 0.0
        self._stopped = False
        self._lock = Lock()
        self._thread = Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        try:
            with gpiod.request_lines(GPIO_CHIP, consumer="pir_cam", config={
                PIR_LINE: gpiod.LineSettings(
                    direction=gpiod.line.Direction.INPUT,
                    bias=gpiod.line.Bias.PULL_DOWN
                )
            }) as req:
                while not self._stopped:
                    values = req.get_values([PIR_LINE])
                    motion = bool(values[0].value)
                    with self._lock:
                        self._motion = motion
                        if motion:
                            self._last_motion_time = time.time()
                    time.sleep(0.1)
        except Exception as e:
            print(f"  PIR sensor error: {e}")
            self._stopped = True

    @property
    def motion_detected(self):
        with self._lock:
            return self._motion

    @property
    def last_motion_time(self):
        with self._lock:
            return self._last_motion_time

    def stop(self):
        self._stopped = True


# --- LDR SENSOR (threaded) ---

class LDRSensor:
    """Reads LDR digital output (comparator DO pin) in a background thread.
    DO = LOW when dark, HIGH when bright (pull-up comparator module).
    """

    def __init__(self):
        self._is_dark = False
        self._stopped = False
        self._lock = Lock()
        self._thread = Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        try:
            with gpiod.request_lines(GPIO_CHIP, consumer="ldr_sensor", config={
                LDR_LINE: gpiod.LineSettings(
                    direction=gpiod.line.Direction.INPUT,
                    bias=gpiod.line.Bias.PULL_UP
                )
            }) as req:
                while not self._stopped:
                    values = req.get_values([LDR_LINE])
                    # DO = LOW (False) means dark with pull-up comparator module
                    dark = not bool(values[0].value)
                    with self._lock:
                        self._is_dark = dark
                    time.sleep(0.5)
        except Exception as e:
            print(f"  LDR sensor error: {e}")
            self._stopped = True

    @property
    def is_dark(self):
        with self._lock:
            return self._is_dark

    def stop(self):
        self._stopped = True


# --- BUZZER (GPIO18 via lgpio tx_pwm) ---

class Buzzer:
    """Drives a passive buzzer on GPIO18 using lgpio tx_pwm."""

    def __init__(self):
        self._h = None
        self._available = False
        try:
            self._h = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self._h, BUZZER_PIN)
            self._available = True
        except Exception as e:
            print(f"  Buzzer warning  : unavailable ({e})")
            if self._h is not None:
                try:
                    lgpio.gpiochip_close(self._h)
                except Exception:
                    pass
                self._h = None
        self._busy = False
        self._lock = Lock()
        if self._available:
            print("  Buzzer          : Ready (GPIO18 via lgpio)")

    def _run_sweep(self):
        if not self._available:
            with self._lock:
                self._busy = False
            return
        try:
            # Buzz up
            for freq in range(200, 400, 10):
                lgpio.tx_pwm(self._h, BUZZER_PIN, freq, 50)
                time.sleep(0.02)

            # Buzz down
            for freq in range(400, 200, -10):
                lgpio.tx_pwm(self._h, BUZZER_PIN, freq, 50)
                time.sleep(0.02)
        finally:
            lgpio.tx_pwm(self._h, BUZZER_PIN, 0, 0)
            with self._lock:
                self._busy = False

    def beep(self, duration=0.3):
        """Run one up/down buzz pattern in a background thread (non-blocking)."""
        if not self._available:
            return
        with self._lock:
            if self._busy:
                return
            self._busy = True
        Thread(target=self._run_sweep, daemon=True).start()

    def cleanup(self):
        if not self._available or self._h is None:
            return
        lgpio.tx_pwm(self._h, BUZZER_PIN, 0, 0)
        try:
            lgpio.gpiochip_close(self._h)
        except Exception:
            pass


# --- MAIN ---

def main():
    print("=" * 55)
    print("   🐘  ELEPHANT DETECTION  |  PIR + LDR + Camera")
    print("=" * 55)

    setup_led()

    print("  Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("  Model loaded.")

    # Start MJPEG web stream (shows idle screen until camera activates)
    stream_thread = Thread(target=start_stream_server, daemon=True)
    stream_thread.start()
    print(f"  Stream at        : http://pielephant.local:{STREAM_PORT}")

    # Start PIR sensor
    print("  Starting PIR sensor...")
    pir = PIRSensor()
    print(f"  PIR sensor warming up ({PIR_WARMUP}s)...")
    time.sleep(PIR_WARMUP)
    print("  PIR sensor ready.")

    # Start LDR sensor
    print("  Starting LDR sensor...")
    ldr = LDRSensor()
    print("  LDR Sensor      : Active")

    # Setup Buzzer
    buzzer = Buzzer()

    print("=" * 55)
    print("  System in STANDBY — waiting for motion...\n")

    running = True

    def handle_exit(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_exit)

    cam = None
    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    fps_counter = 0
    detections = []
    elephant_detected = False
    last_buzz_time = 0.0

    try:
        while running:
            now = time.time()
            last_motion_time = pir.last_motion_time
            last_motion_ago = now - last_motion_time if last_motion_time > 0 else 9999

            motion_active = last_motion_ago < PIR_COOLDOWN

            # === STANDBY MODE (no motion) ===
            if not motion_active:
                # Shut down camera if it was running
                if cam is not None:
                    cam.release()
                    cam = None
                    frame_count = 0
                    detections = []
                    elephant_detected = False
                    set_led(False)
                    # Show idle frame on stream
                    with frame_lock:
                        global latest_frame
                        latest_frame = idle_img.copy()
                    print("\033[H\033[J", end="")
                    print("  Camera stopped — entering STANDBY\n")

                draw_console("standby", [], 0, 0, last_motion_ago, ldr_dark=ldr.is_dark)
                time.sleep(0.5)
                continue

            # === ACTIVE MODE (motion detected recently) ===
            # Start camera if not running
            if cam is None:
                print("\033[H\033[J", end="")
                print("  🚨 Motion detected! Starting camera...\n")
                cam = CameraStream()
                time.sleep(2)  # Let camera initialize
                fps_timer = time.time()
                fps_counter = 0

            if cam.stopped:
                print("  Camera stream ended unexpectedly.")
                cam = None
                time.sleep(1)
                continue

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

            # YOLO inference every Nth frame
            if frame_count % INFER_EVERY == 0:
                small = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
                results = model(small, conf=CONFIDENCE, verbose=False)

                detections = []
                elephant_detected = False

                sx = CAMERA_WIDTH / YOLO_SIZE
                sy = CAMERA_HEIGHT / YOLO_SIZE

                for result in results:
                    if result.boxes is None or len(result.boxes) == 0:
                        continue

                    # With pretrained COCO model, keep only elephant detections.
                    for box in result.boxes:
                        cls_idx = int(box.cls[0]) if box.cls is not None else -1
                        class_name = result.names.get(cls_idx, "")
                        if class_name != TARGET_CLASS_NAME:
                            continue

                        elephant_detected = True
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        detections.append({
                            "confidence": conf,
                            "bbox": (int(x1 * sx), int(y1 * sy),
                                     int(x2 * sx), int(y2 * sy))
                        })

            # LED
            set_led(elephant_detected)

            # Buzzer: beep once every BUZZER_INTERVAL when elephant detected
            if elephant_detected:
                now_t = time.time()
                if now_t - last_buzz_time >= BUZZER_INTERVAL:
                    buzzer.beep(BUZZER_DURATION)
                    last_buzz_time = now_t

            # Console update on inference frames
            if frame_count % INFER_EVERY == 0:
                draw_console("active", detections, frame_count, fps, last_motion_ago, ldr_dark=ldr.is_dark)

            # Update frame for MJPEG web stream
            display_frame = draw_on_frame(frame, detections, fps)
            display_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            with frame_lock:
                latest_frame = display_bgr

            time.sleep(0.02)

    finally:
        if cam is not None:
            cam.release()
        pir.stop()
        ldr.stop()
        buzzer.cleanup()
        set_led(False)
        restore_led()
        print("\n  Shutting down. Camera, PIR, LDR, Buzzer, and LED released.")


if __name__ == "__main__":
    main()
