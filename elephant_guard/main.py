import signal
import time

import cv2
from ultralytics import YOLO

from . import config
from .camera import CameraStream
from .sensors import PIRSensor
from .buzzer import Buzzer
from .gsm import GSMModem
from .stream import MjpegServer
from .ui import create_idle_frame, draw_console, draw_on_frame, build_sms_text


def setup_led():
    try:
        with open(f"{config.LED_PATH}/trigger", "w") as f:
            f.write("none")
        print("  LED Control     : Active")
    except PermissionError:
        print("  ERROR: Run with 'sudo'!")
        raise SystemExit(1)
    except Exception as e:
        print(f"  LED Error       : {e}")


def set_led(state):
    try:
        with open(f"{config.LED_PATH}/brightness", "w") as f:
            f.write("1" if state else "0")
    except Exception:
        pass


def restore_led():
    try:
        with open(f"{config.LED_PATH}/trigger", "w") as f:
            f.write("mmc0")
        print("  LED restored to default.")
    except Exception:
        pass


def main():
    print("=" * 60)
    print("   🐘  ELEPHANT DETECTION  |  PIR + GSM + Buzzer")
    print("=" * 60)

    setup_led()

    print("  Loading YOLO model...")
    model = YOLO(config.MODEL_PATH)
    print("  Model loaded.")

    idle_frame = create_idle_frame()
    stream = MjpegServer("0.0.0.0", config.STREAM_PORT, idle_frame)
    stream.start()
    print(f"  Stream at        : http://pielephant.local:{config.STREAM_PORT}")

    print("  Starting PIR sensor...")
    pir = PIRSensor()
    print(f"  PIR sensor warming up ({config.PIR_WARMUP}s)...")
    time.sleep(config.PIR_WARMUP)
    print("  PIR sensor ready.")

    buzzer = Buzzer()
    gsm = GSMModem(config.GSM_PORT, config.GSM_BAUD, config.SMS_COOLDOWN,
                   sim_pin=config.SIM_PIN) if config.ENABLE_GSM else None

    print("=" * 60)
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
    last_sms_time = 0.0

    try:
        while running:
            now = time.time()
            last_motion_time = pir.last_motion_time
            last_motion_ago = now - last_motion_time if last_motion_time > 0 else 9999
            last_sms_ago = now - last_sms_time if last_sms_time > 0 else 9999

            motion_active = last_motion_ago < config.PIR_COOLDOWN

            if not motion_active:
                if cam is not None:
                    cam.release()
                    cam = None
                    frame_count = 0
                    detections = []
                    elephant_detected = False
                    set_led(False)
                    stream.set_idle(idle_frame.copy())
                    print("\033[H\033[J", end="")
                    print("  Camera stopped — entering STANDBY\n")

                draw_console("standby", [], 0, 0, last_motion_ago, last_sms_ago)
                time.sleep(0.5)
                continue

            if cam is None:
                print("\033[H\033[J", end="")
                print("  🚨 Motion detected! Starting camera...\n")
                cam = CameraStream()
                time.sleep(2)
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

            if frame_count % config.INFER_EVERY == 0:
                small = cv2.resize(frame, (config.YOLO_SIZE, config.YOLO_SIZE))
                results = model(small, conf=config.CONFIDENCE, verbose=False)

                detections = []
                elephant_detected = False

                sx = config.CAMERA_WIDTH / config.YOLO_SIZE
                sy = config.CAMERA_HEIGHT / config.YOLO_SIZE

                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        elephant_detected = True
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            detections.append({
                                "confidence": conf,
                                "bbox": (int(x1 * sx), int(y1 * sy),
                                         int(x2 * sx), int(y2 * sy))
                            })

            set_led(elephant_detected)

            if elephant_detected:
                now_t = time.time()
                if now_t - last_buzz_time >= config.BUZZER_INTERVAL:
                    buzzer.buzz()
                    last_buzz_time = now_t

                if gsm is not None and gsm.send_sms(build_sms_text(config.SMS_TEXT),
                                                    config.SMS_RECIPIENT):
                    last_sms_time = now_t

            if frame_count % config.INFER_EVERY == 0:
                draw_console("active", detections, frame_count, fps, last_motion_ago, last_sms_ago)

            display_frame = draw_on_frame(frame, detections, fps)
            display_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            stream.set_frame(display_bgr)

            time.sleep(0.02)

    finally:
        if cam is not None:
            cam.release()
        pir.stop()
        buzzer.cleanup()
        if gsm is not None:
            gsm.cleanup()
        set_led(False)
        restore_led()
        print("\n  Shutting down. Camera, PIR, GSM, Buzzer, and LED released.")


if __name__ == "__main__":
    main()
