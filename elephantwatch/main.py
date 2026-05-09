"""
ElephantWatch — main entry point.

Run with:  sudo python main.py
"""

import signal
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

import led
import server
from buzzer import Buzzer
from camera import CameraStream
from config import (
    BUZZER_INTERVAL, CAMERA_HEIGHT, CAMERA_WIDTH,
    CONFIDENCE, INFER_EVERY, MODEL_PATH,
    PIR_COOLDOWN, PIR_WARMUP, TARGET_CLASS, YOLO_SIZE,
)
from pir import PIRSensor
from sms import SMSAlerter


# ── Drawing helpers ────────────────────────────────────────────────────────

def _annotate(frame, detections):
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"Elephant {det['confidence']:.1%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - lh - bl - 5), (x1 + lw, y1), (0, 0, 255), -1)
        cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def _console(mode, detections, frame_count, fps, last_motion_ago):
    print("\033[H\033[J", end="")
    print("=" * 55)
    print("   🐘  ELEPHANT DETECTION  |  PIR + Camera")
    print("=" * 55)
    if mode == "standby":
        print(f"  MODE   : 💤 STANDBY  (last motion {last_motion_ago:.0f}s ago)")
        print("  STATUS : ✅ Idle — Waiting for motion")
    else:
        print(f"  MODE   : 📷 ACTIVE  | Frame {frame_count} | FPS {fps:.1f}")
        print(f"  Cooldown: off in {max(0, PIR_COOLDOWN - last_motion_ago):.0f}s")
        if detections:
            print(f"  STATUS : 🚨 ELEPHANT DETECTED ({len(detections)} object(s))")
            for i, d in enumerate(detections, 1):
                x1, y1, x2, y2 = d["bbox"]
                print(f"  [{i}] conf={d['confidence']:.1%}  box=({x1},{y1})→({x2},{y2})")
        else:
            print("  STATUS : ✅ Clear — No elephant")
    print("=" * 55)


# ── YOLO inference ─────────────────────────────────────────────────────────

def _infer(model, frame):
    small = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
    results = model(small, conf=CONFIDENCE, verbose=False)
    sx, sy = CAMERA_WIDTH / YOLO_SIZE, CAMERA_HEIGHT / YOLO_SIZE
    detections = []
    for r in results:
        if not r.boxes:
            continue
        for box in r.boxes:
            cls = int(box.cls[0]) if box.cls is not None else -1
            if r.names.get(cls) != TARGET_CLASS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "confidence": float(box.conf[0]),
                "bbox": [int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)],
            })
    return detections


# ── Main loop ──────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   🐘  ElephantWatch  |  Booting…")
    print("=" * 55)

    led.setup()

    print("  Loading YOLO model…")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print("  Model loaded.")

    server.start()

    sms = SMSAlerter()

    print(f"  Starting PIR (warm-up {PIR_WARMUP}s)…")
    pir = PIRSensor()
    time.sleep(PIR_WARMUP)
    print("  PIR ready.")

    buzzer = Buzzer()

    running = True

    def _quit(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _quit)
    signal.signal(signal.SIGTERM, _quit)

    cam = None
    frame_count = 0
    fps = fps_counter = 0.0
    fps_timer = time.time()
    detections = []
    elephant_detected = False
    last_buzz_time = 0.0

    try:
        while running:
            now = time.time()
            last_motion_ago = now - pir.last_motion_time if pir.last_motion_time else 9999
            motion_active = last_motion_ago < PIR_COOLDOWN

            # ── STANDBY ────────────────────────────────────────────────────
            if not motion_active:
                if cam:
                    cam.release()
                    cam = None
                    frame_count = detections = []
                    frame_count = 0
                    detections = []
                    elephant_detected = False
                    led.set(False)
                    server.set_frame(None)

                server.update_state(
                    mode="standby", elephant_detected=False, detections=[],
                    frame_count=0, fps=0.0,
                    last_motion_ago=round(last_motion_ago, 1), buzzer_active=False,
                )
                _console("standby", [], 0, 0, last_motion_ago)
                time.sleep(0.5)
                continue

            # ── ACTIVE ─────────────────────────────────────────────────────
            if cam is None:
                print("\n  🚨 Motion! Starting camera…")
                cam = CameraStream()
                time.sleep(2)
                fps_timer = time.time()
                fps_counter = 0

            if cam.stopped:
                print("  Camera ended unexpectedly.")
                cam = None
                time.sleep(1)
                continue

            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            fps_counter += 1
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = now

            # Inference every Nth frame
            if frame_count % INFER_EVERY == 0:
                detections = _infer(model, frame)
                elephant_detected = bool(detections)
                _console("active", detections, frame_count, fps, last_motion_ago)

            led.set(elephant_detected)

            # Buzzer
            if elephant_detected and now - last_buzz_time >= BUZZER_INTERVAL:
                buzzer.beep()
                last_buzz_time = now

            # SMS (30-min cooldown enforced inside SMSAlerter)
            if elephant_detected:
                sms.alert()

            buzz_active = elephant_detected and (now - last_buzz_time < 1.0)
            server.update_state(
                mode="active", elephant_detected=elephant_detected,
                detections=[{"confidence": d["confidence"], "bbox": d["bbox"]} for d in detections],
                frame_count=frame_count, fps=round(fps, 1),
                last_motion_ago=round(last_motion_ago, 1), buzzer_active=buzz_active,
            )

            # Push annotated frame to MJPEG stream
            display = _annotate(frame, detections)
            server.set_frame(cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

            time.sleep(0.02)

    finally:
        if cam:
            cam.release()
        pir.stop()
        buzzer.cleanup()
        sms.cleanup()
        led.set(False)
        led.restore()
        print("\n  Shutdown complete.")


if __name__ == "__main__":
    main()
