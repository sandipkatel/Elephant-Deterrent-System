import time

import cv2
import numpy as np

from . import config


def create_idle_frame():
    idle_img = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
    cv2.putText(idle_img, "STANDBY - Waiting for motion...",
                (40, config.CAMERA_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
    cv2.putText(idle_img, "PIR sensor active",
                (180, config.CAMERA_HEIGHT // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    return idle_img


def draw_console(state, detections, frame_count, fps, last_motion_ago, last_sms_ago):
    print("\033[H\033[J", end="")
    print("=" * 60)
    print("   🐘  ELEPHANT DETECTION  |  PIR + GSM + Buzzer")
    print("=" * 60)

    if state == "standby":
        print("  MODE   : 💤 STANDBY  (camera off, PIR watching)")
        print(f"  MOTION : last {last_motion_ago:.0f}s ago")
        print("-" * 60)
        print("  ✅ STATUS : Idle — Waiting for motion")
    else:
        print("  MODE   : 📷 ACTIVE   (camera + YOLO running)")
        print(f"  MOTION : last {last_motion_ago:.0f}s ago")
        print(f"  Frame  : {frame_count:<10}  FPS: {fps:.1f}")
        print(f"  Cooldown: camera off in {max(0, config.PIR_COOLDOWN - last_motion_ago):.0f}s")
        print(f"  SMS    : last {last_sms_ago:.0f}s ago")
        print("-" * 60)

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
    print("=" * 60)
    print("  Ctrl+C to quit  |  Stream: http://pielephant.local:5000")
    print("=" * 60)


def draw_on_frame(frame, detections, fps):
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

    cv2.rectangle(display, (0, config.CAMERA_HEIGHT - 40),
                  (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), banner_color, -1)
    cv2.putText(display, banner_text, (10, config.CAMERA_HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return display


def build_sms_text(base_text):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"{base_text} ({timestamp})"
