"""
Flask dashboard — MJPEG stream at /stream, JSON status at /status, HTML at /.
All state is written by the main loop via update_state() and read_frame().
"""

import json
import time
import subprocess
from pathlib import Path
from threading import Thread, Lock

import cv2
import numpy as np
import psutil
from flask import Flask, Response

from config import CAMERA_WIDTH, CAMERA_HEIGHT, STREAM_PORT

# ── Shared state ───────────────────────────────────────────────────────────

_state: dict = {
    "mode": "standby",
    "elephant_detected": False,
    "detections": [],
    "frame_count": 0,
    "fps": 0.0,
    "last_motion_ago": 9999,
    "buzzer_active": False,
}
_state_lock = Lock()

_latest_frame = None
_frame_lock = Lock()

_idle_img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
cv2.putText(_idle_img, "STANDBY - Waiting for motion...",
            (40, CAMERA_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 200), 2)


def update_state(**kwargs):
    with _state_lock:
        _state.update(kwargs)


def set_frame(frame):
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame


# ── System metrics ─────────────────────────────────────────────────────────

def _system_metrics() -> dict:
    cpu_temp = 0.0
    try:
        cpu_temp = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except Exception:
        pass

    throttle = "OK"
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=2)
        val = int(out.split("=")[1], 16)
        if val:
            flags = {0x1: "UNDER-VOLT", 0x2: "FREQ-CAP", 0x4: "THROTTLED", 0x8: "TEMP-LIMIT"}
            throttle = "+".join(v for k, v in flags.items() if val & k) or "WARN"
    except Exception:
        throttle = "N/A"

    vm = psutil.virtual_memory()
    du = psutil.disk_usage("/")
    secs = int(time.time() - psutil.boot_time())
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)

    return {
        "cpu_temp":    round(cpu_temp, 1),
        "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
        "ram_used":    round(vm.used / 1024 / 1024),
        "ram_total":   round(vm.total / 1024 / 1024),
        "disk_percent": round(du.percent, 1),
        "uptime":      f"{h}h {m:02d}m {s:02d}s",
        "throttle":    throttle,
    }


# ── Flask routes ───────────────────────────────────────────────────────────

app = Flask(__name__)

DASHBOARD_HTML = open(Path(__file__).parent / "dashboard.html").read()


@app.route("/")
def index():
    return DASHBOARD_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/stream")
def stream():
    def _gen():
        while True:
            with _frame_lock:
                frame = _latest_frame if _latest_frame is not None else _idle_img
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            time.sleep(0.033)
    return Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with _state_lock:
        payload = dict(_state)
        payload["detections"] = list(_state["detections"])
    payload["system"] = _system_metrics()
    return Response(json.dumps(payload), mimetype="application/json")


def start(port: int = STREAM_PORT):
    Thread(target=lambda: app.run(host="0.0.0.0", port=port, threaded=True), daemon=True).start()
    print(f"  Dashboard       : http://pielephant.local:{port}")
