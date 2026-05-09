import subprocess
import numpy as np
import cv2
from threading import Thread, Lock
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


class CameraStream:
    """Threaded rpicam-vid reader — always has the latest frame ready."""

    def __init__(self):
        cmd = [
            "rpicam-vid",
            "--width",     str(CAMERA_WIDTH),
            "--height",    str(CAMERA_HEIGHT),
            "--framerate", str(CAMERA_FPS),
            "--codec",     "yuv420",
            "--output",    "-",
            "--timeout",   "0",
            "--nopreview",
        ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frame = None
        self.stopped = False
        self._lock = Lock()
        Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        frame_size = int(CAMERA_WIDTH * CAMERA_HEIGHT * 1.5)
        while not self.stopped:
            raw = self.process.stdout.read(frame_size)
            if len(raw) < frame_size:
                err = self.process.stderr.read().decode(errors="ignore").strip()
                if err:
                    print(f"  Camera error: {err}")
                self.stopped = True
                break
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((CAMERA_HEIGHT * 3 // 2, CAMERA_WIDTH))
            with self._lock:
                self.frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)

    def read(self):
        with self._lock:
            return self.frame

    def release(self):
        self.stopped = True
        self.process.terminate()
        self.process.wait()
