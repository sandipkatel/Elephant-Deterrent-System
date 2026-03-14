import subprocess
from threading import Thread, Lock

import numpy as np
import cv2

from . import config


class CameraStream:
    """Threaded camera reader — always has the latest frame ready."""

    def __init__(self):
        cmd = [
            "rpicam-vid",
            "--width",     str(config.CAMERA_WIDTH),
            "--height",    str(config.CAMERA_HEIGHT),
            "--framerate", str(config.CAMERA_FPS),
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
        frame_size = int(config.CAMERA_WIDTH * config.CAMERA_HEIGHT * 1.5)
        while not self.stopped:
            raw = self.process.stdout.read(frame_size)
            if len(raw) < frame_size:
                err = self.process.stderr.read().decode(errors='ignore').strip()
                if err:
                    print(f"  rpicam-vid error: {err}")
                self.stopped = True
                break
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                (config.CAMERA_HEIGHT * 3 // 2, config.CAMERA_WIDTH)
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
