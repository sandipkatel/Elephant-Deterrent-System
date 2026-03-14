import time
from threading import Thread, Lock

import gpiod

from . import config


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
            with gpiod.request_lines(config.GPIO_CHIP, consumer="pir_cam", config={
                config.PIR_LINE: gpiod.LineSettings(
                    direction=gpiod.line.Direction.INPUT,
                    bias=gpiod.line.Bias.PULL_DOWN
                )
            }) as req:
                while not self._stopped:
                    values = req.get_values([config.PIR_LINE])
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
