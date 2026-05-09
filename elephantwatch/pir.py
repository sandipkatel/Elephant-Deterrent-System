import time
from threading import Thread, Lock
import gpiod
from config import GPIO_CHIP, PIR_LINE


class PIRSensor:
    """Reads the PIR sensor in a background thread."""

    def __init__(self):
        self._motion = False
        self._last_motion_time = 0.0
        self._stopped = False
        self._lock = Lock()
        Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            with gpiod.request_lines(
                GPIO_CHIP,
                consumer="pir_cam",
                config={
                    PIR_LINE: gpiod.LineSettings(
                        direction=gpiod.line.Direction.INPUT,
                        bias=gpiod.line.Bias.PULL_DOWN,
                    )
                },
            ) as req:
                while not self._stopped:
                    motion = bool(req.get_values([PIR_LINE])[0].value)
                    with self._lock:
                        self._motion = motion
                        if motion:
                            self._last_motion_time = time.time()
                    time.sleep(0.1)
        except Exception as e:
            print(f"  PIR error: {e}")

    @property
    def motion_detected(self) -> bool:
        with self._lock:
            return self._motion

    @property
    def last_motion_time(self) -> float:
        with self._lock:
            return self._last_motion_time

    def stop(self):
        self._stopped = True
