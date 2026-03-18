import time
import gpiod
from threading import Thread, Lock

# GPIO Configuration
GPIO_CHIP = "/dev/gpiochip4"
PIR_LINE = 27 

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


if __name__ == "__main__":
    sensor = PIRSensor()
    print("PIR Motion Sensor Reading Started...")
    print("Waiting for sensor to stabilize...")
    time.sleep(2)
    
    try:
        while True:
            if sensor.motion_detected:
                print("Motion Detected!")
            else:
                print("No Motion")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nProgram stopped")
    finally:
        sensor.stop()
