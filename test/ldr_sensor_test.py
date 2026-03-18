import time
from threading import Lock, Thread
import gpiod

# GPIO configuration
GPIO_CHIP = "/dev/gpiochip4"
LDR_LINE = 17 

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


def test_ldr_sensor():
    """Test LDR sensor status."""
    sensor = LDRSensor()
    
    try:
        print("LDR Sensor Test Started...")
        for i in range(10):
            status = "DARK" if sensor.is_dark else "BRIGHT"
            print(f"Reading {i+1}: {status}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        sensor.stop()
        print("LDR Sensor Test Completed")


if __name__ == "__main__":
    test_ldr_sensor()