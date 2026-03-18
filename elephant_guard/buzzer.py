import time
import math
from threading import Thread, Lock

import lgpio

from . import config


class Buzzer: # TODO: Change to USB sound card and play actual sound file instead of PWM sweep
    """Drives a passive buzzer on GPIO18 using lgpio tx_pwm."""

    def __init__(self):
        self._h = None
        self._available = False
        try:
            self._h = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self._h, config.BUZZER_PIN)
            self._available = True
        except Exception as e:
            print(f"  Buzzer warning  : unavailable ({e})")
            if self._h is not None:
                try:
                    lgpio.gpiochip_close(self._h)
                except Exception:
                    pass
                self._h = None
        self._busy = False
        self._lock = Lock()
        if self._available:
            print("  Buzzer          : Ready (GPIO18 via lgpio)")

    def _bee_buzz(self):
        if not self._available:
            with self._lock:
                self._busy = False
            return
        start = time.time()
        duration = config.BUZZER_BEE_DURATION
        duty = max(0, min(100, int(config.BUZZER_DUTY)))
        min_f = config.BUZZER_MIN_FREQ
        max_f = config.BUZZER_MAX_FREQ

        try:
            while time.time() - start < duration:
                t = time.time() - start
                wave = (math.sin(t * 2.0 * math.pi * 6.0) + 1.0) / 2.0
                freq = min_f + (max_f - min_f) * wave
                lgpio.tx_pwm(self._h, config.BUZZER_PIN, int(freq), duty)
                time.sleep(config.BUZZER_STEP_SEC)
        finally:
            lgpio.tx_pwm(self._h, config.BUZZER_PIN, 0, 0)
            with self._lock:
                self._busy = False

    def buzz(self):
        """Run a low-volume bee-buzz pattern in a background thread."""
        if not self._available:
            return
        with self._lock:
            if self._busy:
                return
            self._busy = True
        Thread(target=self._bee_buzz, daemon=True).start()

    def cleanup(self):
        if not self._available or self._h is None:
            return
        lgpio.tx_pwm(self._h, config.BUZZER_PIN, 0, 0)
        try:
            lgpio.gpiochip_close(self._h)
        except Exception:
            pass
