import queue
import re
import time
from threading import Thread

try:
    import serial
except Exception:  # pragma: no cover - optional on dev machines
    serial = None

from . import config


class GSMModem:
    """Non-blocking SMS sender with cooldown and background worker."""

    def __init__(self, port, baud, cooldown, sim_pin=None):
        self._available = False
        self._ser = None
        self._cooldown = cooldown
        self._last_sent = 0.0
        self._queue = queue.Queue()
        self._stop = False
        self._worker = Thread(target=self._run, daemon=True)
        self._last_init = 0.0

        if serial is None:
            print("  GSM warning     : pyserial not available")
            return

        try:
            self._ser = serial.Serial(port, baudrate=baud, timeout=1)
            self._available = True
        except Exception as e:
            print(f"  GSM warning     : unavailable ({e})")
            return

        if self._setup(sim_pin):
            print(f"  GSM modem       : Ready ({port} @ {baud})")
        else:
            print("  GSM warning     : setup failed; disabling")
            self._available = False
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
            return

        self._worker.start()

    def _read_until(self, tokens, timeout=5.0):
        end_time = time.time() + timeout
        buffer = ""
        while time.time() < end_time:
            try:
                data = self._ser.read(self._ser.in_waiting or 1)
            except Exception:
                break
            if data:
                buffer += data.decode(errors="ignore")
                for token in tokens:
                    if token in buffer:
                        return token, buffer
            time.sleep(0.05)
        return None, buffer

    def _send_at(self, cmd, expect=("OK",), timeout=5.0):
        if not self._available:
            return False, ""
        try:
            self._ser.reset_input_buffer()
            self._ser.write((cmd + "\r").encode())
            token, output = self._read_until(expect, timeout=timeout)
            return token in expect, output
        except Exception:
            return False, ""

    def _setup(self, sim_pin):
        ok, _ = self._send_at("AT")
        if not ok:
            return False
        self._send_at("ATE0")
        self._send_at("AT+CMEE=2")
        if sim_pin:
            self._send_at(f'AT+CPIN="{sim_pin}"', timeout=10.0)
        ok, out = self._send_at("AT+CPIN?", expect=("OK",))
        if out.strip():
            print(f"  GSM SIM         : {out.strip()}")
        ok, _ = self._send_at("AT+CMGF=1")
        if not ok:
            return False
        ok, _ = self._send_at('AT+CSCS="GSM"')
        if not ok:
            return False
        ok, out = self._send_at("AT+CSQ", expect=("OK",))
        if out.strip():
            print(f"  GSM signal      : {out.strip()}")
        if config.GSM_WAIT_FOR_NETWORK:
            self._wait_for_network(config.GSM_NETWORK_TIMEOUT)
        self._last_init = time.time()
        return True

    def _wait_for_network(self, timeout):
        start = time.time()
        while time.time() - start < timeout:
            ok, out = self._send_at("AT+CREG?", expect=("OK",), timeout=2.0)
            if ok and self._is_registered(out):
                return True
            time.sleep(2.0)
        print("  GSM warning     : network registration timeout")
        return False

    @staticmethod
    def _is_registered(output):
        match = re.search(r"\+CREG:\s*\d,(\d)", output)
        if not match:
            return False
        status = match.group(1)
        return status in ("1", "5")

    def _send_sms(self, text, to_number):
        if not self._available:
            return False
        ok, output = self._send_at(f'AT+CMGS="{to_number}"', expect=(">",), timeout=5.0)
        if not ok:
            if output:
                print(f"  GSM error       : no prompt ({output.strip()})")
            return False
        try:
            self._ser.write(text.encode("utf-8") + b"\x1A")
            token, output = self._read_until(("OK", "ERROR"), timeout=30.0)
            if token == "OK":
                print(f"  GSM SMS sent    : {to_number}")
                return True
            if "ERROR" in output:
                print(f"  GSM error       : {output.strip()}")
            return False
        except Exception:
            return False

    def _run(self):
        while not self._stop:
            try:
                text, to_number = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            sent = self._send_sms(text, to_number)
            if not sent and time.time() - self._last_init > config.GSM_RETRY_SECONDS:
                print("  GSM warning     : send failed, reinitializing")
                self._setup(config.SIM_PIN)

    def send_sms(self, text, to_number):
        if not self._available:
            return False
        now = time.time()
        if now - self._last_sent < self._cooldown:
            return False
        if to_number.strip() == "+10000000000":
            print("  GSM warning     : set SMS_RECIPIENT to a real number")
            return False
        self._last_sent = now
        self._queue.put((text, to_number))
        return True

    def cleanup(self):
        self._stop = True
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
