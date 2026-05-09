"""
SMS alerter via serial GSM modem.
Sends a burst to all configured numbers, then enforces a cooldown period
so the system isn't spammy during a prolonged detection event.
"""

import re
import time
import serial
from threading import Thread, Lock
from config import SMS_PORT, SMS_BAUD_LIST, SMS_NUMBERS, SMS_MESSAGE, SMS_COOLDOWN


# ── Low-level modem helpers ────────────────────────────────────────────────

def _read_until(ser, tokens, timeout=5.0):
    deadline = time.time() + timeout
    buf = ""
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk.decode(errors="ignore")
            for tok in tokens:
                if tok in buf:
                    return tok, buf
        time.sleep(0.05)
    return None, buf


def _at(ser, cmd, expect=("OK",), timeout=5.0):
    ser.reset_input_buffer()
    ser.write((cmd + "\r").encode())
    tok, out = _read_until(ser, expect, timeout)
    return tok in expect, out


def _wait_network(ser, timeout=30.0):
    start = time.time()
    while time.time() - start < timeout:
        ok, out = _at(ser, "AT+CREG?", timeout=2.0)
        if ok:
            m = re.search(r"\+CREG:\s*\d,(\d)", out)
            if m and m.group(1) in ("1", "5"):
                return True
        time.sleep(2.0)
    return False


def _send_one(ser, number, text):
    ok, out = _at(ser, f'AT+CMGS="{number}"', expect=(">",))
    if not ok:
        print(f"  SMS: CMGS failed for {number}: {out.strip()}")
        return False
    ser.write(text.encode("utf-8") + b"\x1A")
    tok, out = _read_until(ser, ("OK", "ERROR"), timeout=30.0)
    if tok == "OK":
        print(f"  SMS: Sent to {number}")
        return True
    print(f"  SMS: Send failed ({number}): {out.strip()}")
    return False


def _open_modem():
    for baud in SMS_BAUD_LIST:
        try:
            ser = serial.Serial(SMS_PORT, baud, timeout=5)
        except Exception as e:
            print(f"  SMS: Open {SMS_PORT}@{baud} failed: {e}")
            continue
        ok, _ = _at(ser, "AT")
        if ok:
            print(f"  SMS: Modem found @ {baud} baud")
            return ser
        ser.close()
    return None


def _init_modem(ser):
    for cmd in ("ATE0", "AT+CMEE=2", "AT+CMGF=1", 'AT+CSCS="GSM"'):
        _at(ser, cmd)


# ── Public class ───────────────────────────────────────────────────────────

class SMSAlerter:
    """
    Thread-safe SMS alerter with a configurable cooldown.

    Usage:
        alerter = SMSAlerter()
        alerter.alert()   # call whenever elephant is detected
    """

    def __init__(self):
        self._last_sent = 0.0
        self._lock = Lock()
        self._ser = None
        self._ready = False
        Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        ser = _open_modem()
        if ser is None:
            print("  SMS: No modem detected — SMS disabled")
            return
        _init_modem(ser)
        if not _wait_network(ser):
            print("  SMS: Network registration timeout — SMS disabled")
            ser.close()
            return
        self._ser = ser
        self._ready = True
        print("  SMS: Ready")

    def alert(self, message: str = SMS_MESSAGE):
        """Send SMS burst if cooldown has elapsed (non-blocking)."""
        with self._lock:
            if not self._ready:
                return
            elapsed = time.time() - self._last_sent
            if elapsed < SMS_COOLDOWN:
                remaining = int(SMS_COOLDOWN - elapsed)
                print(f"  SMS: Cooldown active ({remaining}s remaining)")
                return
            self._last_sent = time.time()   # mark now so concurrent calls skip
        Thread(target=self._send_burst, args=(message,), daemon=True).start()

    def _send_burst(self, message):
        print(f"  SMS: Sending alert to {len(SMS_NUMBERS)} number(s)…")
        for number in SMS_NUMBERS:
            _send_one(self._ser, number, message)
            time.sleep(2)

    def cleanup(self):
        if self._ser:
            self._ser.close()
