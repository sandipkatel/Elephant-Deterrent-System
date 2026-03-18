import re
import time
import serial

PORT = "/dev/ttyAMA0"
BAUD_CANDIDATES = [9600, 115200, 57600, 38400, 19200]
TIMEOUT = 5

numbers = [
    # "+9779848588910",
    "+9779863481927",
]

message = "Hello from PiElephant!"


def read_until(ser, tokens, timeout=5.0):
    end_time = time.time() + timeout
    buffer = ""
    while time.time() < end_time:
        data = ser.read(ser.in_waiting or 1)
        if data:
            buffer += data.decode(errors="ignore")
            for token in tokens:
                if token in buffer:
                    return token, buffer
        time.sleep(0.05)
    return None, buffer


def send_at(ser, cmd, expect=("OK",), timeout=5.0):
    ser.reset_input_buffer()
    ser.write((cmd + "\r").encode())
    token, output = read_until(ser, expect, timeout=timeout)
    return token in expect, output


def wait_for_network(ser, timeout=30.0):
    start = time.time()
    while time.time() - start < timeout:
        ok, out = send_at(ser, "AT+CREG?", expect=("OK",), timeout=2.0)
        if ok:
            match = re.search(r"\+CREG:\s*\d,(\d)", out)
            if match and match.group(1) in ("1", "5"):
                return True
        time.sleep(2.0)
    return False


def send_sms(ser, number, text):
    ok, out = send_at(ser, f'AT+CMGS="{number}"', expect=(">",), timeout=5.0)
    if not ok:
        print(f"CMGS prompt failed: {out.strip()}")
        return False
    ser.write(text.encode("utf-8") + b"\x1A")
    token, output = read_until(ser, ("OK", "ERROR"), timeout=30.0)
    if token == "OK":
        print(f"Sent to {number}")
        return True
    print(f"Send failed: {output.strip()}")
    return False


ser = None
detected_baud = None
for baud in BAUD_CANDIDATES:
    try:
        trial = serial.Serial(PORT, baud, timeout=TIMEOUT)
    except Exception as e:
        print(f"Open {PORT} @ {baud} failed: {e}")
        continue
    ok, out = send_at(trial, "AT")
    if ok:
        ser = trial
        detected_baud = baud
        print(f"Detected baud: {baud}")
        break
    trial.close()

if ser is None:
    print("No modem response on common baud rates")
    raise SystemExit(2)

send_at(ser, "ATE0")
send_at(ser, "AT+CMEE=2")
send_at(ser, "AT+CMGF=1")
send_at(ser, 'AT+CSCS="GSM"')

ok, out = send_at(ser, "AT+CPIN?", expect=("OK",))
if out.strip():
    print(out.strip())

ok, out = send_at(ser, "AT+CSQ", expect=("OK",))
if out.strip():
    print(out.strip())

ok, out = send_at(ser, "AT+CREG?", expect=("OK",))
if out.strip():
    print(out.strip())

if not wait_for_network(ser, timeout=30.0):
    print("Network registration timeout")

for number in numbers:
    send_sms(ser, number, message)
    time.sleep(2)

ser.close()
