# Central configuration for the elephant guard system.
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
LED_PATH = "/sys/class/leds/ACT"
CONFIDENCE = 0.8
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_SIZE = 256
INFER_EVERY = 10
CAMERA_FPS = 15

STREAM_PORT = 5000

# PIR Sensor
GPIO_CHIP = "/dev/gpiochip4"
PIR_LINE = 27
PIR_COOLDOWN = 30
PIR_WARMUP = 5

# Audio alert
BEE_SOUND_PATH = BASE_DIR / "media" / "bee.mp3"
BUZZER_INTERVAL = 5.0
AUDIO_DEVICE = "hw:2,0"
AUDIO_VOLUME = 5

# GSM (SIM800L/SIM900 via UART)
ENABLE_GSM = True
GSM_PORT = "/dev/ttyAMA0"
GSM_BAUD = 9600
SMS_RECIPIENT = "+9779848588910"
SMS_COOLDOWN = 60.0
SMS_TEXT = "Elephant detected"
SIM_PIN = None  # e.g. "1234" if SIM requires a PIN

# GSM diagnostics and retries
GSM_WAIT_FOR_NETWORK = True
GSM_NETWORK_TIMEOUT = 30.0
GSM_RETRY_SECONDS = 10.0
