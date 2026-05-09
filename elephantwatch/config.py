from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH   = Path(__file__).resolve().parent.parent / "models" / "best.pt"
LED_PATH     = "/sys/class/leds/ACT"

# ── Camera ─────────────────────────────────────────────────────────────────
CAMERA_WIDTH  = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS    = 30

# ── Inference ──────────────────────────────────────────────────────────────
YOLO_SIZE        = 256
INFER_EVERY      = 10
CONFIDENCE       = 0.6
TARGET_CLASS     = "elephant"

# ── PIR ────────────────────────────────────────────────────────────────────
GPIO_CHIP    = "/dev/gpiochip4"
PIR_LINE     = 27
PIR_COOLDOWN = 30          # seconds camera stays on after last motion
PIR_WARMUP   = 5

# ── Buzzer ─────────────────────────────────────────────────────────────────
SOUND_FILE = "media/bee.mp3"
AUDIO_DEVICE = "hw:2,0"
VOLUME = 100
BUZZER_INTERVAL = 5.0      # seconds between successive beeps

# ── SMS ────────────────────────────────────────────────────────────────────
SMS_PORT        = "/dev/ttyAMA0"
SMS_BAUD_LIST   = [9600, 115200, 57600, 38400, 19200]
SMS_NUMBERS     = [
    # "+9779749840709",
    # "+9779848588910",
    "+9779863481927",
]
SMS_MESSAGE         = "🐘 ElephantWatch ALERT: Elephant detected at site!"
SMS_COOLDOWN        = 1800  # 30 minutes between SMS bursts

# ── Web dashboard ──────────────────────────────────────────────────────────
STREAM_PORT = 5000
