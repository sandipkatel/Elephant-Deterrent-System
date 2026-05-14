import sys
from config import LED_PATH


def setup():
    try:
        Path(LED_PATH + "/trigger").write_text("none")
        print("  LED             : Active")
    except PermissionError:
        print("  ERROR: Run with sudo!")
        sys.exit(1)
    except Exception as e:
        print(f"  LED Error       : {e}")


def set(state: bool):
    try:
        open(LED_PATH + "/brightness", "w").write("1" if state else "0")
    except Exception:
        pass


def restore():
    try:
        open(LED_PATH + "/trigger", "w").write("mmc0")
    except Exception:
        pass


# Allow `from led import Path` to work without importing pathlib everywhere
from pathlib import Path  # noqa: E402 (kept at bottom intentionally)
