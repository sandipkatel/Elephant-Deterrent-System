import subprocess
from threading import Thread, Lock
from config import SOUND_FILE, AUDIO_DEVICE, VOLUME


class Buzzer:
    """Plays a sound file through a USB audio device (non-blocking)."""

    def __init__(self):
        self._busy = False
        self._lock = Lock()
        print(f"  Buzzer          : Ready ({AUDIO_DEVICE})")

    def _play(self):
        try:
            command = (
                f'ffmpeg -i {SOUND_FILE} -filter:a "volume={VOLUME}" '
                f'-ac 2 -ar 44100 -f wav - | aplay -D {AUDIO_DEVICE}'
            )
            subprocess.run(command, shell=True)
        except Exception as e:
            print(f"  Buzzer error: {e}")
        finally:
            with self._lock:
                self._busy = False

    def beep(self):
        """Play the sound once; silently skips if already playing."""
        with self._lock:
            if self._busy:
                return
            self._busy = True
        Thread(target=self._play, daemon=True).start()

    def cleanup(self):
        pass  # nothing to release