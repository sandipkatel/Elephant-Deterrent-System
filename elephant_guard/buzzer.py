import shutil
import subprocess
from threading import Thread, Lock

from . import config


class Buzzer:
    """Plays a bee sound via ffmpeg piped to aplay (USB sound device)."""

    def __init__(self):
        self._process = None
        self._available = False
        self._busy = False
        self._lock = Lock()

        self._sound_path = config.BEE_SOUND_PATH
        self._audio_device = config.AUDIO_DEVICE
        self._volume = config.AUDIO_VOLUME

        if not self._sound_path.exists():
            print(f"  Buzzer warning  : missing sound file ({self._sound_path})")
            return

        if shutil.which("ffmpeg") is None or shutil.which("aplay") is None:
            print("  Buzzer warning  : ffmpeg/aplay not found")
            return

        self._available = True
        print(f"  Buzzer          : Ready (ffmpeg -> aplay on {self._audio_device})")

    def _bee_buzz(self):
        ffmpeg_proc = None
        try:
            ffmpeg_proc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(self._sound_path),
                    "-filter:a",
                    f"volume={self._volume}",
                    "-ac",
                    "2",
                    "-ar",
                    "44100",
                    "-f",
                    "wav",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self._process = subprocess.Popen(
                ["aplay", "-q", "-D", self._audio_device],
                stdin=ffmpeg_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            if ffmpeg_proc.stdout is not None:
                ffmpeg_proc.stdout.close()

            _, aplay_err = self._process.communicate()
            ffmpeg_err = ffmpeg_proc.stderr.read().decode(errors="ignore") if ffmpeg_proc.stderr else ""
            ffmpeg_rc = ffmpeg_proc.wait()

            if self._process.returncode != 0:
                err = (aplay_err or "").strip()
                print(f"  Buzzer warning  : aplay failed ({err or self._process.returncode})")
            if ffmpeg_rc != 0:
                err = (ffmpeg_err or "").strip()
                print(f"  Buzzer warning  : ffmpeg failed ({err or ffmpeg_rc})")
        except Exception as e:
            print(f"  Buzzer warning  : failed to play sound ({e})")
        finally:
            if ffmpeg_proc is not None and ffmpeg_proc.poll() is None:
                try:
                    ffmpeg_proc.terminate()
                except Exception:
                    pass
            with self._lock:
                self._busy = False
                self._process = None

    def buzz(self):
        """Play the bee alert sound once in a background thread."""
        if not self._available:
            return
        with self._lock:
            if self._busy:
                return
            self._busy = True
        Thread(target=self._bee_buzz, daemon=True).start()

    def cleanup(self):
        with self._lock:
            proc = self._process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
