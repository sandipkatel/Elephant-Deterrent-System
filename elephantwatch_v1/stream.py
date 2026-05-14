import time
from threading import Thread, Lock

import cv2
from flask import Flask, Response


class MjpegServer:
    def __init__(self, host, port, idle_frame):
        self._host = host
        self._port = port
        self._app = Flask(__name__)
        self._frame_lock = Lock()
        self._latest_frame = None
        self._idle_frame = idle_frame

        self._app.add_url_rule("/", "index", self._index)
        self._app.add_url_rule("/stream", "stream", self._stream)

    def _index(self):
        return (
            '<html><head><title>Elephant Detection</title></head>'
            '<body style="margin:0;background:#111;display:flex;'
            'justify-content:center;align-items:center;height:100vh">'
            '<img src="/stream" style="max-width:100%;max-height:100vh"/>'
            '</body></html>'
        )

    def _generate_mjpeg(self):
        while True:
            with self._frame_lock:
                frame = self._latest_frame if self._latest_frame is not None else self._idle_frame
                _, jpeg = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)

    def _stream(self):
        return Response(self._generate_mjpeg(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def set_frame(self, frame):
        with self._frame_lock:
            self._latest_frame = frame

    def set_idle(self, frame):
        with self._frame_lock:
            self._latest_frame = frame

    def start(self):
        thread = Thread(target=self._app.run, kwargs={
            "host": self._host,
            "port": self._port,
            "threaded": True
        }, daemon=True)
        thread.start()
        return thread
