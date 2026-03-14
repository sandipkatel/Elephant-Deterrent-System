import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Tuple

import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO

# --- Flask stream state ---
app = Flask(__name__)
latest_frame = None
frame_lock = Lock()


@app.route('/')
def index():
    return (
        '<html><head><title>Elephant Detection</title></head>'
        '<body style="margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh">'
        '<img src="/stream" style="max-width:100%;max-height:100vh"/>'
        '</body></html>'
    )


@app.route('/stream')
def stream():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_mjpeg():
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            ok, jpeg = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)


def start_stream_server(port: int):
    app.run(host='0.0.0.0', port=port, threaded=True)


class CameraStream:
    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.stopped = False
        self.frame = None
        self._lock = Lock()

        cmd = [
            'rpicam-vid',
            '--width', str(width),
            '--height', str(height),
            '--framerate', str(fps),
            '--codec', 'yuv420',
            '--output', '-',
            '--timeout', '0',
            '--nopreview',
        ]

        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.reader = Thread(target=self._reader_loop, daemon=True)
        self.reader.start()

    def _reader_loop(self):
        frame_size = int(self.width * self.height * 1.5)

        while not self.stopped:
            raw = self.process.stdout.read(frame_size)
            if len(raw) < frame_size:
                err = self.process.stderr.read().decode(errors='ignore').strip()
                if err:
                    print(f'⚠️ rpicam-vid error:\n{err}')
                self.stopped = True
                break

            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            with self._lock:
                self.frame = rgb

    def read(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.stopped = True
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait()


class LedController:
    def __init__(self, led_path: str, enabled: bool):
        self.led_path = led_path
        self.enabled = enabled

    def setup(self):
        if not self.enabled:
            return
        try:
            with open(f'{self.led_path}/trigger', 'w') as f:
                f.write('none')
            print('✅ LED control active')
        except PermissionError:
            print("❌ LED permission denied. Run with sudo or disable LED with --no-led")
            sys.exit(1)
        except Exception as exc:
            print(f'❌ LED setup error: {exc}')
            sys.exit(1)

    def set_state(self, on: bool):
        if not self.enabled:
            return
        try:
            with open(f'{self.led_path}/brightness', 'w') as f:
                f.write('1' if on else '0')
        except Exception:
            pass

    def restore(self):
        if not self.enabled:
            return
        try:
            with open(f'{self.led_path}/trigger', 'w') as f:
                f.write('mmc0')
            print('🔄 LED restored to default')
        except Exception:
            pass


def draw_console(detections: List[Dict], frame_count: int, fps: float, model_path: str):
    print('\033[H\033[J', end='')
    print('=' * 58)
    print('   🐘 ELEPHANT DETECTION | Raspberry Pi 5 | rpicam-vid')
    print('=' * 58)
    print(f'  Model: {model_path}')
    print(f'  Frame: {frame_count}   FPS: {fps:.1f}')
    print('-' * 58)

    if detections:
        print(f'  🚨 STATUS: ELEPHANT DETECTED ({len(detections)} object(s))')
        for i, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            print(f'   [{i}] conf={conf:.1%} box=({x1},{y1})→({x2},{y2})')
    else:
        print('  ✅ STATUS: Clear — no elephant detected')

    print('=' * 58)


def draw_overlay(frame_rgb, detections: List[Dict], fps: float, width: int, height: int):
    display = frame_rgb.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f'Elephant {conf:.1%}'
        (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_top = max(0, y1 - lh - base - 4)
        cv2.rectangle(display, (x1, y_top), (x1 + lw + 4, y1), (0, 0, 255), -1)
        cv2.putText(display, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(display, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if detections:
        banner_color = (0, 0, 180)
        banner = f'ELEPHANT DETECTED ({len(detections)})'
    else:
        banner_color = (0, 150, 0)
        banner = 'Clear - No Elephant'

    cv2.rectangle(display, (0, height - 40), (width, height), banner_color, -1)
    cv2.putText(display, banner, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return display


def parse_args():
    parser = argparse.ArgumentParser(description='Raspberry Pi 5 rpicam-vid inference with multi-format YOLO model support')
    parser.add_argument('--model', required=True, help='Model path (.pt, .tflite, *_ncnn_model directory, etc.)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument('--camera-fps', type=int, default=15, help='Camera capture FPS')
    parser.add_argument('--imgsz', type=int, default=256, help='Inference resize size')
    parser.add_argument('--infer-every', type=int, default=10, help='Run inference every N frames')
    parser.add_argument('--sleep-ms', type=int, default=20, help='Loop sleep in milliseconds')
    parser.add_argument('--stream-port', type=int, default=5000, help='Flask MJPEG stream port')
    parser.add_argument('--led-path', default='/sys/class/leds/ACT', help='LED sysfs path')
    parser.add_argument('--no-led', action='store_true', help='Disable LED control')
    parser.add_argument('--no-stream', action='store_true', help='Disable Flask MJPEG stream server')
    parser.add_argument('--save-video', default='', help='Optional output video path (mp4)')
    return parser.parse_args()


def ensure_model_path(path_str: str):
    p = Path(path_str)
    if p.exists():
        return
    if path_str.startswith('yolo'):
        return
    raise FileNotFoundError(f'Model not found: {path_str}')


def main():
    args = parse_args()
    ensure_model_path(args.model)

    led = LedController(args.led_path, enabled=not args.no_led)
    led.setup()

    print(f'Loading model: {args.model}')
    model = YOLO(args.model)
    print('✅ Model loaded')

    print('Starting camera via rpicam-vid...')
    cam = CameraStream(args.width, args.height, args.camera_fps)
    time.sleep(2)
    if cam.stopped:
        raise RuntimeError('Camera stream failed to start')
    print('✅ Camera streaming')

    if not args.no_stream:
        t = Thread(target=start_stream_server, args=(args.stream_port,), daemon=True)
        t.start()
        print(f'🌐 MJPEG stream: http://<pi-ip>:{args.stream_port}')

    writer = None
    if args.save_video:
        out_path = Path(args.save_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            max(1, args.camera_fps),
            (args.width, args.height),
        )
        print(f'💾 Saving video: {out_path}')

    running = True

    def on_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_sigint)

    frame_count = 0
    fps = 0.0
    fps_counter = 0
    fps_timer = time.time()
    detections: List[Dict] = []
    detected = False

    try:
        while running:
            if cam.stopped:
                print('❌ Camera stream ended')
                break

            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()

            if frame_count % max(1, args.infer_every) == 0:
                small = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
                results = model.predict(source=small, conf=args.conf, imgsz=args.imgsz, verbose=False)

                detections = []
                detected = False
                sx = args.width / float(args.imgsz)
                sy = args.height / float(args.imgsz)

                for result in results:
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    detected = True
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append({
                            'confidence': float(box.conf[0]),
                            'bbox': (
                                int(max(0, min(args.width - 1, x1 * sx))),
                                int(max(0, min(args.height - 1, y1 * sy))),
                                int(max(0, min(args.width - 1, x2 * sx))),
                                int(max(0, min(args.height - 1, y2 * sy))),
                            ),
                        })

                draw_console(detections, frame_count, fps, args.model)

            led.set_state(detected)

            overlay = draw_overlay(frame, detections, fps, args.width, args.height)
            bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

            if writer is not None:
                writer.write(bgr)

            with frame_lock:
                global latest_frame
                latest_frame = bgr

            time.sleep(max(0.0, args.sleep_ms / 1000.0))

    finally:
        if writer is not None:
            writer.release()
        cam.release()
        led.set_state(False)
        led.restore()
        print('\n👋 Shutdown complete. Camera and LED released.')


if __name__ == '__main__':
    main()
