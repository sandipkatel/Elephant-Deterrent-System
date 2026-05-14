Elephant Deterrent System
=========================

Overview
--------
The Elephant Deterrent System is an integrated hardware + software project that detects elephant presence near protected areas and triggers deterrent actions (alerts, buzzers, lights, SMS, dashboard updates). It contains two co-existing versions of the core application:

- `elephantwatch_v1`: Original monolithic embedded-focused runtime with sensor integrations (buzzer, camera, GSM, PIR, UI).
- `elephantwatch_v2`: Reworked server/dashboard oriented version with web dashboard, LED control, SMS and a server component for remote monitoring.

This repository also includes a mobile app scaffold, trained models and test scripts used during development.

Highlights
----------

- Multiple deployment targets: Raspberry Pi / embedded device code in `elephantwatch_v1`, and a server/dashboard + peripheral controllers in `elephantwatch_v2`.
- Computer-vision models located in `models/` (e.g., `best_float32.tflite`, PyTorch weights) for detecting animals in camera frames.
- Documentation and a project report in `Documentation/Report.pdf`.
- A Flutter mobile app scaffold under `mobile-app/` for companion functionality.

Repository Structure
--------------------

- `elephantwatch_v1/` — v1 runtime and device-oriented modules. Key modules:
  - `buzzer.py`, `camera.py`, `gsm.py`, `sensors.py`, `stream.py`, `ui.py`, `main.py`

- `elephantwatch_v2/` — v2 runtime with server/dashboard and revised peripherals. Key files:
  - `main.py`, `server.py`, `sms.py`, `led.py`, `pir.py`, `dashboard.html`

- `models/` — trained models and related scripts used by the detection pipeline.

- `mobile-app/` — Flutter mobile application scaffold (Android, iOS, web, desktop targets).

- `Documentation/Report.pdf` — detailed design, experiments, and results. Refer to this for architecture diagrams and evaluation metrics.

- `requirements.txt` — Python dependencies for the server/device code (install with `pip`).

- `test/` — assorted test scripts used during development and validation.

Quick Start
-----------

1. Install Python dependencies (recommended in a virtualenv):

```
python -m venv .venv
.venv\Scripts\activate   # on Windows
pip install -r requirements.txt
```

2. Run the device/embedded version (v1):

```
python -m elephantwatch_v1.main
```

or directly:

```
python elephantwatch_v1/main.py
```

3. Run the server/dashboard version (v2):

```
python -m elephantwatch_v2.server
```

Notes: both `elephantwatch_v1` and `elephantwatch_v2` contain `main.py` entry points. The exact runtime flags or configuration are defined in each module's `config.py` — check those files for device-specific settings (serial ports, GPIO pins, camera indexes, GSM config, and SMS gateway settings).

Models
------

Model files are available under `models/`. The repository contains multiple formats (TFLite, PyTorch). Choose the model matching your deployment constraints (e.g., `best_float32.tflite` for edge devices).

Development & Testing
---------------------

- Use scripts in `test/` for quick checks (camera tests, PIR tests, SMS tests).
- Unit / integration tests are ad-hoc; run the individual scripts to validate hardware integrations.

Documentation
-------------

See the full project report for design rationale, experimental results, hardware schematics, and evaluation metrics in `Documentation/Report.pdf`.

Contributing
------------

If you'd like to contribute:

1. Open an issue describing the change.
2. Create a branch named `feature/...` or `fix/...`.
3. Submit a pull request with descriptive commit messages.

License
-------

This project includes a `LICENSE` file at the repository root — consult it for license terms.

Contact
-------

For questions, open an issue or contact the maintainers listed in the project metadata.
