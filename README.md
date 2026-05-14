Intelligent Elephant Detection & Deterrent System
=========================

Overview
--------
The Intelligent Elephant Detection and Deterrent System is an integrated hardware + software project that detects elephant presence near Buffer Zone and triggers deterrent actions (alerts, buzzers, SMS, dashboard updates). It contains two co-existing versions of the core application:
- `mobile-app`: Flutter application developed specailly for remote monitoring and Raspberry Pi's CPU operation like Voltage, Temperature.
- `elephantwatch_v2`: Reworked server/dashboard oriented version with web dashboard, LED control, SMS and a server component for remote monitoring.

This repository also includes a trained models and test scripts used during development.

Highlights
----------

- Multiple deployment targets: Raspberry Pi 5, and a server/dashboard + peripheral controllers in `elephantwatch_v2`.
- Computer-vision models located in `models/` (e.g., `best_float32.tflite`, PyTorch weights) for detecting animals in camera frames.
- Documentation and a project report in `Documentation/Report.pdf`.
- A Flutter mobile app scaffold under `mobile-app/` for companion functionality.

Quick Start
-----------
1. Clone the Repository in Raspberry Pi 5 OS:
   ```
   git clone https://github.com/sandipkatel/Elephant-Detection-&-Deterrent-System
   ```
2. Install Python dependencies (recommended in a virtualenv):

```
python -m venv .venv
.venv\Scripts\activate   # on Windows
pip install -r requirements.txt
```

2. Run the device/embedded version (v2):

```
sudo .venv/bin/python elephantwatch_v2/main
```

Models
------

Model files are available under `models/`. The repository contains multiple formats (TFLite, PyTorch). Choose the model matching your deployment constraints (e.g., `best_float32.tflite` for edge devices).


Documentation
-------------

See the full project report for design rationale, experimental results, hardware schematics, and evaluation metrics in [Documentation/Report.pdf](https://github.com/sandipkatel/Elephant-Deterrent-System/Documentation/Report.pdf).
You can also find project presentation slides in [Canva](https://canva.link/zdheqfcpkp3b6ap)

Contributers
------------
[Sandip Katel](https://github.com/sandipkatel)
[Saphal Rimal](https://github.com/saphalr)
[Sijan Joshi](https://github.com/sijanj)
