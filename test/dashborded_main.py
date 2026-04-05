import sys
import json
import signal
import subprocess
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import time
import cv2
import gpiod
import lgpio
import psutil
from flask import Flask, Response
from threading import Thread, Lock

# --- CONFIGURATION ---
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"
LED_PATH = "/sys/class/leds/ACT"
CONFIDENCE = 0.8
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

YOLO_SIZE = 256
INFER_EVERY = 10

TARGET_CLASS_NAME = "elephant"

STREAM_PORT = 5000

# PIR Sensor
GPIO_CHIP = "/dev/gpiochip4"
PIR_LINE = 27
PIR_COOLDOWN = 30
PIR_WARMUP = 5

# Buzzer
BUZZER_PIN = 18
BUZZER_DURATION = 0.3
BUZZER_INTERVAL = 5.0

# --- DASHBOARD HTML (embedded) ---
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ElephantWatch \u2014 Field Control</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #0a0c0f;
    --bg2: #111418;
    --bg3: #181d23;
    --border: rgba(255,255,255,0.06);
    --border-hi: rgba(255,255,255,0.13);
    --text: #e8eaed;
    --muted: #6b7280;
    --accent: #f59e0b;
    --accent2: #10b981;
    --danger: #ef4444;
    --info: #3b82f6;
    --font-display: 'Syne', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-display);
    min-height: 100vh;
    overflow-x: hidden;
  }
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px);
    pointer-events: none;
    z-index: 1000;
  }
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    height: 58px;
    border-bottom: 1px solid var(--border);
    background: var(--bg2);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.02em;
  }
  .logo-icon {
    width: 32px; height: 32px;
    background: var(--accent);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .logo span { color: var(--accent); }
  .topbar-right {
    display: flex;
    align-items: center;
    gap: 20px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
  }
  .clock { color: var(--text); font-size: 13px; }
  .status-pill {
    display: flex; align-items: center; gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    border: 1px solid;
    transition: all 0.4s;
  }
  .status-pill.standby { border-color: rgba(107,114,128,0.4); color: var(--muted); background: rgba(107,114,128,0.08); }
  .status-pill.active  { border-color: rgba(16,185,129,0.4); color: var(--accent2); background: rgba(16,185,129,0.08); }
  .status-pill.alert   { border-color: rgba(239,68,68,0.5); color: var(--danger); background: rgba(239,68,68,0.08); }
  .pill-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .pill-dot.pulse { animation: blink 1s ease-in-out infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
  .main {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 1px;
    background: var(--border);
    min-height: calc(100vh - 58px);
  }
  .camera-panel {
    background: var(--bg);
    position: relative;
    display: flex;
    flex-direction: column;
  }
  .panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--bg2);
  }
  .panel-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .camera-feed {
    flex: 1;
    position: relative;
    background: #000;
    overflow: hidden;
    min-height: 400px;
  }
  .camera-feed img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .camera-overlay {
    position: absolute;
    inset: 0;
    pointer-events: none;
  }
  .corner { position: absolute; width: 24px; height: 24px; border-color: var(--accent); border-style: solid; opacity: 0.6; }
  .corner.tl { top: 16px; left: 16px; border-width: 2px 0 0 2px; }
  .corner.tr { top: 16px; right: 16px; border-width: 2px 2px 0 0; }
  .corner.bl { bottom: 56px; left: 16px; border-width: 0 0 2px 2px; }
  .corner.br { bottom: 56px; right: 16px; border-width: 0 2px 2px 0; }
  .camera-hud {
    position: absolute;
    top: 20px; left: 50%; transform: translateX(-50%);
    font-family: var(--font-mono);
    font-size: 10px;
    color: rgba(245,158,11,0.7);
    letter-spacing: 0.15em;
    text-align: center;
    white-space: nowrap;
  }
  .alert-banner {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    padding: 12px 20px;
    display: flex; align-items: center; gap: 10px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.05em;
    transition: background 0.3s, color 0.3s;
  }
  .alert-banner.clear   { background: rgba(16,185,129,0.15); color: var(--accent2); border-top: 1px solid rgba(16,185,129,0.2); }
  .alert-banner.elephant{ background: rgba(239,68,68,0.2); color: #fca5a5; border-top: 1px solid rgba(239,68,68,0.3); animation: alertpulse 1.5s ease-in-out infinite; }
  .alert-banner.standby { background: rgba(107,114,128,0.1); color: var(--muted); border-top: 1px solid var(--border); }
  @keyframes alertpulse { 0%,100%{background:rgba(239,68,68,0.15)} 50%{background:rgba(239,68,68,0.3)} }
  .camera-footer {
    display: flex; gap: 1px;
    background: var(--border);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }
  .cam-stat {
    flex: 1;
    background: var(--bg2);
    padding: 10px 14px;
    font-family: var(--font-mono);
  }
  .cam-stat-label { font-size: 9px; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 3px; }
  .cam-stat-val { font-size: 14px; font-weight: 500; color: var(--text); }
  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 1px;
    background: var(--border);
    overflow-y: auto;
    max-height: calc(100vh - 58px);
  }
  .stat-card {
    background: var(--bg2);
    padding: 18px 20px;
    flex-shrink: 0;
  }
  .stat-card-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
  }
  .stat-card-label::before {
    content: '';
    display: inline-block;
    width: 3px; height: 10px;
    background: var(--accent);
    border-radius: 2px;
  }
  .stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }
  .stat-name { font-size: 12px; color: var(--muted); font-family: var(--font-mono); }
  .stat-value { font-size: 13px; font-weight: 500; color: var(--text); font-family: var(--font-mono); }
  .stat-value.good { color: var(--accent2); }
  .stat-value.warn { color: var(--accent); }
  .stat-value.danger { color: var(--danger); }
  .bar-wrap { height: 3px; background: var(--bg3); border-radius: 2px; margin-top: 2px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 2px; transition: width 0.8s ease, background 0.4s; }
  .log-panel {
    background: var(--bg2);
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 160px;
  }
  .log-list {
    flex: 1;
    overflow-y: auto;
    padding: 4px 0;
  }
  .log-list::-webkit-scrollbar { width: 3px; }
  .log-list::-webkit-scrollbar-track { background: transparent; }
  .log-list::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }
  .log-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 14px;
    font-family: var(--font-mono);
    font-size: 11px;
    border-bottom: 1px solid var(--border);
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn { from{opacity:0;transform:translateY(-4px)} to{opacity:1;transform:none} }
  .log-time { color: var(--muted); min-width: 58px; flex-shrink: 0; }
  .log-type { font-weight: 500; min-width: 52px; flex-shrink: 0; }
  .log-type.motion   { color: var(--info); }
  .log-type.elephant { color: var(--danger); }
  .log-type.clear    { color: var(--accent2); }
  .log-type.standby  { color: var(--muted); }
  .log-type.buzz     { color: var(--accent); }
  .log-type.system   { color: #a78bfa; }
  .log-msg { color: #9ca3af; flex: 1; }
  .det-panel {
    background: var(--bg2);
    padding: 16px;
    flex-shrink: 0;
  }
  .det-list { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
  .det-item {
    background: var(--bg3);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 6px;
    padding: 10px 12px;
    font-family: var(--font-mono);
    font-size: 11px;
  }
  .det-conf { font-size: 16px; font-weight: 700; color: var(--danger); margin-bottom: 4px; }
  .det-bbox { color: var(--muted); line-height: 1.6; }
  .det-empty { font-family: var(--font-mono); font-size: 12px; color: var(--muted); text-align: center; padding: 12px 0; }
  .sparkline-wrap { margin-top: 10px; }
  canvas.spark { width: 100%; height: 36px; display: block; }
  .alert-flash {
    position: fixed;
    inset: 0;
    background: rgba(239,68,68,0.07);
    pointer-events: none;
    opacity: 0;
    z-index: 999;
    transition: opacity 0.2s;
  }
  .alert-flash.show { opacity: 1; }
  @media (max-width: 860px) {
    .main { grid-template-columns: 1fr; }
    .sidebar { max-height: none; overflow-y: visible; }
    .camera-feed { min-height: 260px; }
  }
</style>
</head>
<body>

<div class="alert-flash" id="alertFlash"></div>

<header class="topbar">
  <div class="logo">
    <div class="logo-icon">&#x1F418;</div>
    Elephant<span>Watch</span>
  </div>
  <div class="topbar-right">
    <div class="clock" id="clock">--:--:--</div>
    <div class="status-pill standby" id="statusPill">
      <div class="pill-dot" id="pillDot"></div>
      <span id="pillText">STANDBY</span>
    </div>
  </div>
</header>

<div class="main">

  <!-- CAMERA -->
  <div class="camera-panel">
    <div class="panel-header">
      <div class="panel-title">Live Feed &mdash; rpicam-vid</div>
      <div style="font-family:var(--font-mono);font-size:11px;color:var(--muted)">1920&times;1080 &middot; 30fps &middot; YOLOv8</div>
    </div>
    <div class="camera-feed">
      <img src="/stream" id="liveImg" alt="Camera feed"/>
      <div class="camera-overlay">
        <div class="corner tl"></div>
        <div class="corner tr"></div>
        <div class="corner bl"></div>
        <div class="corner br"></div>
        <div class="camera-hud">&#9679; REC &nbsp;&middot;&nbsp; YOLO-v8 &nbsp;&middot;&nbsp; CONF 80% &nbsp;&middot;&nbsp; INFER/10fr</div>
      </div>
      <div class="alert-banner standby" id="alertBanner">
        <span id="alertIcon">&#9711;</span>
        <span id="alertText">STANDBY &mdash; Waiting for motion</span>
      </div>
    </div>
    <div class="camera-footer">
      <div class="cam-stat"><div class="cam-stat-label">Frames</div><div class="cam-stat-val" id="frameCount">&mdash;</div></div>
      <div class="cam-stat"><div class="cam-stat-label">FPS</div><div class="cam-stat-val" id="fpsVal">&mdash;</div></div>
      <div class="cam-stat"><div class="cam-stat-label">Cooldown</div><div class="cam-stat-val" id="cooldownVal">&mdash;</div></div>
      <div class="cam-stat"><div class="cam-stat-label">Last motion</div><div class="cam-stat-val" id="lastMotion">&mdash;</div></div>
      <div class="cam-stat"><div class="cam-stat-label">Camera</div><div class="cam-stat-val" id="camStatus">&mdash;</div></div>
    </div>
  </div>

  <!-- SIDEBAR -->
  <div class="sidebar">

    <!-- System Health -->
    <div class="stat-card">
      <div class="stat-card-label">System Health</div>

      <div class="stat-row">
        <span class="stat-name">CPU Temp</span>
        <span class="stat-value" id="cpuTemp">&mdash;</span>
      </div>
      <div class="bar-wrap"><div class="bar-fill" id="cpuTempBar" style="width:0%;background:var(--accent)"></div></div>

      <div class="stat-row" style="margin-top:10px">
        <span class="stat-name">CPU Usage</span>
        <span class="stat-value" id="cpuUsage">&mdash;</span>
      </div>
      <div class="bar-wrap"><div class="bar-fill" id="cpuUsageBar" style="width:0%;background:var(--info)"></div></div>

      <div class="stat-row" style="margin-top:10px">
        <span class="stat-name">RAM</span>
        <span class="stat-value" id="ramUsage">&mdash;</span>
      </div>
      <div class="bar-wrap"><div class="bar-fill" id="ramBar" style="width:0%;background:var(--accent2)"></div></div>

      <div class="stat-row" style="margin-top:10px">
        <span class="stat-name">Disk</span>
        <span class="stat-value" id="diskUsage">&mdash;</span>
      </div>
      <div class="bar-wrap"><div class="bar-fill" id="diskBar" style="width:0%;background:#8b5cf6"></div></div>

      <div class="stat-row" style="margin-top:10px">
        <span class="stat-name">Uptime</span>
        <span class="stat-value good" id="uptime">&mdash;</span>
      </div>
      <div class="stat-row">
        <span class="stat-name">Throttle</span>
        <span class="stat-value good" id="throttle">&mdash;</span>
      </div>

      <div class="sparkline-wrap">
        <div style="font-size:9px;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">CPU Temp history (60 samples)</div>
        <canvas class="spark" id="sparkCanvas"></canvas>
      </div>
    </div>

    <!-- Session Stats -->
    <div class="stat-card">
      <div class="stat-card-label">Session Stats</div>
      <div class="stat-row">
        <span class="stat-name">Mode</span>
        <span class="stat-value" id="modeVal">&mdash;</span>
      </div>
      <div class="stat-row">
        <span class="stat-name">Elephant detections</span>
        <span class="stat-value danger" id="detCount">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-name">Buzzer fires</span>
        <span class="stat-value warn" id="buzzCount">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-name">PIR events</span>
        <span class="stat-value" id="pirCount">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-name">LED state</span>
        <span class="stat-value" id="ledState">&mdash;</span>
      </div>
    </div>

    <!-- Active Detections -->
    <div class="det-panel">
      <div class="stat-card-label">Active Detections</div>
      <div class="det-list" id="detList">
        <div class="det-empty">No detections</div>
      </div>
    </div>

    <!-- Event Log -->
    <div class="log-panel">
      <div class="panel-header">
        <div class="panel-title">Event Log</div>
        <div style="font-family:var(--font-mono);font-size:10px;color:var(--muted)" id="logCount">0 events</div>
      </div>
      <div class="log-list" id="logList"></div>
    </div>

  </div>
</div>

<script>
// Clock
function updateClock() {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString('en-GB');
}
setInterval(updateClock, 1000);
updateClock();

// Sparkline
const sparkCanvas = document.getElementById('sparkCanvas');
const sCtx = sparkCanvas.getContext('2d');
const tempHistory = [];
const MAX_SPARK = 60;

function drawSparkline() {
  const W = sparkCanvas.offsetWidth || 280;
  const H = 36;
  sparkCanvas.width = W;
  sparkCanvas.height = H;
  sCtx.clearRect(0, 0, W, H);
  if (tempHistory.length < 2) return;
  const minV = Math.min(...tempHistory) - 2;
  const maxV = Math.max(...tempHistory) + 2;
  const range = maxV - minV || 1;
  const pts = tempHistory.map((v, i) => ({
    x: (i / (tempHistory.length - 1)) * W,
    y: H - ((v - minV) / range) * (H - 4) - 2
  }));
  sCtx.beginPath();
  sCtx.moveTo(pts[0].x, pts[0].y);
  pts.slice(1).forEach(p => sCtx.lineTo(p.x, p.y));
  sCtx.strokeStyle = '#f59e0b';
  sCtx.lineWidth = 1.5;
  sCtx.stroke();
  sCtx.lineTo(W, H); sCtx.lineTo(0, H); sCtx.closePath();
  sCtx.fillStyle = 'rgba(245,158,11,0.08)';
  sCtx.fill();
  const last = pts[pts.length - 1];
  sCtx.beginPath();
  sCtx.arc(last.x, last.y, 3, 0, Math.PI * 2);
  sCtx.fillStyle = '#f59e0b';
  sCtx.fill();
}

// Event log
const logs = [];
let logCount = 0;
let buzzerFires = 0;
let pirEvents = 0;
let detectionCount = 0;

function addLog(type, msg) {
  const time = new Date().toLocaleTimeString('en-GB');
  logs.unshift({ time, type, msg });
  if (logs.length > 200) logs.pop();
  logCount++;
  renderLog();
}

function renderLog() {
  const el = document.getElementById('logList');
  el.innerHTML = logs.slice(0, 50).map(l =>
    '<div class="log-item">' +
    '<span class="log-time">' + l.time + '</span>' +
    '<span class="log-type ' + l.type + '">' + l.type.toUpperCase() + '</span>' +
    '<span class="log-msg">' + l.msg + '</span>' +
    '</div>'
  ).join('');
  document.getElementById('logCount').textContent = logCount + ' events';
}

function renderDetections(detections) {
  const el = document.getElementById('detList');
  if (!detections || detections.length === 0) {
    el.innerHTML = '<div class="det-empty">No active detections</div>';
    return;
  }
  el.innerHTML = detections.map(function(d) {
    const w = d.bbox[2] - d.bbox[0];
    const h = d.bbox[3] - d.bbox[1];
    const cx = Math.round((d.bbox[0] + d.bbox[2]) / 2);
    const cy = Math.round((d.bbox[1] + d.bbox[3]) / 2);
    return '<div class="det-item">' +
      '<div class="det-conf">' + (d.confidence * 100).toFixed(1) + '% confidence</div>' +
      '<div class="det-bbox">Box: (' + d.bbox[0] + ', ' + d.bbox[1] + ') &rarr; (' + d.bbox[2] + ', ' + d.bbox[3] + ')</div>' +
      '<div class="det-bbox">Size: ' + w + '&times;' + h + 'px &middot; Center: (' + cx + ', ' + cy + ')</div>' +
      '</div>';
  }).join('');
}

function setBar(id, pct, dangerT, warnT) {
  const bar = document.getElementById(id);
  if (!bar) return;
  bar.style.width = Math.min(100, Math.max(0, pct)) + '%';
  if (dangerT && pct >= dangerT) bar.style.background = 'var(--danger)';
  else if (warnT && pct >= warnT) bar.style.background = 'var(--accent)';
}

function colorClass(val, dangerT, warnT) {
  if (dangerT && val >= dangerT) return 'stat-value danger';
  if (warnT && val >= warnT) return 'stat-value warn';
  return 'stat-value good';
}

// State tracking
let prevMode = null;
let prevElephant = false;
let prevBuzz = false;

async function poll() {
  try {
    const res = await fetch('/status');
    if (!res.ok) return;
    const d = await res.json();

    // Status pill
    const pill = document.getElementById('statusPill');
    const dot  = document.getElementById('pillDot');
    const pillText = document.getElementById('pillText');
    if (d.elephant_detected) {
      pill.className = 'status-pill alert';
      dot.className  = 'pill-dot pulse';
      pillText.textContent = 'ELEPHANT DETECTED';
    } else if (d.mode === 'active') {
      pill.className = 'status-pill active';
      dot.className  = 'pill-dot pulse';
      pillText.textContent = 'ACTIVE \u2014 MONITORING';
    } else {
      pill.className = 'status-pill standby';
      dot.className  = 'pill-dot';
      pillText.textContent = 'STANDBY';
    }

    // Alert banner
    const banner    = document.getElementById('alertBanner');
    const alertIcon = document.getElementById('alertIcon');
    const alertText = document.getElementById('alertText');
    if (d.elephant_detected) {
      banner.className = 'alert-banner elephant';
      alertIcon.textContent = '\u26A0';
      alertText.textContent = 'ELEPHANT DETECTED \u2014 ' + d.detections.length + ' object(s) \u00B7 Deterrent active';
      if (!prevElephant) {
        const flash = document.getElementById('alertFlash');
        flash.classList.add('show');
        setTimeout(function(){ flash.classList.remove('show'); }, 600);
        addLog('elephant', 'Detected ' + d.detections.length + ' elephant(s)');
        detectionCount += d.detections.length;
        document.getElementById('detCount').textContent = detectionCount;
      }
    } else if (d.mode === 'active') {
      banner.className = 'alert-banner clear';
      alertIcon.textContent = '\u2713';
      alertText.textContent = 'Clear \u2014 No elephant detected';
    } else {
      banner.className = 'alert-banner standby';
      alertIcon.textContent = '\u25CB';
      alertText.textContent = 'STANDBY \u2014 Waiting for motion';
    }

    // Mode change logs
    if (d.mode !== prevMode) {
      if (d.mode === 'active') {
        addLog('motion', 'PIR triggered \u2014 camera started');
        pirEvents++;
        document.getElementById('pirCount').textContent = pirEvents;
      } else if (prevMode === 'active') {
        addLog('standby', 'Motion timeout \u2014 entering standby');
      }
      prevMode = d.mode;
    }
    if (!d.elephant_detected && prevElephant) {
      addLog('clear', 'No elephant in frame');
    }
    prevElephant = d.elephant_detected;

    // Camera footer
    document.getElementById('frameCount').textContent =
      d.frame_count != null ? d.frame_count.toLocaleString() : '\u2014';
    document.getElementById('fpsVal').textContent =
      d.fps ? d.fps.toFixed(1) : '\u2014';
    document.getElementById('lastMotion').textContent =
      (d.last_motion_ago != null && d.last_motion_ago < 9000)
        ? Math.round(d.last_motion_ago) + 's ago' : '\u2014';
    document.getElementById('cooldownVal').textContent =
      d.mode === 'active'
        ? Math.max(0, 30 - Math.round(d.last_motion_ago || 30)) + 's left'
        : '\u2014';
    document.getElementById('camStatus').textContent =
      d.mode === 'active' ? 'RUNNING' : 'OFF';
    document.getElementById('camStatus').className =
      'cam-stat-val ' + (d.mode === 'active' ? '' : '');

    // Mode
    document.getElementById('modeVal').textContent =
      d.mode === 'active' ? 'ACTIVE' : 'STANDBY';
    document.getElementById('modeVal').className =
      'stat-value ' + (d.mode === 'active' ? 'good' : '');

    // LED
    document.getElementById('ledState').textContent =
      d.elephant_detected ? 'ON' : 'OFF';
    document.getElementById('ledState').className =
      'stat-value ' + (d.elephant_detected ? 'danger' : 'good');

    // System metrics
    if (d.system) {
      const s = d.system;

      const cpuT = s.cpu_temp;
      const cpuTEl = document.getElementById('cpuTemp');
      cpuTEl.textContent = cpuT.toFixed(1) + '\u00B0C';
      cpuTEl.className = colorClass(cpuT, 75, 65);
      setBar('cpuTempBar', (cpuT / 85) * 100, 88, 76);
      document.getElementById('cpuTempBar').style.background =
        cpuT >= 75 ? 'var(--danger)' : 'var(--accent)';

      const cpuP = s.cpu_percent;
      document.getElementById('cpuUsage').textContent = cpuP.toFixed(0) + '%';
      document.getElementById('cpuUsage').className = colorClass(cpuP, 90, 70);
      setBar('cpuUsageBar', cpuP, 90, 70);

      const ramPct = s.ram_total > 0 ? (s.ram_used / s.ram_total * 100) : 0;
      document.getElementById('ramUsage').textContent =
        (s.ram_used / 1024).toFixed(1) + ' / ' + (s.ram_total / 1024).toFixed(1) + ' GB';
      document.getElementById('ramUsage').className = colorClass(ramPct, 90, 75);
      setBar('ramBar', ramPct, 90, 75);
      if (ramPct < 75) document.getElementById('ramBar').style.background = 'var(--accent2)';

      document.getElementById('diskUsage').textContent = s.disk_percent.toFixed(0) + '%';
      document.getElementById('diskUsage').className = colorClass(s.disk_percent, 90, 80);
      setBar('diskBar', s.disk_percent, 90, 80);
      if (s.disk_percent < 80) document.getElementById('diskBar').style.background = '#8b5cf6';

      document.getElementById('uptime').textContent = s.uptime;

      const thrEl = document.getElementById('throttle');
      thrEl.textContent = s.throttle;
      thrEl.className = 'stat-value ' + (s.throttle === 'OK' ? 'good' : 'danger');

      tempHistory.push(cpuT);
      if (tempHistory.length > MAX_SPARK) tempHistory.shift();
      drawSparkline();
    }

    // Buzzer
    if (d.buzzer_active && !prevBuzz) {
      addLog('buzz', 'Deterrent buzzer fired');
      buzzerFires++;
      document.getElementById('buzzCount').textContent = buzzerFires;
    }
    prevBuzz = !!d.buzzer_active;

    renderDetections(d.detections);

  } catch (e) {
    // silently retry — server may be mid-restart
  }
}

setInterval(poll, 1500);
poll();
</script>
</body>
</html>"""


# --- FLASK APP ---
app = Flask(__name__)
latest_frame = None
frame_lock = Lock()

# Shared state updated by main loop, read by /status endpoint
system_state = {
    "mode": "standby",
    "elephant_detected": False,
    "detections": [],
    "frame_count": 0,
    "fps": 0.0,
    "last_motion_ago": 9999,
    "buzzer_active": False,
}
state_lock = Lock()

# Idle placeholder image
idle_img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
cv2.putText(idle_img, "STANDBY - Waiting for motion...", (40, CAMERA_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 200), 2)
cv2.putText(idle_img, "PIR sensor active", (40, CAMERA_HEIGHT // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)


def generate_mjpeg():
    """Yield JPEG frames for the MJPEG stream."""
    while True:
        with frame_lock:
            frame = latest_frame if latest_frame is not None else idle_img
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)


def get_system_metrics():
    """Read live system metrics from the Pi."""
    # CPU temperature
    cpu_temp = 0.0
    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        cpu_temp = int(raw) / 1000.0
    except Exception:
        pass

    # Throttle status via vcgencmd
    throttle = "OK"
    try:
        out = subprocess.check_output(
            ["vcgencmd", "get_throttled"], text=True, timeout=2
        ).strip()
        hex_val = int(out.split("=")[1], 16)
        if hex_val != 0:
            flags = []
            if hex_val & 0x1: flags.append("UNDER-VOLT")
            if hex_val & 0x2: flags.append("FREQ-CAP")
            if hex_val & 0x4: flags.append("THROTTLED")
            if hex_val & 0x8: flags.append("TEMP-LIMIT")
            throttle = "+".join(flags) if flags else "WARN"
    except Exception:
        throttle = "N/A"

    # RAM
    vm = psutil.virtual_memory()

    # Disk
    du = psutil.disk_usage("/")

    # Uptime
    up_secs = int(time.time() - psutil.boot_time())
    h, rem = divmod(up_secs, 3600)
    m, s = divmod(rem, 60)
    uptime_str = f"{h}h {m:02d}m {s:02d}s"

    return {
        "cpu_temp":    round(cpu_temp, 1),
        "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
        "ram_used":    round(vm.used / 1024 / 1024),    # MB
        "ram_total":   round(vm.total / 1024 / 1024),   # MB
        "disk_percent": round(du.percent, 1),
        "uptime":      uptime_str,
        "throttle":    throttle,
    }


@app.route('/')
def index():
    """Serve the embedded dashboard."""
    return DASHBOARD_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route('/stream')
def stream():
    """MJPEG camera stream."""
    return Response(
        generate_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/status')
def status():
    """JSON status polled by dashboard every 1.5 s."""
    with state_lock:
        payload = dict(system_state)
        # Deep-copy detections list so it's safe to serialize
        payload["detections"] = list(system_state["detections"])
    payload["system"] = get_system_metrics()
    return Response(json.dumps(payload), mimetype="application/json")


def start_stream_server():
    app.run(host='0.0.0.0', port=STREAM_PORT, threaded=True)


# --- LED CONTROL ---

def setup_led():
    try:
        with open(f"{LED_PATH}/trigger", "w") as f:
            f.write("none")
        print("  LED Control     : Active")
    except PermissionError:
        print("  ERROR: Run with 'sudo'!")
        sys.exit(1)
    except Exception as e:
        print(f"  LED Error       : {e}")


def set_led(state):
    try:
        with open(f"{LED_PATH}/brightness", "w") as f:
            f.write("1" if state else "0")
    except Exception:
        pass


def restore_led():
    try:
        with open(f"{LED_PATH}/trigger", "w") as f:
            f.write("mmc0")
        print("  LED restored to default.")
    except Exception:
        pass


# --- CONSOLE DISPLAY ---

def draw_console(state, detections, frame_count, fps, last_motion_ago):
    print("\033[H\033[J", end="")
    print("=" * 55)
    print("   \U0001F418  ELEPHANT DETECTION  |  PIR + Camera")
    print("=" * 55)

    if state == "standby":
        print("  MODE   : \U0001F4A4 STANDBY  (camera off, PIR watching)")
        print(f"  MOTION : last {last_motion_ago:.0f}s ago")
        print("-" * 55)
        print("  \u2705 STATUS : Idle \u2014 Waiting for motion")
    else:
        print("  MODE   : \U0001F4F7 ACTIVE   (camera + YOLO running)")
        print(f"  MOTION : last {last_motion_ago:.0f}s ago")
        print(f"  Frame  : {frame_count:<10}  FPS: {fps:.1f}")
        print(f"  Cooldown: camera off in {max(0, PIR_COOLDOWN - last_motion_ago):.0f}s")
        print("-" * 55)

        if detections:
            print(f"  \U0001F6A8 STATUS : ELEPHANT DETECTED  ({len(detections)} object(s))")
            print()
            for i, det in enumerate(detections, 1):
                conf = det["confidence"]
                x1, y1, x2, y2 = det["bbox"]
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w // 2
                cy = y1 + h // 2
                print(f"  [{i}] Confidence : {conf:.1%}")
                print(f"      Box        : ({x1}, {y1}) -> ({x2}, {y2})")
                print(f"      Size       : {w}px x {h}px")
                print(f"      Center     : ({cx}, {cy})")
                print()
        else:
            print("  \u2705 STATUS : Clear \u2014 No elephant detected")

    print()
    print("=" * 55)
    print(f"  Ctrl+C to quit  |  Dashboard: http://pielephant.local:{STREAM_PORT}")
    print("=" * 55)


# --- DRAW ON FRAME ---

def draw_on_frame(frame, detections, fps):
    """Draw bounding boxes, labels and FPS on the OpenCV frame."""
    display = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"Elephant {conf:.1%}"
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display,
                      (x1, y1 - lh - baseline - 5),
                      (x1 + lw, y1),
                      (0, 0, 255), -1)
        cv2.putText(display, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(display,
                  (0, CAMERA_HEIGHT - 40),
                  (CAMERA_WIDTH, CAMERA_HEIGHT),
                  banner_color, -1)
    cv2.putText(display, banner_text, (10, CAMERA_HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return display


# --- CAMERA via rpicam-vid pipe (threaded) ---

class CameraStream:
    """Threaded camera reader — always has the latest frame ready."""

    def __init__(self):
        cmd = [
            "rpicam-vid",
            "--width",     str(CAMERA_WIDTH),
            "--height",    str(CAMERA_HEIGHT),
            "--framerate", str(CAMERA_FPS),
            "--codec",     "yuv420",
            "--output",    "-",
            "--timeout",   "0",
            "--nopreview",
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.frame = None
        self.stopped = False
        self._lock = Lock()
        self._thread = Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        frame_size = int(CAMERA_WIDTH * CAMERA_HEIGHT * 1.5)
        while not self.stopped:
            raw = self.process.stdout.read(frame_size)
            if len(raw) < frame_size:
                err = self.process.stderr.read().decode(errors='ignore').strip()
                if err:
                    print(f"  rpicam-vid error: {err}")
                self.stopped = True
                break
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                (CAMERA_HEIGHT * 3 // 2, CAMERA_WIDTH)
            )
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            with self._lock:
                self.frame = rgb

    def read(self):
        with self._lock:
            return self.frame

    def release(self):
        self.stopped = True
        self.process.terminate()
        self.process.wait()


# --- PIR SENSOR (threaded) ---

class PIRSensor:
    """Reads the PIR sensor in a background thread."""

    def __init__(self):
        self._motion = False
        self._last_motion_time = 0.0
        self._stopped = False
        self._lock = Lock()
        self._thread = Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        try:
            with gpiod.request_lines(
                GPIO_CHIP,
                consumer="pir_cam",
                config={
                    PIR_LINE: gpiod.LineSettings(
                        direction=gpiod.line.Direction.INPUT,
                        bias=gpiod.line.Bias.PULL_DOWN,
                    )
                },
            ) as req:
                while not self._stopped:
                    values = req.get_values([PIR_LINE])
                    motion = bool(values[0].value)
                    with self._lock:
                        self._motion = motion
                        if motion:
                            self._last_motion_time = time.time()
                    time.sleep(0.1)
        except Exception as e:
            print(f"  PIR sensor error: {e}")
            self._stopped = True

    @property
    def motion_detected(self):
        with self._lock:
            return self._motion

    @property
    def last_motion_time(self):
        with self._lock:
            return self._last_motion_time

    def stop(self):
        self._stopped = True


# --- BUZZER (GPIO18 via lgpio tx_pwm) ---

class Buzzer:  # TODO: Change to USB sound card and play actual sound file instead of PWM sweep
    """Drives a passive buzzer on GPIO18 using lgpio tx_pwm."""

    def __init__(self):
        self._h = None
        self._available = False
        try:
            self._h = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self._h, BUZZER_PIN)
            self._available = True
        except Exception as e:
            print(f"  Buzzer warning  : unavailable ({e})")
            if self._h is not None:
                try:
                    lgpio.gpiochip_close(self._h)
                except Exception:
                    pass
                self._h = None
        self._busy = False
        self._lock = Lock()
        if self._available:
            print("  Buzzer          : Ready (GPIO18 via lgpio)")

    def _run_sweep(self):
        if not self._available:
            with self._lock:
                self._busy = False
            return
        try:
            for freq in range(200, 400, 10):
                lgpio.tx_pwm(self._h, BUZZER_PIN, freq, 50)
                time.sleep(0.02)
            for freq in range(400, 200, -10):
                lgpio.tx_pwm(self._h, BUZZER_PIN, freq, 50)
                time.sleep(0.02)
        finally:
            lgpio.tx_pwm(self._h, BUZZER_PIN, 0, 0)
            with self._lock:
                self._busy = False

    def beep(self, duration=0.3):
        """Run one up/down buzz pattern in a background thread (non-blocking)."""
        if not self._available:
            return
        with self._lock:
            if self._busy:
                return
            self._busy = True
        Thread(target=self._run_sweep, daemon=True).start()

    def cleanup(self):
        if not self._available or self._h is None:
            return
        lgpio.tx_pwm(self._h, BUZZER_PIN, 0, 0)
        try:
            lgpio.gpiochip_close(self._h)
        except Exception:
            pass


# --- MAIN ---

def main():
    print("=" * 55)
    print("   \U0001F418  ELEPHANT DETECTION  |  PIR + Camera")
    print("=" * 55)

    setup_led()

    print("  Loading YOLO model...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print("  Model loaded.")

    # Start MJPEG web stream + dashboard
    stream_thread = Thread(target=start_stream_server, daemon=True)
    stream_thread.start()
    print(f"  Dashboard at     : http://pielephant.local:{STREAM_PORT}")

    # Start PIR sensor
    print("  Starting PIR sensor...")
    pir = PIRSensor()
    print(f"  PIR warming up ({PIR_WARMUP}s)...")
    time.sleep(PIR_WARMUP)
    print("  PIR sensor ready.")

    # Setup Buzzer
    buzzer = Buzzer()

    print("=" * 55)
    print("  System in STANDBY \u2014 waiting for motion...\n")

    running = True

    def handle_exit(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_exit)

    cam = None
    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    fps_counter = 0
    detections = []
    elephant_detected = False
    last_buzz_time = 0.0

    try:
        while running:
            now = time.time()
            last_motion_time = pir.last_motion_time
            last_motion_ago = now - last_motion_time if last_motion_time > 0 else 9999

            motion_active = last_motion_ago < PIR_COOLDOWN

            # === STANDBY MODE ===
            if not motion_active:
                if cam is not None:
                    cam.release()
                    cam = None
                    frame_count = 0
                    detections = []
                    elephant_detected = False
                    set_led(False)
                    with frame_lock:
                        global latest_frame
                        latest_frame = idle_img.copy()
                    print("\033[H\033[J", end="")
                    print("  Camera stopped \u2014 entering STANDBY\n")

                # Update shared state
                with state_lock:
                    system_state.update({
                        "mode": "standby",
                        "elephant_detected": False,
                        "detections": [],
                        "frame_count": 0,
                        "fps": 0.0,
                        "last_motion_ago": round(last_motion_ago, 1),
                        "buzzer_active": False,
                    })

                draw_console("standby", [], 0, 0, last_motion_ago)
                time.sleep(0.5)
                continue

            # === ACTIVE MODE ===
            if cam is None:
                print("\033[H\033[J", end="")
                print("  \U0001F6A8 Motion detected! Starting camera...\n")
                cam = CameraStream()
                time.sleep(2)  # Let camera initialise
                fps_timer = time.time()
                fps_counter = 0

            if cam.stopped:
                print("  Camera stream ended unexpectedly.")
                cam = None
                time.sleep(1)
                continue

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

            # YOLO inference every Nth frame
            if frame_count % INFER_EVERY == 0:
                small = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
                results = model(small, conf=CONFIDENCE, verbose=False)

                detections = []
                elephant_detected = False

                sx = CAMERA_WIDTH / YOLO_SIZE
                sy = CAMERA_HEIGHT / YOLO_SIZE

                for result in results:
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    for box in result.boxes:
                        cls_idx = int(box.cls[0]) if box.cls is not None else -1
                        class_name = result.names.get(cls_idx, "")
                        if class_name != TARGET_CLASS_NAME:
                            continue
                        elephant_detected = True
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        detections.append({
                            "confidence": conf,
                            "bbox": (
                                int(x1 * sx), int(y1 * sy),
                                int(x2 * sx), int(y2 * sy),
                            ),
                        })

            # LED
            set_led(elephant_detected)

            # Buzzer
            if elephant_detected:
                now_t = time.time()
                if now_t - last_buzz_time >= BUZZER_INTERVAL:
                    buzzer.beep(BUZZER_DURATION)
                    last_buzz_time = now_t

            # Update shared dashboard state
            buzz_just_fired = elephant_detected and (time.time() - last_buzz_time < 1.0)
            with state_lock:
                system_state.update({
                    "mode": "active",
                    "elephant_detected": elephant_detected,
                    "detections": [
                        {"confidence": det["confidence"], "bbox": list(det["bbox"])}
                        for det in detections
                    ],
                    "frame_count": frame_count,
                    "fps": round(fps, 1),
                    "last_motion_ago": round(last_motion_ago, 1),
                    "buzzer_active": buzz_just_fired,
                })

            # Console update on inference frames
            if frame_count % INFER_EVERY == 0:
                draw_console("active", detections, frame_count, fps, last_motion_ago)

            # Update MJPEG frame
            display_frame = draw_on_frame(frame, detections, fps)
            display_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            with frame_lock:
                latest_frame = display_bgr

            time.sleep(0.02)

    finally:
        if cam is not None:
            cam.release()
        pir.stop()
        buzzer.cleanup()
        set_led(False)
        restore_led()
        print("\n  Shutting down. Camera, PIR, Buzzer, and LED released.")


if __name__ == "__main__":
    main()