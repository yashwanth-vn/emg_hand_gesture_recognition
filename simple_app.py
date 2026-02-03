"""
EMG Gesture Recognition - Premium UI Version
Features:
- Gesture-adaptive EMG waveforms
- 8-channel activity bars
- Virtual bulb with smooth transitions
- Confidence ring with color gradient
- Medical-tech dark theme
"""
import os
import sys
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

app = Flask(__name__)

models = {}
label_encoder = None
norm_stats = None

def load_models():
    global models, label_encoder, norm_stats
    try:
        models['rf'] = joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib'))
        models['hgb'] = joblib.load(os.path.join(MODELS_DIR, 'hist_gradient_boosting.joblib'))
        models['lr'] = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.joblib'))
        label_encoder = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))
        norm_stats = joblib.load(os.path.join(MODELS_DIR, 'normalization_stats.joblib'))
        print("[OK] Models loaded")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return False

def extract_features(emg_data):
    emg = np.array(emg_data, dtype=np.float64)
    mav = np.abs(emg)
    rms = emg ** 2
    return np.concatenate([mav, rms])

def normalize(features):
    if norm_stats is None:
        return features
    mean = norm_stats.get('mean', np.zeros(16))
    std = norm_stats.get('std', np.ones(16))
    std[std == 0] = 1
    return (features - mean) / std

def predict(emg_data):
    import time
    start = time.time()
    
    features = extract_features(emg_data)
    features_norm = normalize(features).reshape(1, -1)
    
    votes = {}
    confidences = {}
    
    for name, model in models.items():
        pred = model.predict(features_norm)[0]
        prob = model.predict_proba(features_norm)[0]
        gesture = label_encoder.inverse_transform([pred])[0]
        votes[name] = gesture
        confidences[name] = float(np.max(prob))
    
    gesture_counts = {}
    for g in votes.values():
        gesture_counts[g] = gesture_counts.get(g, 0) + 1
    
    final_gesture = max(gesture_counts, key=gesture_counts.get)
    avg_confidence = np.mean(list(confidences.values()))
    
    actions = {'fist': 'ON', 'open': 'OFF', 'pinch': 'TOGGLE', 'rest': 'IDLE'}
    
    latency = (time.time() - start) * 1000
    
    return {
        'gesture': final_gesture,
        'confidence': round(avg_confidence, 4),
        'action': actions.get(final_gesture, 'IDLE'),
        'model_votes': {
            'random_forest': votes.get('rf'),
            'hist_gradient_boosting': votes.get('hgb'),
            'logistic_regression': votes.get('lr')
        },
        'latency': {
            'total_ms': round(latency, 2),
            'inference_ms': round(latency, 2),
            'preprocessing_ms': 0.1
        }
    }

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>EMG Neural Interface</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #050508;
            --bg-surface: #0c0c12;
            --bg-card: #13131c;
            --bg-elevated: #1a1a26;
            --primary: #00f5d4;
            --secondary: #7b2cbf;
            --accent: #00d4aa;
            --success: #00ff88;
            --warning: #ffcc00;
            --danger: #ff4466;
            --text: #ffffff;
            --text-dim: #6a6a80;
            --text-muted: #3a3a50;
            --glow-primary: rgba(0, 245, 212, 0.4);
            --glow-warning: rgba(255, 200, 0, 0.6);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Background Effects */
        .bg-grid {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: 
                linear-gradient(rgba(0,245,212,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,245,212,0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: 0;
        }
        
        .bg-glow {
            position: fixed;
            top: -200px;
            left: 50%;
            transform: translateX(-50%);
            width: 800px;
            height: 600px;
            background: radial-gradient(ellipse, rgba(0,245,212,0.08) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }
        
        .container {
            position: relative;
            z-index: 1;
            max-width: 1500px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header */
        header {
            text-align: center;
            padding: 25px 0 35px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 10px;
        }
        
        .logo-icon {
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            box-shadow: 0 0 30px var(--glow-primary);
        }
        
        h1 {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            color: var(--text-dim);
            font-size: 13px;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        
        /* Main Layout */
        .main-grid {
            display: grid;
            grid-template-columns: 300px 1fr 300px;
            gap: 24px;
        }
        
        /* Panels */
        .panel {
            background: var(--bg-surface);
            border-radius: 20px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.04);
            box-shadow: 0 4px 30px rgba(0,0,0,0.3);
        }
        
        .panel-title {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-dim);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .panel-title::before {
            content: '';
            width: 4px;
            height: 14px;
            background: linear-gradient(to bottom, var(--primary), var(--secondary));
            border-radius: 2px;
        }
        
        /* Buttons */
        .btn {
            width: 100%;
            padding: 16px 24px;
            margin-bottom: 12px;
            border: none;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            position: relative;
            overflow: hidden;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: #000;
        }
        
        .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 40px var(--glow-primary);
        }
        
        .btn-secondary {
            background: var(--bg-card);
            color: var(--text);
            border: 1px solid rgba(255,255,255,0.08);
        }
        
        .btn-secondary:hover {
            background: var(--bg-elevated);
            border-color: rgba(0,245,212,0.3);
        }
        
        .btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none !important;
        }
        
        /* Center Section */
        .center-section {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        
        /* Bulb Section */
        .bulb-panel {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 80px;
            padding: 40px;
            min-height: 320px;
        }
        
        .gesture-display {
            text-align: center;
        }
        
        .gesture-icon-container {
            width: 120px;
            height: 120px;
            background: var(--bg-card);
            border-radius: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            font-size: 60px;
            transition: all 0.4s ease;
            border: 2px solid transparent;
        }
        
        .gesture-icon-container.active {
            border-color: var(--primary);
            box-shadow: 0 0 40px var(--glow-primary);
        }
        
        .gesture-name {
            font-size: 32px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 4px;
            margin-bottom: 8px;
            transition: all 0.3s ease;
        }
        
        .gesture-name.rest { color: var(--text-dim); }
        .gesture-name.fist { color: var(--danger); }
        .gesture-name.open { color: var(--primary); }
        .gesture-name.pinch { color: var(--warning); }
        .gesture-name.uncertain { color: var(--text-muted); }
        
        .action-badge {
            display: inline-block;
            padding: 8px 20px;
            background: var(--bg-card);
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 2px;
            color: var(--text-dim);
            border: 1px solid rgba(255,255,255,0.05);
        }
        
        /* Virtual Bulb */
        .bulb-wrapper {
            text-align: center;
        }
        
        .confidence-ring-container {
            position: relative;
            width: 200px;
            height: 200px;
        }
        
        .confidence-ring {
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
        }
        
        .confidence-ring svg {
            transform: rotate(-90deg);
        }
        
        .ring-bg {
            fill: none;
            stroke: var(--bg-card);
            stroke-width: 8;
        }
        
        .ring-fg {
            fill: none;
            stroke-width: 8;
            stroke-linecap: round;
            stroke-dasharray: 565;
            stroke-dashoffset: 565;
            transition: stroke-dashoffset 0.5s ease, stroke 0.3s ease;
        }
        
        .bulb-container {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }
        
        .bulb {
            width: 100px;
            height: 100px;
            background: var(--bg-elevated);
            border-radius: 50% 50% 45% 45%;
            position: relative;
            transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .bulb::before {
            content: '';
            position: absolute;
            top: 10%; left: 20%;
            width: 25%;
            height: 35%;
            background: rgba(255,255,255,0.15);
            border-radius: 50%;
            transition: all 0.5s ease;
        }
        
        .bulb.on {
            background: linear-gradient(135deg, #fffbe6 0%, #ffe566 30%, #ffcc00 60%, #ffaa00 100%);
            box-shadow: 
                0 0 40px rgba(255, 200, 0, 0.8),
                0 0 80px rgba(255, 180, 0, 0.5),
                0 0 120px rgba(255, 150, 0, 0.3),
                0 0 160px rgba(255, 100, 0, 0.15),
                inset 0 -10px 30px rgba(255, 150, 0, 0.3);
        }
        
        .bulb.on::before {
            background: rgba(255,255,255,0.6);
        }
        
        .bulb.flickering {
            animation: flicker 0.3s ease-in-out;
        }
        
        @keyframes flicker {
            0%, 100% { opacity: 1; }
            25% { opacity: 0.4; }
            50% { opacity: 0.8; }
            75% { opacity: 0.3; }
        }
        
        .bulb.uncertain-glow {
            box-shadow: 0 0 30px rgba(255, 100, 100, 0.3);
        }
        
        .bulb-base {
            width: 45px;
            height: 20px;
            background: linear-gradient(to bottom, #666, #444);
            margin: -2px auto 0;
            border-radius: 0 0 10px 10px;
        }
        
        .bulb-status {
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 4px;
            margin-top: 15px;
            transition: all 0.3s ease;
        }
        
        .bulb-status.on {
            color: var(--warning);
            text-shadow: 0 0 20px var(--glow-warning);
        }
        
        .bulb-status.off { color: var(--text-dim); }
        
        /* Channel Activity Bars */
        .channel-bars-section {
            padding: 20px;
        }
        
        .channel-bars-container {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            height: 120px;
            gap: 8px;
            padding: 0 10px;
        }
        
        .channel-bar-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        
        .channel-bar-track {
            width: 100%;
            height: 100px;
            background: var(--bg-card);
            border-radius: 6px;
            position: relative;
            overflow: hidden;
        }
        
        .channel-bar-fill {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 0%;
            border-radius: 6px;
            transition: height 0.15s ease-out, background 0.3s ease;
        }
        
        .channel-bar-fill.dominant {
            box-shadow: 0 0 15px currentColor;
        }
        
        .channel-label {
            font-size: 10px;
            color: var(--text-dim);
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Waveform Canvas */
        .waveform-section {
            padding: 20px;
        }
        
        .waveform-canvas {
            width: 100%;
            height: 180px;
            background: var(--bg-dark);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.03);
        }
        
        .wave-legend {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 15px;
        }
        
        .wave-legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: var(--text-dim);
        }
        
        .wave-legend-color {
            width: 12px;
            height: 3px;
            border-radius: 2px;
        }
        
        /* Metrics Panel */
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 24px;
        }
        
        .metric-card {
            background: var(--bg-card);
            padding: 18px;
            border-radius: 14px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.03);
        }
        
        .metric-value {
            font-size: 26px;
            font-weight: 700;
            color: var(--primary);
            font-family: 'JetBrains Mono', monospace;
        }
        
        .metric-label {
            font-size: 10px;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 6px;
        }
        
        .vote-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 16px;
            background: var(--bg-card);
            border-radius: 10px;
            margin-bottom: 10px;
            border: 1px solid rgba(255,255,255,0.03);
        }
        
        .vote-name {
            font-size: 12px;
            color: var(--text-dim);
        }
        
        .vote-value {
            font-size: 13px;
            font-weight: 600;
            color: var(--primary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Upload Section */
        .upload-zone {
            border: 2px dashed rgba(255,255,255,0.1);
            border-radius: 14px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 16px;
        }
        
        .upload-zone:hover {
            border-color: var(--primary);
            background: rgba(0,245,212,0.03);
        }
        
        .upload-zone p {
            color: var(--text-dim);
            font-size: 14px;
        }
        
        .progress-bar {
            height: 6px;
            background: var(--bg-card);
            border-radius: 3px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            width: 0%;
            transition: width 0.15s ease;
        }
        
        .progress-text {
            text-align: center;
            font-size: 12px;
            color: var(--text-dim);
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Status */
        #status {
            margin-top: 20px;
            padding: 12px;
            background: var(--bg-card);
            border-radius: 10px;
            font-size: 12px;
            color: var(--text-dim);
            text-align: center;
            border: 1px solid rgba(255,255,255,0.03);
        }
        
        .hidden { display: none !important; }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
        }
        
        .tab {
            flex: 1;
            padding: 12px;
            background: var(--bg-card);
            border-radius: 10px;
            text-align: center;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }
        
        .tab.active {
            background: rgba(0,245,212,0.1);
            border-color: var(--primary);
            color: var(--primary);
        }
        
        .tab:hover:not(.active) {
            background: var(--bg-elevated);
        }
        
        @media (max-width: 1200px) {
            .main-grid { grid-template-columns: 1fr; }
            .bulb-panel { flex-direction: column; gap: 40px; }
        }
    </style>
</head>
<body>
    <div class="bg-grid"></div>
    <div class="bg-glow"></div>
    
    <div class="container">
        <header>
            <div class="logo">
                <div class="logo-icon">🧠</div>
                <h1>EMG Neural Interface</h1>
            </div>
            <p class="subtitle">Real-Time Gesture Recognition • Virtual Control System</p>
        </header>
        
        <div class="main-grid">
            <!-- Left Panel -->
            <div class="panel">
                <div class="tabs">
                    <div class="tab active" onclick="showTab('detect')">⚡ Detect</div>
                    <div class="tab" onclick="showTab('upload')">📁 CSV</div>
                </div>
                
                <div id="detect-panel">
                    <p class="panel-title">Signal Detection</p>
                    <button class="btn btn-primary" onclick="detectRandom()">
                        🎲 Detect Gesture
                    </button>
                    <button class="btn btn-secondary" id="btn-auto" onclick="toggleAuto()">
                        ▶ Start Auto-Detect
                    </button>
                </div>
                
                <div id="upload-panel" class="hidden">
                    <p class="panel-title">CSV Playback</p>
                    <div class="upload-zone" onclick="document.getElementById('file').click()">
                        <p>📊 Upload CSV File</p>
                        <p style="font-size:11px; color:#444; margin-top:8px;">8 EMG channels per row</p>
                    </div>
                    <input type="file" id="file" accept=".csv" style="display:none" onchange="loadFile(this)">
                    <div id="file-status"></div>
                    <div id="playback" class="hidden">
                        <button class="btn btn-primary" id="btn-play" onclick="startPlayback()">▶ Play</button>
                        <button class="btn btn-secondary" id="btn-stop" onclick="stopPlayback()" disabled>⏹ Stop</button>
                        <div class="progress-bar"><div class="progress-fill" id="prog-bar"></div></div>
                        <div class="progress-text" id="prog-text">0 / 0</div>
                    </div>
                </div>
                
                <div id="status">Ready</div>
            </div>
            
            <!-- Center Section -->
            <div class="center-section">
                <!-- Bulb + Gesture Panel -->
                <div class="panel bulb-panel">
                    <div class="gesture-display">
                        <div class="gesture-icon-container" id="gesture-container">
                            <span id="icon">✋</span>
                        </div>
                        <div class="gesture-name rest" id="name">REST</div>
                        <div class="action-badge" id="action">IDLE</div>
                    </div>
                    
                    <div class="bulb-wrapper">
                        <p class="panel-title" style="justify-content: center;">Virtual Bulb Control</p>
                        <div class="confidence-ring-container">
                            <div class="confidence-ring">
                                <svg width="200" height="200">
                                    <defs>
                                        <linearGradient id="confGradLow" x1="0%" y1="0%" x2="100%" y2="0%">
                                            <stop offset="0%" style="stop-color:#ff4466"/>
                                            <stop offset="100%" style="stop-color:#ff6644"/>
                                        </linearGradient>
                                        <linearGradient id="confGradMed" x1="0%" y1="0%" x2="100%" y2="0%">
                                            <stop offset="0%" style="stop-color:#ffaa00"/>
                                            <stop offset="100%" style="stop-color:#ffcc00"/>
                                        </linearGradient>
                                        <linearGradient id="confGradHigh" x1="0%" y1="0%" x2="100%" y2="0%">
                                            <stop offset="0%" style="stop-color:#00cc66"/>
                                            <stop offset="100%" style="stop-color:#00ff88"/>
                                        </linearGradient>
                                    </defs>
                                    <circle class="ring-bg" cx="100" cy="100" r="90"/>
                                    <circle class="ring-fg" id="conf-ring" cx="100" cy="100" r="90"/>
                                </svg>
                            </div>
                            <div class="bulb-container">
                                <div class="bulb" id="bulb"></div>
                                <div class="bulb-base"></div>
                            </div>
                        </div>
                        <div class="bulb-status off" id="bulb-status">OFF</div>
                    </div>
                </div>
                
                <!-- Channel Activity Bars -->
                <div class="panel channel-bars-section">
                    <p class="panel-title">Channel Activity</p>
                    <div class="channel-bars-container" id="channel-bars">
                        <!-- Generated by JS -->
                    </div>
                </div>
                
                <!-- EMG Waveform -->
                <div class="panel waveform-section">
                    <p class="panel-title">EMG Waveform</p>
                    <canvas class="waveform-canvas" id="waveform"></canvas>
                    <div class="wave-legend" id="legend"></div>
                </div>
            </div>
            
            <!-- Right Panel -->
            <div class="panel">
                <p class="panel-title">Performance</p>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="lat">--</div>
                        <div class="metric-label">Latency ms</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="conf-text">--</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                </div>
                
                <p class="panel-title">Model Consensus</p>
                <div class="vote-card">
                    <span class="vote-name">Random Forest</span>
                    <span class="vote-value" id="v-rf">--</span>
                </div>
                <div class="vote-card">
                    <span class="vote-name">Gradient Boost</span>
                    <span class="vote-value" id="v-hgb">--</span>
                </div>
                <div class="vote-card">
                    <span class="vote-name">Logistic Regression</span>
                    <span class="vote-value" id="v-lr">--</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Constants
        const icons = { rest: '✋', fist: '✊', open: '🖐️', pinch: '🤏', uncertain: '❓' };
        const channelColors = ['#00f5d4', '#7b2cbf', '#ff6b6b', '#ffc107', '#17a2b8', '#00ff88', '#ff4466', '#00aaff'];
        
        // State
        let autoTimer = null;
        let csvData = [];
        let csvIdx = 0;
        let playTimer = null;
        let bulbState = false;
        let emgHistory = [];
        const maxHistory = 120;
        let canvas, ctx;
        let currentGesture = 'rest';
        let targetWaveParams = { amplitude: 0.1, frequency: 1, noise: 0.02 };
        let currentWaveParams = { amplitude: 0.1, frequency: 1, noise: 0.02 };
        let animationFrame;
        
        // Wave parameters per gesture
        const gestureWaveParams = {
            rest: { amplitude: 0.08, frequency: 0.5, noise: 0.02 },
            fist: { amplitude: 0.9, frequency: 3, noise: 0.15 },
            open: { amplitude: 0.25, frequency: 1, noise: 0.03 },
            pinch: { amplitude: 0.6, frequency: 5, noise: 0.4 },
            uncertain: { amplitude: 0.3, frequency: 2, noise: 0.25 }
        };
        
        // Initialize
        window.onload = function() {
            canvas = document.getElementById('waveform');
            ctx = canvas.getContext('2d');
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);
            generateChannelBars();
            generateLegend();
            checkStatus();
            animateWaveform();
        };
        
        function resizeCanvas() {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
        }
        
        function generateChannelBars() {
            const container = document.getElementById('channel-bars');
            container.innerHTML = '';
            for (let i = 0; i < 8; i++) {
                container.innerHTML += `
                    <div class="channel-bar-wrapper">
                        <div class="channel-bar-track">
                            <div class="channel-bar-fill" id="bar-${i}" style="background: ${channelColors[i]}"></div>
                        </div>
                        <span class="channel-label">CH${i+1}</span>
                    </div>
                `;
            }
        }
        
        function generateLegend() {
            const legend = document.getElementById('legend');
            legend.innerHTML = '';
            for (let i = 0; i < 8; i++) {
                legend.innerHTML += `
                    <div class="wave-legend-item">
                        <div class="wave-legend-color" style="background: ${channelColors[i]}"></div>
                        CH${i+1}
                    </div>
                `;
            }
        }
        
        function showTab(tab) {
            document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('active', i === (tab === 'detect' ? 0 : 1)));
            document.getElementById('detect-panel').classList.toggle('hidden', tab !== 'detect');
            document.getElementById('upload-panel').classList.toggle('hidden', tab !== 'upload');
            if (tab === 'upload' && autoTimer) toggleAuto();
            if (tab === 'detect' && playTimer) stopPlayback();
        }
        
        async function detectRandom() {
            try {
                const res = await fetch('/api/detect_random', { method: 'POST' });
                const data = await res.json();
                if (data.error) throw new Error(data.error);
                updateUI(data);
            } catch (e) {
                document.getElementById('status').textContent = 'Error: ' + e.message;
            }
        }
        
        function toggleAuto() {
            const btn = document.getElementById('btn-auto');
            if (autoTimer) {
                clearInterval(autoTimer);
                autoTimer = null;
                btn.innerHTML = '▶ Start Auto-Detect';
            } else {
                autoTimer = setInterval(detectRandom, 200);
                btn.innerHTML = '⏹ Stop Auto-Detect';
            }
        }
        
        function loadFile(input) {
            const file = input.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = e => {
                csvData = e.target.result.trim().split('\\n')
                    .map(line => line.split(',').map(v => parseFloat(v.trim())))
                    .filter(row => row.length >= 8 && !isNaN(row[0]));
                document.getElementById('file-status').innerHTML = '<span style="color:#00f5d4">✓ ' + csvData.length + ' samples loaded</span>';
                document.getElementById('playback').classList.remove('hidden');
                document.getElementById('prog-text').textContent = '0 / ' + csvData.length;
            };
            reader.readAsText(file);
        }
        
        async function startPlayback() {
            if (!csvData.length) return;
            csvIdx = 0;
            document.getElementById('btn-play').disabled = true;
            document.getElementById('btn-stop').disabled = false;
            
            playTimer = setInterval(async () => {
                if (csvIdx >= csvData.length) { stopPlayback(); return; }
                try {
                    const res = await fetch('/api/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ emg_data: csvData[csvIdx].slice(0, 8) })
                    });
                    const data = await res.json();
                    if (!data.error) {
                        data.emg_data = csvData[csvIdx].slice(0, 8);
                        updateUI(data);
                    }
                } catch (e) {}
                csvIdx++;
                document.getElementById('prog-bar').style.width = (csvIdx / csvData.length * 100) + '%';
                document.getElementById('prog-text').textContent = csvIdx + ' / ' + csvData.length;
            }, 150);
        }
        
        function stopPlayback() {
            if (playTimer) { clearInterval(playTimer); playTimer = null; }
            document.getElementById('btn-play').disabled = false;
            document.getElementById('btn-stop').disabled = true;
        }
        
        function updateUI(data) {
            const g = data.gesture || 'rest';
            currentGesture = g;
            
            // Update wave parameters target
            targetWaveParams = gestureWaveParams[g] || gestureWaveParams.rest;
            
            // Gesture display
            document.getElementById('icon').textContent = icons[g] || '❓';
            const nameEl = document.getElementById('name');
            nameEl.textContent = g.toUpperCase();
            nameEl.className = 'gesture-name ' + g;
            
            const container = document.getElementById('gesture-container');
            container.classList.add('active');
            setTimeout(() => container.classList.remove('active'), 300);
            
            // Action
            const action = data.action || 'IDLE';
            document.getElementById('action').textContent = action;
            
            // Confidence ring with color gradient
            const conf = data.confidence || 0;
            const confPct = Math.round(conf * 100);
            document.getElementById('conf-text').textContent = confPct + '%';
            
            const ring = document.getElementById('conf-ring');
            const circumference = 565;
            ring.style.strokeDashoffset = circumference * (1 - conf);
            
            // Color based on confidence
            if (conf < 0.5) {
                ring.style.stroke = 'url(#confGradLow)';
            } else if (conf < 0.8) {
                ring.style.stroke = 'url(#confGradMed)';
            } else {
                ring.style.stroke = 'url(#confGradHigh)';
            }
            
            // Bulb control
            updateBulb(action, g);
            
            // Latency
            if (data.latency) {
                document.getElementById('lat').textContent = data.latency.total_ms.toFixed(1);
            }
            
            // Model votes
            if (data.model_votes) {
                document.getElementById('v-rf').textContent = data.model_votes.random_forest || '--';
                document.getElementById('v-hgb').textContent = data.model_votes.hist_gradient_boosting || '--';
                document.getElementById('v-lr').textContent = data.model_votes.logistic_regression || '--';
            }
            
            // Channel bars
            if (data.emg_data) {
                updateChannelBars(data.emg_data);
                emgHistory.push(data.emg_data);
                if (emgHistory.length > maxHistory) emgHistory.shift();
            }
        }
        
        function updateChannelBars(emg) {
            const max = Math.max(...emg);
            for (let i = 0; i < 8; i++) {
                const bar = document.getElementById('bar-' + i);
                const height = Math.min(100, emg[i] * 100);
                bar.style.height = height + '%';
                
                // Highlight dominant channels
                if (emg[i] === max && emg[i] > 0.3) {
                    bar.classList.add('dominant');
                } else {
                    bar.classList.remove('dominant');
                }
            }
        }
        
        function updateBulb(action, gesture) {
            const bulb = document.getElementById('bulb');
            const status = document.getElementById('bulb-status');
            
            // Remove previous states
            bulb.classList.remove('flickering', 'uncertain-glow');
            
            if (action === 'ON') {
                bulbState = true;
            } else if (action === 'OFF') {
                bulbState = false;
            } else if (action === 'TOGGLE') {
                bulbState = !bulbState;
                bulb.classList.add('flickering');
                setTimeout(() => bulb.classList.remove('flickering'), 300);
            }
            
            if (gesture === 'uncertain') {
                bulb.classList.add('uncertain-glow');
            }
            
            bulb.classList.toggle('on', bulbState);
            status.textContent = bulbState ? 'ON' : 'OFF';
            status.className = 'bulb-status ' + (bulbState ? 'on' : 'off');
        }
        
        function animateWaveform() {
            // Smooth interpolation of wave parameters
            currentWaveParams.amplitude += (targetWaveParams.amplitude - currentWaveParams.amplitude) * 0.1;
            currentWaveParams.frequency += (targetWaveParams.frequency - currentWaveParams.frequency) * 0.1;
            currentWaveParams.noise += (targetWaveParams.noise - currentWaveParams.noise) * 0.1;
            
            drawWaveform();
            animationFrame = requestAnimationFrame(animateWaveform);
        }
        
        function drawWaveform() {
            if (!ctx) return;
            const w = canvas.width;
            const h = canvas.height;
            const numChannels = 8;
            const channelHeight = h / numChannels;
            const time = Date.now() / 1000;
            
            // Clear with fade effect
            ctx.fillStyle = 'rgba(5, 5, 8, 0.3)';
            ctx.fillRect(0, 0, w, h);
            
            // Draw channels
            for (let ch = 0; ch < numChannels; ch++) {
                const yBase = ch * channelHeight + channelHeight / 2;
                
                // Grid line
                ctx.strokeStyle = 'rgba(255,255,255,0.03)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(0, yBase);
                ctx.lineTo(w, yBase);
                ctx.stroke();
                
                // Draw signal based on history + gesture morphing
                ctx.strokeStyle = channelColors[ch];
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                
                // Use history if available, otherwise generate based on gesture
                if (emgHistory.length > 1) {
                    const sampleWidth = w / maxHistory;
                    for (let i = 0; i < emgHistory.length; i++) {
                        const x = i * sampleWidth;
                        const val = emgHistory[i][ch] || 0;
                        const y = yBase - (val - 0.5) * channelHeight * 0.8;
                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }
                } else {
                    // Generate wave based on gesture
                    for (let x = 0; x < w; x++) {
                        const t = (x / w) * 10 + time * 2;
                        const amp = currentWaveParams.amplitude;
                        const freq = currentWaveParams.frequency;
                        const noise = currentWaveParams.noise;
                        
                        let val = Math.sin(t * freq + ch * 0.5) * amp;
                        val += (Math.random() - 0.5) * noise;
                        
                        const y = yBase - val * channelHeight * 0.7;
                        if (x === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            }
        }
        
        function checkStatus() {
            fetch('/api/status').then(r => r.json()).then(s => {
                document.getElementById('status').textContent = s.models_loaded ? '✓ Models loaded' : '✗ Models not loaded';
            }).catch(() => {
                document.getElementById('status').textContent = '✗ Cannot connect';
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def status():
    return jsonify({
        'models_loaded': len(models) > 0,
        'num_models': len(models)
    })

@app.route('/api/detect_random', methods=['POST'])
def detect_random():
    try:
        pattern = np.random.choice(['low', 'high', 'mixed', 'random'])
        if pattern == 'low':
            emg = np.random.uniform(0.0, 0.15, 8)
        elif pattern == 'high':
            emg = np.random.uniform(0.4, 0.9, 8)
        elif pattern == 'mixed':
            emg = np.array([np.random.uniform(0.5, 0.9) if np.random.random() > 0.5 
                           else np.random.uniform(0.0, 0.2) for _ in range(8)])
        else:
            emg = np.random.uniform(0.0, 1.0, 8)
        
        result = predict(emg)
        result['emg_data'] = emg.tolist()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        if not data or 'emg_data' not in data:
            return jsonify({'error': 'emg_data required'}), 400
        
        emg = data['emg_data']
        if len(emg) < 8:
            return jsonify({'error': 'Need 8 channels'}), 400
        
        result = predict(emg[:8])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\\n" + "="*60)
    print("  EMG NEURAL INTERFACE - PREMIUM UI")
    print("="*60 + "\\n")
    
    if load_models():
        print("Starting server at http://127.0.0.1:5001\\n")
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        print("ERROR: Could not load models!")
