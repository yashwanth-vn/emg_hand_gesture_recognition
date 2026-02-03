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
window.onload = function () {
    // Remove loader
    setTimeout(() => {
        const loader = document.getElementById('loader');
        if (loader) loader.style.display = 'none';
    }, 800);

    canvas = document.getElementById('waveform');
    if (canvas) {
        ctx = canvas.getContext('2d');
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        generateChannelBars();
        generateLegend();
        checkStatus();
        animateWaveform();
    }
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
                <span class="channel-label">CH${i + 1}</span>
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
                CH${i + 1}
            </div>
        `;
    }
}

window.showTab = function (tab) {
    document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('active', i === (tab === 'detect' ? 0 : 1)));
    document.getElementById('detect-panel').classList.toggle('hidden', tab !== 'detect');
    document.getElementById('upload-panel').classList.toggle('hidden', tab !== 'upload');
    if (tab === 'upload' && autoTimer) toggleAuto();
    if (tab === 'detect' && playTimer) stopPlayback();
}

window.detectRandom = async function () {
    try {
        const res = await fetch('/api/detect_random', { method: 'POST' });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        updateUI(data);
    } catch (e) {
        document.getElementById('status').textContent = 'Error: ' + e.message;
    }
}

window.toggleAuto = function () {
    const btn = document.getElementById('btn-auto');
    if (autoTimer) {
        clearInterval(autoTimer);
        autoTimer = null;
        btn.innerHTML = '▶ Start Auto-Detect';
    } else {
        autoTimer = setInterval(window.detectRandom, 1000);
        btn.innerHTML = '⏹ Stop Auto-Detect';
    }
}

window.loadFile = function (input) {
    const file = input.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
        csvData = e.target.result.trim().split('\n')
            .map(line => line.split(',').map(v => parseFloat(v.trim())))
            .filter(row => row.length >= 8 && !isNaN(row[0]));
        document.getElementById('file-status').innerHTML = '<span style="color:#00f5d4">✓ ' + csvData.length + ' samples loaded</span>';
        document.getElementById('playback').classList.remove('hidden');
        document.getElementById('prog-text').textContent = '0 / ' + csvData.length;
    };
    reader.readAsText(file);
}

window.startPlayback = async function () {
    if (!csvData.length) return;
    csvIdx = 0;
    document.getElementById('btn-play').disabled = true;
    document.getElementById('btn-stop').disabled = false;

    playTimer = setInterval(async () => {
        if (csvIdx >= csvData.length) { window.stopPlayback(); return; }
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
        } catch (e) { }
        csvIdx++;
        document.getElementById('prog-bar').style.width = (csvIdx / csvData.length * 100) + '%';
        document.getElementById('prog-text').textContent = csvIdx + ' / ' + csvData.length;
    }, 1000);
}

window.stopPlayback = function () {
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
        if (bar) {
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
