/**
 * DSP Visualization Renderer
 * 
 * Draws 4 interactive charts on HTML5 Canvas elements.
 * (The spectrogram is a pre-rendered image, so no Canvas needed for it.)
 * 
 * Each function follows the same pattern:
 * 1. Get canvas and 2D drawing context
 * 2. Set canvas dimensions to match container
 * 3. Draw background, grid, and axes
 * 4. Plot the data
 * 5. Add labels and annotations
 */

// Color palette matching the website theme
const COLORS = {
    bg: '#0f0a1a',
    grid: 'rgba(167, 139, 250, 0.08)',
    axis: 'rgba(167, 139, 250, 0.3)',
    text: '#a8a3b8',
    textBright: '#c4b5fd',
    purple: '#a78bfa',
    pink: '#f472b6',
    cyan: '#67e8f9',
    green: '#4ade80',
    yellow: '#facc15',
    gradientStart: 'rgba(167, 139, 250, 0.6)',
    gradientEnd: 'rgba(244, 114, 182, 0.1)',
};

/**
 * Set up canvas dimensions for high-DPI (Retina) displays.
 * Without this, Canvas looks blurry on modern screens.
 */
function setupCanvas(canvasId) {
    const canvas = document.getElementById(canvasId);
    const container = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    // devicePixelRatio = 2 on Retina displays, 1 on standard

    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = 250 * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = '250px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    // Scale all drawing operations so 1 unit = 1 CSS pixel

    return { canvas, ctx, width: rect.width, height: 250 };
}

/**
 * Draw grid lines and axis labels.
 */
function drawGrid(ctx, width, height, padding, xLabels, yLabels) {
    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.font = '10px Poppins, sans-serif';
    ctx.fillStyle = COLORS.text;

    // Horizontal grid lines
    const hLines = 5;
    for (let i = 0; i <= hLines; i++) {
        const y = padding.top + (plotH / hLines) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();

        if (yLabels && yLabels[i] !== undefined) {
            ctx.textAlign = 'right';
            ctx.fillText(yLabels[i], padding.left - 8, y + 3);
        }
    }

    // Vertical grid lines
    const vLines = Math.min(xLabels ? xLabels.length - 1 : 5, 8);
    for (let i = 0; i <= vLines; i++) {
        const x = padding.left + (plotW / vLines) * i;
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, height - padding.bottom);
        ctx.stroke();

        if (xLabels && xLabels[i] !== undefined) {
            ctx.textAlign = 'center';
            ctx.fillText(xLabels[i], x, height - padding.bottom + 15);
        }
    }
}


// ============================================
// 1. WAVEFORM DRAWING
// ============================================
function drawWaveform(canvasId, data) {
    const { ctx, width, height } = setupCanvas(canvasId);
    const padding = { top: 20, right: 20, bottom: 30, left: 50 };
    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    // Clear background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, width, height);

    // Generate axis labels
    const maxTime = data.times[data.times.length - 1] || 1;
    const xLabels = [];
    for (let i = 0; i <= 5; i++) {
        xLabels.push((maxTime * i / 5).toFixed(1) + 's');
    }
    const yLabels = ['1.0', '0.5', '0.0', '-0.5', '-1.0', ''];

    drawGrid(ctx, width, height, padding, xLabels, yLabels);

    // Draw center line (zero amplitude)
    const centerY = padding.top + plotH / 2;
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padding.left, centerY);
    ctx.lineTo(width - padding.right, centerY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw waveform as filled area
    const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    gradient.addColorStop(0, COLORS.gradientStart);
    gradient.addColorStop(0.5, 'rgba(167, 139, 250, 0.02)');
    gradient.addColorStop(1, COLORS.gradientStart);

    ctx.beginPath();
    ctx.moveTo(padding.left, centerY);

    for (let i = 0; i < data.times.length; i++) {
        const x = padding.left + (data.times[i] / maxTime) * plotW;
        const y = centerY - (data.amplitudes[i] * plotH / 2);
        ctx.lineTo(x, y);
    }

    ctx.lineTo(padding.left + plotW, centerY);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw waveform line on top
    ctx.beginPath();
    for (let i = 0; i < data.times.length; i++) {
        const x = padding.left + (data.times[i] / maxTime) * plotW;
        const y = centerY - (data.amplitudes[i] * plotH / 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = COLORS.purple;
    ctx.lineWidth = 1.5;
    ctx.stroke();
}


// ============================================
// 2. FFT SPECTRUM DRAWING
// ============================================
function drawFFT(canvasId, data) {
    const { ctx, width, height } = setupCanvas(canvasId);
    const padding = { top: 20, right: 20, bottom: 30, left: 50 };
    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, width, height);

    const maxFreq = data.frequencies[data.frequencies.length - 1] || 4000;
    const minDb = Math.max(-80, Math.min(...data.magnitudes));
    const maxDb = 0;  // Normalized to 0 dB

    // Axis labels
    const xLabels = [];
    for (let i = 0; i <= 4; i++) {
        xLabels.push(Math.round(maxFreq * i / 4) + ' Hz');
    }
    const yLabels = [];
    for (let i = 0; i <= 5; i++) {
        yLabels.push(Math.round(maxDb - (maxDb - minDb) * i / 5) + ' dB');
    }

    drawGrid(ctx, width, height, padding, xLabels, yLabels);

    // Draw FFT bars
    const barWidth = Math.max(1, plotW / data.frequencies.length);

    for (let i = 0; i < data.frequencies.length; i++) {
        const x = padding.left + (data.frequencies[i] / maxFreq) * plotW;
        const dbNorm = (data.magnitudes[i] - minDb) / (maxDb - minDb);
        const barH = Math.max(0, dbNorm * plotH);
        const y = padding.top + plotH - barH;

        // Color based on magnitude
        const hue = 260 + dbNorm * 60;  // Purple to pink
        ctx.fillStyle = `hsla(${hue}, 80%, 65%, ${0.3 + dbNorm * 0.7})`;
        ctx.fillRect(x, y, barWidth + 0.5, barH);
    }

    // Mark peak frequency
    if (data.peak_freq > 0) {
        const peakX = padding.left + (data.peak_freq / maxFreq) * plotW;

        // Draw vertical line at peak
        ctx.strokeStyle = COLORS.green;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(peakX, padding.top);
        ctx.lineTo(peakX, padding.top + plotH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label
        ctx.fillStyle = COLORS.green;
        ctx.font = 'bold 11px Poppins, sans-serif';
        ctx.textAlign = 'center';
        const label = data.peak_note
            ? `${data.peak_note} (${data.peak_freq} Hz)`
            : `${data.peak_freq} Hz`;
        ctx.fillText(label, peakX, padding.top - 5);
    }

    // Mark harmonics
    if (data.harmonics) {
        data.harmonics.forEach((h, idx) => {
            const hx = padding.left + (h / maxFreq) * plotW;
            if (hx < width - padding.right) {
                ctx.strokeStyle = 'rgba(250, 204, 21, 0.5)';
                ctx.lineWidth = 1;
                ctx.setLineDash([2, 3]);
                ctx.beginPath();
                ctx.moveTo(hx, padding.top);
                ctx.lineTo(hx, padding.top + plotH);
                ctx.stroke();
                ctx.setLineDash([]);

                ctx.fillStyle = COLORS.yellow;
                ctx.font = '9px Poppins, sans-serif';
                ctx.fillText(`H${idx + 2}`, hx, padding.top + plotH + 12);
            }
        });
    }
}


// ============================================
// 3. RMS ENERGY DRAWING
// ============================================
function drawRMS(canvasId, data) {
    const { ctx, width, height } = setupCanvas(canvasId);
    const padding = { top: 20, right: 20, bottom: 30, left: 50 };
    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, width, height);

    const maxTime = data.times[data.times.length - 1] || 1;
    const xLabels = [];
    for (let i = 0; i <= 5; i++) {
        xLabels.push((maxTime * i / 5).toFixed(1) + 's');
    }
    const yLabels = ['1.0', '0.75', '0.5', '0.25', '0.0', ''];

    drawGrid(ctx, width, height, padding, xLabels, yLabels);

    // Draw filled area
    const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    gradient.addColorStop(0, 'rgba(244, 114, 182, 0.5)');
    gradient.addColorStop(1, 'rgba(244, 114, 182, 0.02)');

    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + plotH);

    for (let i = 0; i < data.times.length; i++) {
        const x = padding.left + (data.times[i] / maxTime) * plotW;
        const y = padding.top + plotH - (data.rms[i] * plotH);
        ctx.lineTo(x, y);
    }

    ctx.lineTo(padding.left + plotW, padding.top + plotH);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw line
    ctx.beginPath();
    for (let i = 0; i < data.times.length; i++) {
        const x = padding.left + (data.times[i] / maxTime) * plotW;
        const y = padding.top + plotH - (data.rms[i] * plotH);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = COLORS.pink;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw silence threshold line
    const thresholdY = padding.top + plotH - (0.05 * plotH);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padding.left, thresholdY);
    ctx.lineTo(width - padding.right, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.font = '9px Poppins, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Silence threshold', padding.left + 5, thresholdY - 4);
}


// ============================================
// 4. PITCH CONTOUR DRAWING
// ============================================
function drawPitchContour(canvasId, data) {
    const { ctx, width, height } = setupCanvas(canvasId);
    const padding = { top: 20, right: 20, bottom: 30, left: 60 };
    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, width, height);

    // Filter out unvoiced frames (frequency = 0)
    const voiced = [];
    for (let i = 0; i < data.times.length; i++) {
        if (data.frequencies[i] > 0) {
            voiced.push({
                time: data.times[i],
                freq: data.frequencies[i],
                note: data.notes[i],
            });
        }
    }

    if (voiced.length === 0) {
        ctx.fillStyle = COLORS.text;
        ctx.font = '14px Poppins, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No pitch detected in this audio', width / 2, height / 2);
        return;
    }

    const maxTime = data.times[data.times.length - 1] || 1;
    const minFreq = Math.min(...voiced.map(v => v.freq)) * 0.9;
    const maxFreq = Math.max(...voiced.map(v => v.freq)) * 1.1;
    const freqRange = maxFreq - minFreq || 1;

    // Axis labels
    const xLabels = [];
    for (let i = 0; i <= 5; i++) {
        xLabels.push((maxTime * i / 5).toFixed(1) + 's');
    }
    const yLabels = [];
    for (let i = 0; i <= 5; i++) {
        yLabels.push(Math.round(maxFreq - freqRange * i / 5) + ' Hz');
    }

    drawGrid(ctx, width, height, padding, xLabels, yLabels);

    // Draw pitch points and connecting lines
    ctx.beginPath();
    let started = false;

    for (let i = 0; i < voiced.length; i++) {
        const x = padding.left + (voiced[i].time / maxTime) * plotW;
        const y = padding.top + plotH - ((voiced[i].freq - minFreq) / freqRange) * plotH;

        if (!started) {
            ctx.moveTo(x, y);
            started = true;
        } else {
            // Break the line if there's a big time gap (rest between notes)
            if (i > 0 && voiced[i].time - voiced[i - 1].time > 0.3) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
    }

    ctx.strokeStyle = 'rgba(103, 232, 249, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Draw individual dots
    for (let i = 0; i < voiced.length; i++) {
        const x = padding.left + (voiced[i].time / maxTime) * plotW;
        const y = padding.top + plotH - ((voiced[i].freq - minFreq) / freqRange) * plotH;

        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fillStyle = COLORS.cyan;
        ctx.fill();
    }

    // Label a few note names at pitch changes
    let lastNote = '';
    let labelCount = 0;
    for (let i = 0; i < voiced.length && labelCount < 8; i++) {
        if (voiced[i].note && voiced[i].note !== lastNote) {
            const x = padding.left + (voiced[i].time / maxTime) * plotW;
            const y = padding.top + plotH - ((voiced[i].freq - minFreq) / freqRange) * plotH;

            ctx.fillStyle = COLORS.textBright;
            ctx.font = 'bold 9px Poppins, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(voiced[i].note, x, y - 8);

            lastNote = voiced[i].note;
            labelCount++;
        }
    }
}