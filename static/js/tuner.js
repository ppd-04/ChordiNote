/**
 * ChordSense Live Guitar Tuner
 * 
 * Uses autocorrelation to detect pitch from the microphone in real-time.
 * This is the same DSP technique used by pYIN and most professional tuners.
 * 
 * How autocorrelation works:
 * 1. Capture a buffer of audio samples from the microphone
 * 2. Slide the waveform against itself at different offsets (lags)
 * 3. Find the lag where the waveform best matches itself
 * 4. That lag = the period of the wave → frequency = sampleRate / lag
 * 
 * No deep learning. Pure math.
 */

// ============================================
// TUNER STATE
// ============================================

let tunerAudioContext = null;
let tunerStream = null;
let tunerAnalyser = null;
let tunerAnimFrameId = null;
let isTunerRunning = false;

// Note names in chromatic scale
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

// Standard guitar tuning frequencies
const GUITAR_STRINGS = {
    'E2': 82.41,
    'A2': 110.00,
    'D3': 146.83,
    'G3': 196.00,
    'B3': 246.94,
    'E4': 329.63,
};

// ============================================
// PITCH DETECTION — AUTOCORRELATION
// ============================================

/**
 * Detect the fundamental frequency using autocorrelation.
 * 
 * This is the core DSP algorithm. It works by finding the repeating
 * period in the audio waveform.
 * 
 * @param {Float32Array} buffer - Raw audio samples from the microphone
 * @param {number} sampleRate - Audio sample rate (usually 44100 or 48000)
 * @returns {number} Detected frequency in Hz, or -1 if no pitch found
 */
function detectPitch(buffer, sampleRate) {
    const SIZE = buffer.length;

    // Step 1: Check if there's enough signal (not silence)
    let rms = 0;
    for (let i = 0; i < SIZE; i++) {
        rms += buffer[i] * buffer[i];
    }
    rms = Math.sqrt(rms / SIZE);

    // If the signal is too quiet, there's no note to detect
    if (rms < 0.01) return -1;

    // Step 2: Compute autocorrelation
    // Autocorrelation measures how similar the signal is to a
    // delayed version of itself. When the delay equals the wave's
    // period, the similarity peaks.
    const correlations = new Float32Array(SIZE);

    for (let lag = 0; lag < SIZE; lag++) {
        let sum = 0;
        for (let i = 0; i < SIZE - lag; i++) {
            sum += buffer[i] * buffer[i + lag];
        }
        correlations[lag] = sum;
    }

    // Step 3: Find the first dip after the initial peak
    // The autocorrelation starts high (lag 0 = perfect match with itself),
    // then drops. We need to find where it drops below a threshold
    // and then rises again (that rise = the fundamental period).
    let d = 0;
    while (d < SIZE && correlations[d] > correlations[d + 1]) {
        d++;
    }

    // Step 4: Find the highest peak after the dip
    // This peak corresponds to the fundamental period
    let maxVal = -1;
    let maxPos = -1;

    // Search range: 50 Hz to 1000 Hz
    // Guitar range: E2 (82 Hz) to about E5 (660 Hz) with harmonics
    const minLag = Math.floor(sampleRate / 1000);
    const maxLag = Math.floor(sampleRate / 50);

    for (let lag = Math.max(d, minLag); lag < Math.min(SIZE, maxLag); lag++) {
        if (correlations[lag] > maxVal) {
            maxVal = correlations[lag];
            maxPos = lag;
        }
    }

    // Step 5: Check if the peak is strong enough
    // The peak must be at least 10% of the zero-lag correlation
    if (maxVal < correlations[0] * 0.1) return -1;

    // Step 6: Refine the peak position using parabolic interpolation
    // This gives sub-sample accuracy for better tuning precision
    let refinedLag = maxPos;

    if (maxPos > 0 && maxPos < SIZE - 1) {
        const prev = correlations[maxPos - 1];
        const curr = correlations[maxPos];
        const next = correlations[maxPos + 1];

        const denominator = 2 * (2 * curr - prev - next);
        if (denominator !== 0) {
            refinedLag = maxPos + (prev - next) / denominator;
        }
    }

    // Step 7: Convert lag (period in samples) to frequency
    const frequency = sampleRate / refinedLag;

    return frequency;
}

/**
 * Convert a frequency in Hz to the nearest musical note and cents deviation.
 * 
 * @param {number} frequency - Frequency in Hz
 * @returns {object} { note: "A4", cents: -12.5, midi: 69 }
 */
function frequencyToNote(frequency) {
    if (frequency <= 0) return null;

    // MIDI note number formula:
    // midi = 69 + 12 * log2(freq / 440)
    const midiFloat = 69 + 12 * Math.log2(frequency / 440.0);
    const midiRound = Math.round(midiFloat);

    // Cents deviation: how far off from the nearest note
    // 100 cents = 1 semitone
    const cents = Math.round((midiFloat - midiRound) * 100);

    const noteName = NOTE_NAMES[((midiRound % 12) + 12) % 12];
    const octave = Math.floor(midiRound / 12) - 1;

    return {
        note: noteName + octave,
        noteName: noteName,
        octave: octave,
        cents: cents,
        midi: midiRound,
    };
}

/**
 * Find the closest guitar string to a detected note.
 * 
 * @param {number} frequency - Detected frequency
 * @returns {string|null} Closest guitar string name (e.g., "E2") or null
 */
function findClosestString(frequency) {
    let closest = null;
    let minDiff = Infinity;

    for (const [name, freq] of Object.entries(GUITAR_STRINGS)) {
        // Compare in cents (logarithmic distance)
        const diffCents = Math.abs(1200 * Math.log2(frequency / freq));
        if (diffCents < minDiff) {
            minDiff = diffCents;
            closest = name;
        }
    }

    // Only return if within 1 semitone (100 cents) of a guitar string
    if (minDiff < 100) return closest;
    return null;
}

// ============================================
// TUNER UI UPDATE
// ============================================

/**
 * Update all the tuner display elements with the detected pitch.
 */
function updateTunerDisplay(frequency) {
    const noteEl = document.getElementById('tunerNoteName');
    const freqEl = document.getElementById('tunerFrequency');
    const centsEl = document.getElementById('tunerCents');
    const needleEl = document.getElementById('tunerNeedle');

    if (frequency <= 0) {
        noteEl.textContent = '—';
        noteEl.className = 'tuner-note-name';
        freqEl.textContent = 'Waiting for signal...';
        centsEl.textContent = '';
        centsEl.className = 'tuner-cents';
        needleEl.style.left = '50%';
        needleEl.style.background = '#666';
        highlightString(null);
        return;
    }

    const noteInfo = frequencyToNote(frequency);
    if (!noteInfo) return;

    // Update note name
    noteEl.textContent = noteInfo.note;

    // Update frequency
    freqEl.textContent = frequency.toFixed(1) + ' Hz';

    // Update cents display
    const centsAbs = Math.abs(noteInfo.cents);
    if (noteInfo.cents > 0) {
        centsEl.textContent = '+' + noteInfo.cents + ' cents (sharp)';
    } else if (noteInfo.cents < 0) {
        centsEl.textContent = noteInfo.cents + ' cents (flat)';
    } else {
        centsEl.textContent = 'Perfect!';
    }

    // Color coding based on tuning accuracy
    if (centsAbs <= 5) {
        // In tune — green
        noteEl.className = 'tuner-note-name in-tune';
        centsEl.className = 'tuner-cents in-tune';
        needleEl.style.background = '#4ade80';
    } else if (centsAbs <= 15) {
        // Close — yellow
        noteEl.className = 'tuner-note-name close';
        centsEl.className = 'tuner-cents close';
        needleEl.style.background = '#facc15';
    } else {
        // Off — red/orange
        noteEl.className = 'tuner-note-name off-tune';
        centsEl.className = 'tuner-cents off-tune';
        needleEl.style.background = '#ef4444';
    }

    // Move the needle
    // Cents range: -50 to +50 mapped to 0% to 100%
    const needlePercent = Math.max(2, Math.min(98, 50 + noteInfo.cents));
    needleEl.style.left = needlePercent + '%';

    // Highlight closest guitar string
    const closestString = findClosestString(frequency);
    highlightString(closestString);
}

/**
 * Highlight the matching guitar string button.
 */
function highlightString(stringName) {
    document.querySelectorAll('.string-btn').forEach(btn => {
        btn.classList.remove('active-string');
        btn.classList.remove('in-tune-string');
    });

    if (!stringName) return;

    const btn = document.querySelector(`.string-btn[data-note="${stringName}"]`);
    if (btn) {
        btn.classList.add('active-string');

        // Check if it's in tune
        const centsEl = document.getElementById('tunerCents');
        if (centsEl && centsEl.textContent.includes('Perfect')) {
            btn.classList.add('in-tune-string');
        }
    }
}

// ============================================
// TUNER MAIN LOOP
// ============================================

/**
 * Main tuner loop — runs ~30 times per second.
 * Captures audio, detects pitch, updates display.
 */
function tunerLoop() {
    if (!isTunerRunning) return;

    const buffer = new Float32Array(tunerAnalyser.fftSize);
    tunerAnalyser.getFloatTimeDomainData(buffer);

    const frequency = detectPitch(buffer, tunerAudioContext.sampleRate);
    updateTunerDisplay(frequency);

    tunerAnimFrameId = requestAnimationFrame(tunerLoop);
}

// ============================================
// START / STOP
// ============================================

async function startTuner() {
    try {
        // Request microphone access
        tunerStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,   // Important! Don't process the audio
                noiseSuppression: false,   // We want the raw signal
                autoGainControl: false,    // Don't auto-adjust volume
            }
        });

        // Create audio context and analyser
        tunerAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = tunerAudioContext.createMediaStreamSource(tunerStream);

        // fftSize = 4096 gives good frequency resolution for guitar
        // At 44100 Hz sample rate: resolution ≈ 10.8 Hz per bin
        tunerAnalyser = tunerAudioContext.createAnalyser();
        tunerAnalyser.fftSize = 4096;
        tunerAnalyser.smoothingTimeConstant = 0;  // No smoothing for fast response

        source.connect(tunerAnalyser);

        isTunerRunning = true;

        // Update UI
        document.getElementById('tunerStartBtn').style.display = 'none';
        document.getElementById('tunerStopBtn').style.display = 'inline-flex';

        // Start the detection loop
        tunerLoop();

    } catch (error) {
        console.error('Tuner error:', error);
        alert('Could not access microphone. Please grant permission and try again.');
    }
}

function stopTuner() {
    isTunerRunning = false;

    if (tunerAnimFrameId) {
        cancelAnimationFrame(tunerAnimFrameId);
    }

    if (tunerStream) {
        tunerStream.getTracks().forEach(track => track.stop());
    }

    if (tunerAudioContext) {
        tunerAudioContext.close();
    }

    // Update UI
    document.getElementById('tunerStartBtn').style.display = 'inline-flex';
    document.getElementById('tunerStopBtn').style.display = 'none';

    // Reset display
    updateTunerDisplay(-1);
}

// ============================================
// EVENT LISTENERS
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    const startBtn = document.getElementById('tunerStartBtn');
    const stopBtn = document.getElementById('tunerStopBtn');

    if (startBtn) startBtn.addEventListener('click', startTuner);
    if (stopBtn) stopBtn.addEventListener('click', stopTuner);
});