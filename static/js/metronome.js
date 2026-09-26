/**
 * Crescendo Metronome
 * 
 * Precision timing using Web Audio API's scheduler.
 * 
 * Why not just use setInterval()?
 * setInterval is inaccurate — it can drift by 20-50ms per click,
 * which is very noticeable in music. Instead, we use Web Audio's
 * built-in scheduler that queues sounds at exact sample-accurate times.
 * 
 * Sound generation:
 * - Each click is an oscillator with a fast attack/decay envelope
 * - Different sounds use different oscillator types (sine, square, triangle)
 * - Accented beats use a higher frequency
 */

// ============================================
// METRONOME STATE
// ============================================

let metronomeAudioContext = null;
let isMetronomePlaying = false;
let currentBeat = 0;
let nextNoteTime = 0.0;
let schedulerTimerId = null;

// Settings
let bpm = 120;
let beatsPerMeasure = 4;
let noteValue = 4;
let volume = 0.8;
let selectedSound = 'click';

// Scheduler settings
const LOOKAHEAD_MS = 25.0;        // How often the scheduler runs
const SCHEDULE_AHEAD_TIME = 0.1;  // How far ahead to schedule sounds (seconds)

// Tap tempo state
let tapTimes = [];
const MAX_TAP_MEMORY = 8;

// ============================================
// TEMPO NAMES (Italian musical terms)
// ============================================

const TEMPO_NAMES = [
    { min: 0, max: 40, name: 'Grave' },
    { min: 40, max: 60, name: 'Largo' },
    { min: 60, max: 66, name: 'Larghetto' },
    { min: 66, max: 76, name: 'Adagio' },
    { min: 76, max: 108, name: 'Andante' },
    { min: 108, max: 120, name: 'Moderato' },
    { min: 120, max: 168, name: 'Allegro' },
    { min: 168, max: 200, name: 'Presto' },
    { min: 200, max: 300, name: 'Prestissimo' },
];

function getTempoName(bpm) {
    for (const tempo of TEMPO_NAMES) {
        if (bpm >= tempo.min && bpm < tempo.max) {
            return tempo.name;
        }
    }
    return 'Custom';
}

// ============================================
// SOUND GENERATION
// ============================================

/**
 * Generate a click sound at a specific time.
 * Uses oscillator + envelope for precise, sample-accurate timing.
 */
function playClick(time, isAccent) {
    if (!metronomeAudioContext) return;

    const osc = metronomeAudioContext.createOscillator();
    const gainNode = metronomeAudioContext.createGain();

    osc.connect(gainNode);
    gainNode.connect(metronomeAudioContext.destination);

    // Choose sound characteristics based on selected sound type
    let frequency, waveType, duration;

    switch (selectedSound) {
        case 'click':
            // Classic mechanical metronome click
            frequency = isAccent ? 1600 : 1000;
            waveType = 'square';
            duration = 0.03;
            break;

        case 'wood':
            // Woodblock sound
            frequency = isAccent ? 1200 : 800;
            waveType = 'triangle';
            duration = 0.05;
            break;

        case 'beep':
            // Digital beep
            frequency = isAccent ? 1500 : 900;
            waveType = 'sine';
            duration = 0.08;
            break;

        case 'cowbell':
            // Cowbell sound (two oscillators mixed)
            frequency = isAccent ? 850 : 560;
            waveType = 'square';
            duration = 0.1;
            break;

        default:
            frequency = 1000;
            waveType = 'square';
            duration = 0.03;
    }

    osc.type = waveType;
    osc.frequency.value = frequency;

    // Envelope: fast attack, fast decay (like a percussion hit)
    // This shapes the sound so it doesn't click abruptly (which would sound bad)
    const attackTime = 0.001;
    const decayTime = duration;
    const peakVolume = isAccent ? volume : volume * 0.7;

    gainNode.gain.setValueAtTime(0, time);
    gainNode.gain.linearRampToValueAtTime(peakVolume, time + attackTime);
    gainNode.gain.exponentialRampToValueAtTime(0.001, time + attackTime + decayTime);

    osc.start(time);
    osc.stop(time + attackTime + decayTime + 0.01);
}

// ============================================
// SCHEDULER
// ============================================

/**
 * Advance to the next note based on current tempo and time signature.
 */
function nextNote() {
    // Calculate seconds per beat
    // A beat = quarter note (in 4/4) or eighth note (in 6/8, etc.)
    // For compound meters (6/8, 9/8, 12/8), each beat is an eighth note
    let secondsPerBeat;

    if (noteValue === 8) {
        // Compound time signature — beat is eighth note
        secondsPerBeat = 60.0 / bpm / 2;
    } else {
        // Simple time signature — beat is quarter note
        secondsPerBeat = 60.0 / bpm;
    }

    nextNoteTime += secondsPerBeat;

    currentBeat++;
    if (currentBeat >= beatsPerMeasure) {
        currentBeat = 0;
    }
}

/**
 * Schedule a beat to play at a specific audio time.
 */
function scheduleNote(beatNumber, time) {
    const isAccent = (beatNumber === 0);
    playClick(time, isAccent);

    // Schedule visual update (approximate — visual doesn't need sample accuracy)
    const delayMs = (time - metronomeAudioContext.currentTime) * 1000;
    setTimeout(() => {
        updateBeatVisual(beatNumber);
    }, Math.max(0, delayMs));
}

/**
 * The scheduler runs frequently to queue up upcoming beats.
 */
function scheduler() {
    while (nextNoteTime < metronomeAudioContext.currentTime + SCHEDULE_AHEAD_TIME) {
        scheduleNote(currentBeat, nextNoteTime);
        nextNote();
    }

    if (isMetronomePlaying) {
        schedulerTimerId = setTimeout(scheduler, LOOKAHEAD_MS);
    }
}

// ============================================
// VISUAL UPDATES
// ============================================

function updateBeatVisual(beatNumber) {
    // Update beat lights
    const lights = document.querySelectorAll('.beat-light');
    lights.forEach((light, idx) => {
        light.classList.remove('active');
        if (idx === beatNumber) {
            light.classList.add('active');
        }
    });

    // Swing the pendulum
    const pendulum = document.getElementById('pendulum');
    if (pendulum) {
        // Alternate direction based on beat
        const direction = (beatNumber % 2 === 0) ? 'swing-right' : 'swing-left';
        pendulum.classList.remove('swing-right', 'swing-left');
        void pendulum.offsetWidth;  // Force reflow to restart animation
        pendulum.classList.add(direction);
    }
}

function rebuildBeatLights() {
    const container = document.getElementById('beatLights');
    if (!container) return;

    container.innerHTML = '';
    for (let i = 0; i < beatsPerMeasure; i++) {
        const light = document.createElement('div');
        light.className = 'beat-light';
        if (i === 0) light.classList.add('accent');
        container.appendChild(light);
    }
}

function updateBPMDisplay() {
    const bpmNumber = document.getElementById('bpmNumber');
    const tempoName = document.getElementById('tempoName');
    const bpmSlider = document.getElementById('bpmSlider');

    if (bpmNumber) bpmNumber.textContent = bpm;
    if (tempoName) tempoName.textContent = getTempoName(bpm);
    if (bpmSlider) bpmSlider.value = bpm;

    // Update pendulum swing speed
    const pendulum = document.getElementById('pendulum');
    if (pendulum) {
        const swingDuration = 60.0 / bpm;  // seconds per beat
        pendulum.style.setProperty('--swing-duration', swingDuration + 's');
    }
}

// ============================================
// START / STOP
// ============================================

function startMetronome() {
    if (isMetronomePlaying) return;

    // Create audio context on first play (browser security requirement)
    if (!metronomeAudioContext) {
        metronomeAudioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    // Resume if suspended
    if (metronomeAudioContext.state === 'suspended') {
        metronomeAudioContext.resume();
    }

    isMetronomePlaying = true;
    currentBeat = 0;
    nextNoteTime = metronomeAudioContext.currentTime + 0.05;

    scheduler();

    // Update play button
    const playIcon = document.getElementById('playIcon');
    const playBtn = document.getElementById('metronomePlayBtn');
    if (playIcon) playIcon.className = 'fas fa-stop';
    if (playBtn) playBtn.classList.add('playing');
}

function stopMetronome() {
    isMetronomePlaying = false;

    if (schedulerTimerId) {
        clearTimeout(schedulerTimerId);
    }

    // Reset visuals
    const lights = document.querySelectorAll('.beat-light');
    lights.forEach(light => light.classList.remove('active'));

    const pendulum = document.getElementById('pendulum');
    if (pendulum) {
        pendulum.classList.remove('swing-right', 'swing-left');
    }

    // Update play button
    const playIcon = document.getElementById('playIcon');
    const playBtn = document.getElementById('metronomePlayBtn');
    if (playIcon) playIcon.className = 'fas fa-play';
    if (playBtn) playBtn.classList.remove('playing');
}

function toggleMetronome() {
    if (isMetronomePlaying) {
        stopMetronome();
    } else {
        startMetronome();
    }
}

// ============================================
// TAP TEMPO
// ============================================

function tapTempo() {
    const now = Date.now();
    tapTimes.push(now);

    // Keep only recent taps
    if (tapTimes.length > MAX_TAP_MEMORY) {
        tapTimes.shift();
    }

    // Remove taps older than 3 seconds (user paused)
    tapTimes = tapTimes.filter(t => now - t < 3000);

    if (tapTimes.length >= 2) {
        // Calculate average interval between taps
        const intervals = [];
        for (let i = 1; i < tapTimes.length; i++) {
            intervals.push(tapTimes[i] - tapTimes[i - 1]);
        }
        const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;

        // Convert interval (ms) to BPM
        const newBpm = Math.round(60000 / avgInterval);

        if (newBpm >= 30 && newBpm <= 240) {
            bpm = newBpm;
            updateBPMDisplay();
        }
    }

    // Visual feedback
    const tapBtn = document.getElementById('tapTempoBtn');
    if (tapBtn) {
        tapBtn.classList.add('tapped');
        setTimeout(() => tapBtn.classList.remove('tapped'), 100);
    }
}

// ============================================
// EVENT LISTENERS
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    // Play/Stop button
    const playBtn = document.getElementById('metronomePlayBtn');
    if (playBtn) playBtn.addEventListener('click', toggleMetronome);

    // BPM Slider
    const bpmSlider = document.getElementById('bpmSlider');
    if (bpmSlider) {
        bpmSlider.addEventListener('input', function() {
            bpm = parseInt(this.value);
            updateBPMDisplay();
        });
    }

    // BPM buttons
    document.getElementById('bpmMinus10')?.addEventListener('click', () => {
        bpm = Math.max(30, bpm - 10);
        updateBPMDisplay();
    });
    document.getElementById('bpmMinus1')?.addEventListener('click', () => {
        bpm = Math.max(30, bpm - 1);
        updateBPMDisplay();
    });
    document.getElementById('bpmPlus1')?.addEventListener('click', () => {
        bpm = Math.min(240, bpm + 1);
        updateBPMDisplay();
    });
    document.getElementById('bpmPlus10')?.addEventListener('click', () => {
        bpm = Math.min(240, bpm + 10);
        updateBPMDisplay();
    });

    // Preset buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            bpm = parseInt(this.getAttribute('data-bpm'));
            updateBPMDisplay();
        });
    });

    // Time signature buttons
    document.querySelectorAll('.ts-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.ts-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            beatsPerMeasure = parseInt(this.getAttribute('data-beats'));
            noteValue = parseInt(this.getAttribute('data-note'));
            rebuildBeatLights();

            // Restart if playing to apply new time signature
            if (isMetronomePlaying) {
                stopMetronome();
                setTimeout(startMetronome, 100);
            }
        });
    });

    // Sound buttons
    document.querySelectorAll('.sound-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.sound-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            selectedSound = this.getAttribute('data-sound');
        });
    });

    // Volume slider
    const volumeSlider = document.getElementById('volumeSlider');
    if (volumeSlider) {
        volumeSlider.addEventListener('input', function() {
            volume = parseInt(this.value) / 100;
            const volumeValue = document.getElementById('volumeValue');
            if (volumeValue) volumeValue.textContent = this.value + '%';
        });
    }

    // Tap tempo button
    const tapBtn = document.getElementById('tapTempoBtn');
    if (tapBtn) tapBtn.addEventListener('click', tapTempo);

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT') return;

        if (e.code === 'Space') {
            e.preventDefault();
            toggleMetronome();
        } else if (e.code === 'KeyT') {
            e.preventDefault();
            tapTempo();
        }
    });

    // Initial setup
    rebuildBeatLights();
    updateBPMDisplay();
});

// Cleanup when leaving page
window.addEventListener('beforeunload', function() {
    if (isMetronomePlaying) {
        stopMetronome();
    }
    if (metronomeAudioContext) {
        metronomeAudioContext.close();
    }
});