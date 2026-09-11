/**
 * ChordSense Grand Piano Engine
 * 
 * Uses REAL piano samples from the Salamander Grand Piano project.
 * The Salamander Piano is a Yamaha C5 grand piano recorded in a
 * professional studio. Each note was recorded at multiple velocities.
 * 
 * How Tone.Sampler works:
 * 1. We provide recordings of specific notes (every 3 semitones)
 * 2. Tone.js pitch-shifts these samples to fill in the gaps
 * 3. Result: all 64 keys sound realistic from just ~20 sample files
 * 
 * This is the same technique used by professional VST plugins
 * like Kontakt, Keyscape, and Pianoteq.
 */

// ============================================
// PIANO SAMPLER SETUP
// ============================================

let pianoSampler = null;
let isPianoReady = false;
let pianoReadyPromise = null;

/**
 * Initialize the piano with real grand piano samples.
 * 
 * The samples come from the Salamander Grand Piano project:
 * https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html
 * Hosted on the Tone.js CDN for easy access.
 * 
 * We load samples every 3 semitones (minor thirds).
 * Tone.js automatically pitch-shifts between them.
 * This gives good quality while keeping load time reasonable.
 */
function initPiano() {
    if (pianoReadyPromise) return pianoReadyPromise;

    pianoReadyPromise = new Promise(async (resolve) => {
        // Show loading indicator
        const loadingEl = document.getElementById('pianoLoading');
        if (loadingEl) loadingEl.style.display = 'flex';

        // Base URL for Salamander Grand Piano samples on Tone.js CDN
        const baseUrl = 'https://tonejs.github.io/audio/salamander/';

        // We provide samples every 3 semitones across the full range.
        // Tone.Sampler will pitch-shift these to cover all 64 keys.
        // Format: { "NoteName": "url" }
        const sampleMap = {
            'A1': baseUrl + 'A1.mp3',
            'C2': baseUrl + 'C2.mp3',
            'D#2': baseUrl + 'Ds2.mp3',
            'F#2': baseUrl + 'Fs2.mp3',
            'A2': baseUrl + 'A2.mp3',
            'C3': baseUrl + 'C3.mp3',
            'D#3': baseUrl + 'Ds3.mp3',
            'F#3': baseUrl + 'Fs3.mp3',
            'A3': baseUrl + 'A3.mp3',
            'C4': baseUrl + 'C4.mp3',
            'D#4': baseUrl + 'Ds4.mp3',
            'F#4': baseUrl + 'Fs4.mp3',
            'A4': baseUrl + 'A4.mp3',
            'C5': baseUrl + 'C5.mp3',
            'D#5': baseUrl + 'Ds5.mp3',
            'F#5': baseUrl + 'Fs5.mp3',
            'A5': baseUrl + 'A5.mp3',
            'C6': baseUrl + 'C6.mp3',
            'D#6': baseUrl + 'Ds6.mp3',
            'F#6': baseUrl + 'Fs6.mp3',
            'A6': baseUrl + 'A6.mp3',
            'C7': baseUrl + 'C7.mp3',
            'D#7': baseUrl + 'Ds7.mp3',
            'F#7': baseUrl + 'Fs7.mp3',
            'A7': baseUrl + 'A7.mp3',
        };

        pianoSampler = new Tone.Sampler({
            urls: sampleMap,

            // Release = how long the string rings after key release
            // Real piano strings ring for a long time, especially low notes
            release: 1.5,

            // Volume adjustment to prevent clipping with loud chords
            volume: -4,

            // Called when ALL samples finish loading
            onload: () => {
                isPianoReady = true;
                if (loadingEl) loadingEl.style.display = 'none';
                console.log('🎹 Grand Piano samples loaded!');
                resolve();
            },

            // Called if any sample fails to load
            onerror: (err) => {
                console.error('Piano sample error:', err);
                if (loadingEl) {
                    loadingEl.innerHTML = '<span style="color:#ef4444;">Failed to load piano samples. Check your internet connection.</span>';
                }
            }
        }).toDestination();
    });

    return pianoReadyPromise;
}

/**
 * Play one or more notes on the grand piano.
 * 
 * @param {string|string[]} notes - "C4" or ["C4", "E4", "G4"]
 * @param {string} duration - "4n" (quarter note), "2n" (half), "1n" (whole)
 * @param {number} velocity - How hard the key is struck (0-1)
 */
async function playPianoNotes(notes, duration = "2n", velocity = 0.8) {
    // Browser security: must start audio context after user interaction
    if (Tone.context.state !== 'running') {
        await Tone.start();
    }

    // Wait for samples to finish loading
    if (!isPianoReady) {
        await initPiano();
    }

    if (!pianoSampler) return;

    const noteArray = Array.isArray(notes) ? notes : [notes];

    // triggerAttackRelease(notes, duration, time, velocity)
    pianoSampler.triggerAttackRelease(noteArray, duration, Tone.now(), velocity);
}

/**
 * Press a key (note starts, sustains until released).
 */
async function pressPianoKey(note) {
    if (Tone.context.state !== 'running') {
        await Tone.start();
    }
    if (!isPianoReady) await initPiano();
    if (!pianoSampler) return;

    pianoSampler.triggerAttack(note, Tone.now(), 0.8);
}

/**
 * Release a key (note fades out with string resonance).
 */
function releasePianoKey(note) {
    if (!pianoSampler) return;
    pianoSampler.triggerRelease(note, Tone.now());
}

function stopPiano() {
    if (pianoSampler) {
        pianoSampler.releaseAll();
    }
}

// ============================================
// 64-KEY PIANO DEFINITION (C2 to E7)
// ============================================

const ALL_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function generatePianoKeys(startOctave, endOctave, extraNotes) {
    /**
     * Generate piano key definitions for a range of octaves.
     * 
     * @param {number} startOctave - First octave (e.g., 2)
     * @param {number} endOctave - Last full octave (e.g., 6)
     * @param {string[]} extraNotes - Additional notes beyond the last octave
     * @returns {Array} Array of key objects
     */
    const keys = [];

    for (let oct = startOctave; oct <= endOctave; oct++) {
        ALL_NOTES.forEach(note => {
            keys.push({
                note: note + oct,
                type: note.includes('#') ? 'black' : 'white',
            });
        });
    }

    // Add extra notes in the final octave
    if (extraNotes) {
        extraNotes.forEach(note => {
            keys.push({
                note: note + (endOctave + 1),
                type: note.includes('#') ? 'black' : 'white',
            });
        });
    }

    return keys;
}

// 64 keys: C2 to E7
// C2-B2 (12) + C3-B3 (12) + C4-B4 (12) + C5-B5 (12) + C6-B6 (12) + C7-E7 (4) = 64
const PIANO_KEYS = generatePianoKeys(2, 6, ['C', 'C#', 'D', 'D#', 'E']);

// ============================================
// KEYBOARD MAPPING (2 octaves at a time)
// ============================================

// Maps computer keyboard keys to piano notes
// We map 2 octaves and provide octave shift buttons
const KEYBOARD_MAP_LOWER = {
    'z': 'C', 's': 'C#', 'x': 'D', 'd': 'D#', 'c': 'E',
    'v': 'F', 'g': 'F#', 'b': 'G', 'h': 'G#', 'n': 'A',
    'j': 'A#', 'm': 'B',
};

const KEYBOARD_MAP_UPPER = {
    'q': 'C', '2': 'C#', 'w': 'D', '3': 'D#', 'e': 'E',
    'r': 'F', '5': 'F#', 't': 'G', '6': 'G#', 'y': 'A',
    '7': 'A#', 'u': 'B',
    'i': 'C', '9': 'C#', 'o': 'D', '0': 'D#', 'p': 'E',
};

let keyboardOctave = 3;  // Starting octave for keyboard input

function shiftOctave(direction) {
    keyboardOctave = Math.max(2, Math.min(5, keyboardOctave + direction));
    const label = document.getElementById('octaveLabel');
    if (label) label.textContent = `Octave ${keyboardOctave}-${keyboardOctave + 1}`;
}

// ============================================
// BUILD PIANO HTML
// ============================================

function buildPiano(containerId, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const showLabels = options.showLabels !== false;
    const compact = options.compact || false;

    // Loading indicator
    let html = `
        <div class="piano-loading" id="pianoLoading" style="display:none;">
            <div class="audio-loader">
                <div class="bar"></div><div class="bar"></div>
                <div class="bar"></div><div class="bar"></div>
                <div class="bar"></div>
            </div>
            <p>Loading Grand Piano samples...</p>
        </div>
    `;

    // Octave shift controls
    html += `
        <div class="piano-controls-bar">
            <button class="btn btn-secondary btn-sm" onclick="shiftOctave(-1)">
                <i class="fas fa-arrow-left"></i> Lower
            </button>
            <span class="octave-label" id="octaveLabel">Octave ${keyboardOctave}-${keyboardOctave + 1}</span>
            <button class="btn btn-secondary btn-sm" onclick="shiftOctave(1)">
                Higher <i class="fas fa-arrow-right"></i>
            </button>
            <span class="piano-key-count">64 Keys · C2–E7</span>
        </div>
    `;

    // Piano keyboard
    html += '<div class="piano-keyboard-64">';

    PIANO_KEYS.forEach(k => {
        const isBlack = k.type === 'black';
        const displayNote = k.note.replace('#', '♯');

        // Find keyboard shortcut for this note
        let shortcut = '';
        if (!compact) {
            const noteName = k.note.replace(/\d/, '');
            const oct = parseInt(k.note.slice(-1));
            if (oct === keyboardOctave && KEYBOARD_MAP_LOWER) {
                for (const [key, note] of Object.entries(KEYBOARD_MAP_LOWER)) {
                    if (note === noteName) { shortcut = key.toUpperCase(); break; }
                }
            }
            if (oct === keyboardOctave + 1 && KEYBOARD_MAP_UPPER) {
                for (const [key, note] of Object.entries(KEYBOARD_MAP_UPPER)) {
                    if (note === noteName) { shortcut = key.toUpperCase(); break; }
                }
            }
        }

        html += `<div class="piano-key ${isBlack ? 'black' : 'white'} ${compact ? 'compact' : ''}" 
                      data-note="${k.note}"
                      onmousedown="onPianoKeyDown('${k.note}', this)"
                      onmouseup="onPianoKeyUp('${k.note}', this)"
                      onmouseleave="onPianoKeyUp('${k.note}', this)"
                      ontouchstart.prevent="onPianoKeyDown('${k.note}', this)"
                      ontouchend="onPianoKeyUp('${k.note}', this)">`;

        if (showLabels && !isBlack) {
            html += `<span class="key-label">${displayNote}</span>`;
        }
        if (shortcut && !isBlack) {
            html += `<span class="key-shortcut">${shortcut}</span>`;
        }

        html += '</div>';
    });

    html += '</div>';
    container.innerHTML = html;

    // Pre-load piano samples in the background
    initPiano();

    // Setup keyboard listeners
    setupKeyboardInput();
}

// ============================================
// PIANO INTERACTION
// ============================================

const activeKeys = new Set();

function onPianoKeyDown(note, element) {
    if (activeKeys.has(note)) return;
    activeKeys.add(note);
    element.classList.add('active');
    pressPianoKey(note);
}

function onPianoKeyUp(note, element) {
    if (!activeKeys.has(note)) return;
    activeKeys.delete(note);
    element.classList.remove('active');
    releasePianoKey(note);
}

function setupKeyboardInput() {
    const keyMap = {};

    function rebuildKeyMap() {
        Object.keys(keyMap).forEach(k => delete keyMap[k]);

        for (const [key, noteName] of Object.entries(KEYBOARD_MAP_LOWER)) {
            keyMap[key] = noteName + keyboardOctave;
        }
        for (const [key, noteName] of Object.entries(KEYBOARD_MAP_UPPER)) {
            keyMap[key] = noteName + (keyboardOctave + 1);
        }
    }

    rebuildKeyMap();

    // Rebuild key map when octave changes
    const origShift = window.shiftOctave;
    window.shiftOctave = function(dir) {
        origShift(dir);
        rebuildKeyMap();
        updateKeyLabels();
    };

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (e.repeat) return;

        const note = keyMap[e.key.toLowerCase()];
        if (note) {
            const keyEl = document.querySelector(`.piano-key[data-note="${note}"]`);
            if (keyEl) onPianoKeyDown(note, keyEl);
        }
    });

    document.addEventListener('keyup', (e) => {
        const note = keyMap[e.key.toLowerCase()];
        if (note) {
            const keyEl = document.querySelector(`.piano-key[data-note="${note}"]`);
            if (keyEl) onPianoKeyUp(note, keyEl);
        }
    });
}

function updateKeyLabels() {
    // Update the shortcut labels on visible keys
    document.querySelectorAll('.piano-key .key-shortcut').forEach(el => {
        el.textContent = '';
    });

    for (const [key, noteName] of Object.entries(KEYBOARD_MAP_LOWER)) {
        const note = noteName + keyboardOctave;
        const keyEl = document.querySelector(`.piano-key[data-note="${note}"] .key-shortcut`);
        if (keyEl) keyEl.textContent = key.toUpperCase();
    }
    for (const [key, noteName] of Object.entries(KEYBOARD_MAP_UPPER)) {
        const note = noteName + (keyboardOctave + 1);
        const keyEl = document.querySelector(`.piano-key[data-note="${note}"] .key-shortcut`);
        if (keyEl) keyEl.textContent = key.toUpperCase();
    }
}

// ============================================
// CHORD FUNCTIONS (for results page)
// ============================================

function chordToNotes(chordName) {
    const INTERVALS = {
        '':     [0, 4, 7],
        'm':    [0, 3, 7],
        '7':    [0, 4, 7, 10],
        'm7':   [0, 3, 7, 10],
        'maj7': [0, 4, 7, 11],
        'dim':  [0, 3, 6],
        'aug':  [0, 4, 8],
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
    };

    const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                        'F#', 'G', 'G#', 'A', 'A#', 'B'];

    let root = '';
    let type = '';

    if (chordName.length >= 2 && (chordName[1] === '#' || chordName[1] === '♯')) {
        root = chordName.substring(0, 2).replace('♯', '#');
        type = chordName.substring(2);
    } else {
        root = chordName.substring(0, 1);
        type = chordName.substring(1);
    }

    const rootIndex = NOTE_NAMES.indexOf(root);
    if (rootIndex === -1) return ['C4', 'E4', 'G4'];

    const intervals = INTERVALS[type] || INTERVALS[''];

    return intervals.map(interval => {
        const midi = 60 + rootIndex + interval;
        const noteIdx = midi % 12;
        const octave = Math.floor(midi / 12) - 1;
        return NOTE_NAMES[noteIdx] + octave;
    });
}

function highlightChord(chordName) {
    document.querySelectorAll('.piano-key.chord-highlight').forEach(el => {
        el.classList.remove('chord-highlight');
    });

    if (!chordName) return;

    const notes = chordToNotes(chordName);
    notes.forEach(note => {
        const keyEl = document.querySelector(`.piano-key[data-note="${note}"]`);
        if (keyEl) keyEl.classList.add('chord-highlight');
    });
}

function playChordOnPiano(chordName) {
    const notes = chordToNotes(chordName);
    highlightChord(chordName);
    playPianoNotes(notes, "2n", 0.7);
}

let chordPlaybackInterval = null;
let isChordPlaybackActive = false;

function toggleChordPlayback() {
    const btn = document.getElementById('chordPlayToggle');
    if (!btn) return;

    if (isChordPlaybackActive) {
        isChordPlaybackActive = false;
        clearInterval(chordPlaybackInterval);
        btn.innerHTML = '<i class="fas fa-play"></i> Auto-Play Chords';
        btn.classList.remove('active');
        highlightChord(null);
    } else {
        isChordPlaybackActive = true;
        btn.innerHTML = '<i class="fas fa-stop"></i> Stop';
        btn.classList.add('active');

        const chordBlocks = document.querySelectorAll('.chord-block:not(.rest-block)');
        let index = 0;

        function playNext() {
            if (!isChordPlaybackActive) return;
            if (index >= chordBlocks.length) index = 0;

            const block = chordBlocks[index];
            const chordName = block.querySelector('.chord-name').textContent.trim();

            if (chordName && chordName !== '—') {
                playChordOnPiano(chordName);
                block.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
            }
            index++;
        }

        playNext();
        chordPlaybackInterval = setInterval(playNext, 2000);
    }
}