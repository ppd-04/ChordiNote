let guitarStrings = [];
let bodyResonance = null;
let guitarReverb = null;
let isGuitarReady = false;


const STRING_PARAMS = [
    // E2 (6th string) — thick wound string, warm and boomy
    { note: 'E2', attackNoise: 0.8, dampening: 2200, resonance: 0.97, volume: -8 },
    // A2 (5th string) — wound string, warm
    { note: 'A2', attackNoise: 1.0, dampening: 2800, resonance: 0.96, volume: -7 },
    // D3 (4th string) — wound string, balanced
    { note: 'D3', attackNoise: 1.2, dampening: 3200, resonance: 0.95, volume: -6 },
    // G3 (3rd string) — plain or wound, bright
    { note: 'G3', attackNoise: 1.5, dampening: 3800, resonance: 0.93, volume: -6 },
    // B3 (2nd string) — plain string, bright and snappy
    { note: 'B3', attackNoise: 1.8, dampening: 4500, resonance: 0.90, volume: -7 },
    // E4 (1st string) — thinnest, brightest, most attack
    { note: 'E4', attackNoise: 2.0, dampening: 5500, resonance: 0.88, volume: -8 },
];

// eta ajaira
async function initGuitar() {
    if (isGuitarReady) return;

    //  eshb ke tweak kore sundor bananolagbe
    guitarReverb = new Tone.Freeverb({
        roomSize: 0.6,   
        dampening: 3000,  
        wet: 0.25,        
    }).toDestination();


    bodyResonance = new Tone.MembraneSynth({
        pitchDecay: 0.05,
        octaves: 2,
        oscillator: { type: 'sine' },
        envelope: {
            attack: 0.001,
            decay: 0.3,
            sustain: 0.0,
            release: 0.2,
        },
        volume: -18,
    }).connect(guitarReverb);


    guitarStrings = STRING_PARAMS.map(params => {
        const synth = new Tone.PluckSynth({
            attackNoise: params.attackNoise,
            dampening: params.dampening,
            resonance: params.resonance,
            volume: params.volume,
        }).connect(guitarReverb);
        return synth;
    });

    isGuitarReady = true;
}

// ============================================
// GUITAR CHORD DEFINITIONS (Standard Tuning)
// ============================================

const GUITAR_CHORDS = {
    'C':   { notes: ['x', 'x', 'D3', 'G3', 'B3', 'E4'], frets: 'x32010', name: 'C Major' },
    'D':   { notes: ['x', 'x', 'D3', 'A3', 'D4', 'F#4'], frets: 'xx0232', name: 'D Major' },
    'E':   { notes: ['E2', 'B2', 'E3', 'G#3', 'B3', 'E4'], frets: '022100', name: 'E Major' },
    'F':   { notes: ['x', 'x', 'F3', 'A3', 'C4', 'F4'], frets: 'xx3211', name: 'F Major' },
    'G':   { notes: ['G2', 'B2', 'D3', 'G3', 'B3', 'G4'], frets: '320003', name: 'G Major' },
    'A':   { notes: ['x', 'A2', 'E3', 'A3', 'C#4', 'E4'], frets: 'x02220', name: 'A Major' },
    'B':   { notes: ['x', 'B2', 'F#3', 'B3', 'D#4', 'F#4'], frets: 'x24442', name: 'B Major' },
    'Cm':  { notes: ['x', 'C3', 'Eb3', 'G3', 'C4', 'Eb4'], frets: 'x35543', name: 'C Minor' },
    'Dm':  { notes: ['x', 'x', 'D3', 'A3', 'D4', 'F4'], frets: 'xx0231', name: 'D Minor' },
    'Em':  { notes: ['E2', 'B2', 'E3', 'G3', 'B3', 'E4'], frets: '022000', name: 'E Minor' },
    'Fm':  { notes: ['x', 'x', 'F3', 'Ab3', 'C4', 'F4'], frets: 'xx3111', name: 'F Minor' },
    'Gm':  { notes: ['G2', 'Bb2', 'D3', 'G3', 'Bb3', 'G4'], frets: '355333', name: 'G Minor' },
    'Am':  { notes: ['x', 'A2', 'E3', 'A3', 'C4', 'E4'], frets: 'x02210', name: 'A Minor' },
    'Bm':  { notes: ['x', 'B2', 'F#3', 'B3', 'D4', 'F#4'], frets: 'x24432', name: 'B Minor' },
    'C7':  { notes: ['x', 'C3', 'E3', 'G3', 'Bb3', 'E4'], frets: 'x32310', name: 'C7' },
    'D7':  { notes: ['x', 'x', 'D3', 'A3', 'C4', 'F#4'], frets: 'xx0212', name: 'D7' },
    'E7':  { notes: ['E2', 'B2', 'D3', 'G#3', 'B3', 'E4'], frets: '020100', name: 'E7' },
    'G7':  { notes: ['G2', 'B2', 'D3', 'G3', 'B3', 'F4'], frets: '320001', name: 'G7' },
    'A7':  { notes: ['x', 'A2', 'E3', 'G3', 'C#4', 'E4'], frets: 'x02020', name: 'A7' },
    'Am7': { notes: ['x', 'A2', 'E3', 'G3', 'C4', 'E4'], frets: 'x02010', name: 'Am7' },
    'Dm7': { notes: ['x', 'x', 'D3', 'A3', 'C4', 'F4'], frets: 'xx0211', name: 'Dm7' },
};

const OPEN_STRINGS = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4'];


async function strumChord(chordName, direction = 'down') {
    if (Tone.context.state !== 'running') {
        await Tone.start();
    }
    if (!isGuitarReady) await initGuitar();

    const chord = GUITAR_CHORDS[chordName];
    if (!chord) return;

    const notes = chord.notes;

    // up strum ar down strum alada
    const baseDelay = direction === 'down' ? 0.028 : 0.018;
    // realistic korar jonno ektu delay
    let order;
    if (direction === 'down') {
        order = [5, 4, 3, 2, 1, 0]; // E2 → E4
    } else {
        order = [0, 1, 2, 3, 4, 5]; // E4 → E2
    }

    let delay = 0;
    for (let i = 0; i < order.length; i++) {
        const stringIdx = order[i];
        const note = notes[stringIdx];

        if (note !== 'x') {
            const time = Tone.now() + delay;


            const humanize = (Math.random() - 0.5) * 0.008;

 
            if (guitarStrings[stringIdx]) {
                guitarStrings[stringIdx].triggerAttack(note, time + humanize);
            }

 
            if (stringIdx >= 3 && direction === 'down' && i === 0) {
                bodyResonance.triggerAttackRelease(note, '8n', time, 0.2);
            }
        }


        delay += baseDelay + (Math.random() * 0.005);
    }
}



function setupChordButtons() {
    document.querySelectorAll('.chord-btn').forEach(btn => {
        btn.addEventListener('click', async function() {
            const chordName = this.getAttribute('data-chord');

            this.classList.add('strumming');
            setTimeout(() => this.classList.remove('strumming'), 800);

            await strumChord(chordName, 'down');
            addChordToLoop(chordName);
        });
    });
}

// ============================================
// LOOP PLAYER
// ============================================

let loopSlots = [null, null, null, null];
let isLoopPlaying = false;
let loopIntervalId = null;
let currentLoopStep = 0;

function addChordToLoop(chordName) {
    const emptyIdx = loopSlots.indexOf(null);
    if (emptyIdx === -1) return;
    loopSlots[emptyIdx] = chordName;
    updateLoopUI();
}

function updateLoopUI() {
    const slots = document.querySelectorAll('.loop-slot');
    slots.forEach((slot, idx) => {
        const chordSpan = slot.querySelector('.slot-chord');
        if (loopSlots[idx]) {
            chordSpan.textContent = loopSlots[idx];
            slot.classList.remove('empty');
            slot.classList.add('filled');
        } else {
            chordSpan.textContent = '—';
            slot.classList.add('empty');
            slot.classList.remove('filled');
        }
        slot.classList.remove('active');
    });
}

function clearLoop() {
    loopSlots = [null, null, null, null];
    updateLoopUI();
}

function executeStrum(chordName, pattern, beatInMeasure) {
    if (!chordName) return;

    switch (pattern) {
        case 'down':
            strumChord(chordName, 'down');
            break;
        case 'downup':
            strumChord(chordName, beatInMeasure % 2 === 0 ? 'down' : 'up');
            break;
        case 'island': {
            const p = ['down', 'down', 'up', 'down', 'up'];
            strumChord(chordName, p[beatInMeasure % p.length]);
            break;
        }
        case 'folk': {
            const p = ['down', 'down', 'up', 'up', 'down', 'up'];
            strumChord(chordName, p[beatInMeasure % p.length]);
            break;
        }
        default:
            strumChord(chordName, 'down');
    }
}

async function startLoop() {
    if (isLoopPlaying) return;

    const activeChords = loopSlots.filter(c => c !== null);
    if (activeChords.length === 0) {
        alert('Click some chords first to fill the loop slots!');
        return;
    }

    if (Tone.context.state !== 'running') await Tone.start();
    if (!isGuitarReady) await initGuitar();

    isLoopPlaying = true;
    currentLoopStep = 0;

    const bpm = parseInt(document.getElementById('loopBpm')?.value || 100);
    const pattern = document.getElementById('strumPattern')?.value || 'downup';

    const beatMs = 60000 / bpm;
    let subDivMs = beatMs;
    if (pattern === 'island' || pattern === 'folk') subDivMs = beatMs / 2;

    document.getElementById('loopPlayBtn').style.display = 'none';
    document.getElementById('loopStopBtn').style.display = 'inline-flex';

    function loopTick() {
        if (!isLoopPlaying) return;

        const filledSlots = loopSlots.filter(c => c !== null);
        const divisor = pattern === 'downup' ? 2 : (pattern === 'island' || pattern === 'folk' ? 4 : 1);
        const chordIdx = Math.floor(currentLoopStep / divisor) % filledSlots.length;

        document.querySelectorAll('.loop-slot').forEach(s => s.classList.remove('active'));
        const slotEls = document.querySelectorAll('.loop-slot.filled');
        if (slotEls[chordIdx]) slotEls[chordIdx].classList.add('active');

        executeStrum(filledSlots[chordIdx], pattern, currentLoopStep);
        currentLoopStep++;
    }

    loopTick();
    loopIntervalId = setInterval(loopTick, subDivMs);
}

function stopLoop() {
    isLoopPlaying = false;
    if (loopIntervalId) clearInterval(loopIntervalId);
    document.querySelectorAll('.loop-slot').forEach(s => s.classList.remove('active'));
    document.getElementById('loopPlayBtn').style.display = 'inline-flex';
    document.getElementById('loopStopBtn').style.display = 'none';
}

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    setupChordButtons();

    document.getElementById('loopPlayBtn')?.addEventListener('click', startLoop);
    document.getElementById('loopStopBtn')?.addEventListener('click', stopLoop);
    document.getElementById('loopClearBtn')?.addEventListener('click', clearLoop);

    const bpmSlider = document.getElementById('loopBpm');
    if (bpmSlider) {
        bpmSlider.addEventListener('input', function() {
            document.getElementById('loopBpmValue').textContent = this.value;
            if (isLoopPlaying) {
                stopLoop();
                setTimeout(startLoop, 100);
            }
        });
    }

    document.querySelectorAll('.loop-slot').forEach(slot => {
        slot.addEventListener('click', function() {
            const idx = parseInt(this.getAttribute('data-slot'));
            if (loopSlots[idx]) {
                loopSlots[idx] = null;
                updateLoopUI();
            }
        });
    });
});

async function playGuitarNotes(notes, velocity = 0.8) {
    if (Tone.context.state !== 'running') {
        await Tone.start();
    }
    if (!isGuitarReady) await initGuitar();

    const noteArray = Array.isArray(notes) ? notes : [notes];
    
    const time = Tone.now();
    noteArray.forEach((note, i) => {
        let stringIdx = 5; 
        if (typeof chooseBestPositions === 'function') {
           const m = note.match(/^([A-G][#b]?)(\d+)$/);
           if (m) {
               const noteNames = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
               const noteFlat  = {'Db':'C#','Eb':'D#','Fb':'E','Gb':'F#','Ab':'G#','Bb':'A#','Cb':'B'};
               const pname = noteFlat[m[1]] || m[1];
               const idx   = noteNames.indexOf(pname);
               const midi = (parseInt(m[2]) + 1) * 12 + idx;
               const pos = chooseBestPositions([midi]);
               if (pos && pos.length > 0) {
                   stringIdx = pos[0].string;
               }
           }
        }
        
        if (guitarStrings[stringIdx]) {
            guitarStrings[stringIdx].triggerAttack(note, time + (i * 0.015)); 
        }
    });
}