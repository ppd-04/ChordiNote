/**
 * ChordiNote — Realistic Guitar Fretboard Visualizer
 *
 * Renders an SVG guitar neck with:
 *   - Rosewood-style dark gradient background
 *   - 6 strings with varied thickness & metallic colors (E A D G B e)
 *   - Nickel fret dividers + ivory nut
 *   - Mother-of-pearl inlay dots (3, 5, 7, 9, double at 12)
 *   - Glowing animated fingering markers when notes are detected
 *   - Fret numbers along the bottom
 *   - String labels (EADGBE) on the left
 *   - Full real-time sync with the audio playback engine on results page
 *
 * No libraries required — pure SVG + CSS + vanilla JS.
 */

// ─────────────────────────────────────────────────────────────────────────────
//  CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const FB_STRINGS = 6;   // E A D G B e  (low to high, top to bottom visually)
const FB_FRETS   = 15;  // frets 0..14 (open + 14 fretted positions shown)

// Standard tuning — MIDI numbers for open strings, low E first
const OPEN_MIDI = [40, 45, 50, 55, 59, 64]; // E2 A2 D3 G3 B3 e4

// String labels shown at the nut end
const STRING_LABELS = ['E', 'A', 'D', 'G', 'B', 'e'];

// Inlay dot positions (fret numbers)
const SINGLE_DOTS  = [3, 5, 7, 9];
const DOUBLE_DOTS  = [12];

// String visual thickness (px) — low E is thickest
const STRING_THICKNESS = [2.6, 2.1, 1.75, 1.4, 1.1, 0.85];

// String metallic colors — wound strings are bronze-ish, plain strings silver
const STRING_COLORS = [
    '#b8934a',
    '#b8934a',
    '#c8a060',
    '#d4b070',
    '#d8d0c0',
    '#e8e4dc',
];

// Layout (in SVG user units)
const FB_LEFT_MARGIN    = 52;
const FB_RIGHT_MARGIN   = 24;
const FB_TOP_MARGIN     = 28;
const FB_BOT_MARGIN     = 32;
const FB_NUT_WIDTH      = 8;
const FB_STRING_SPACING = 34;
const FB_FRET_SPACING   = 76;

// Derived total SVG size
const FB_SVG_WIDTH  = FB_LEFT_MARGIN + FB_NUT_WIDTH + FB_FRETS * FB_FRET_SPACING + FB_RIGHT_MARGIN;
const FB_SVG_HEIGHT = FB_TOP_MARGIN + (FB_STRINGS - 1) * FB_STRING_SPACING + FB_BOT_MARGIN;

const FB_COLORS = {
    dotFill:      'rgba(230,220,200,0.18)',
    dotStroke:    'rgba(230,220,200,0.35)',
    fretLabel:    'rgba(200,185,160,0.6)',
    stringLabel:  'rgba(230,210,180,0.75)',
    markerColors: ['#a78bfa', '#f472b6', '#34d399', '#60a5fa', '#fbbf24', '#fb923c'],
};

// ─────────────────────────────────────────────────────────────────────────────
//  MIDI → FRET MAPPING
// ─────────────────────────────────────────────────────────────────────────────

function midiToFretPositions(midi) {
    const positions = [];
    for (let s = 0; s < FB_STRINGS; s++) {
        const fret = midi - OPEN_MIDI[s];
        if (fret >= 0 && fret <= FB_FRETS) {
            positions.push({ string: s, fret });
        }
    }
    return positions;
}

function chooseBestPositions(midiSet) {
    if (!midiSet || midiSet.length === 0) return [];
    const allCandidates = midiSet.map(midi => midiToFretPositions(midi));
    const chosen = [];
    const usedStrings = new Set();

    for (let i = 0; i < allCandidates.length; i++) {
        const candidates = allCandidates[i];
        let best = null;
        for (const pos of candidates) {
            if (!usedStrings.has(pos.string)) {
                if (!best || pos.fret < best.fret) best = pos;
            }
        }
        if (!best && candidates.length > 0) {
            best = candidates.reduce((a, b) => a.fret < b.fret ? a : b);
        }
        if (best) {
            chosen.push({ ...best, midiSource: midiSet[i] });
            usedStrings.add(best.string);
        }
    }
    return chosen;
}

// ─────────────────────────────────────────────────────────────────────────────
//  SVG HELPERS
// ─────────────────────────────────────────────────────────────────────────────

const svgNS = 'http://www.w3.org/2000/svg';

function svgEl(tag, attrs = {}) {
    const el = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
}

function fretX(fret) {
    return FB_LEFT_MARGIN + FB_NUT_WIDTH + fret * FB_FRET_SPACING;
}
function stringY(stringIdx) {
    return FB_TOP_MARGIN + stringIdx * FB_STRING_SPACING;
}
function markerX(fret) {
    if (fret === 0) return FB_LEFT_MARGIN - 14;
    return fretX(fret) - FB_FRET_SPACING / 2;
}

// ─────────────────────────────────────────────────────────────────────────────
//  BUILD STATIC FRETBOARD SVG
// ─────────────────────────────────────────────────────────────────────────────

function buildFretboard(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return null;

    const svg = svgEl('svg', {
        id: 'fretboard-svg',
        viewBox: `0 0 ${FB_SVG_WIDTH} ${FB_SVG_HEIGHT}`,
        preserveAspectRatio: 'xMidYMid meet',
    });
    svg.style.width = '100%';
    svg.style.height = 'auto';
    svg.style.display = 'block';
    svg.style.maxHeight = '240px';

    // ── Defs ────────────────────────────────────────────────────────────────
    const defs = svgEl('defs');

    // Neck wood gradient
    const neckGrad = svgEl('linearGradient', { id: 'neckGrad', x1: '0', y1: '0', x2: '0', y2: '1' });
    [
        ['0%',   '#1c0e09'],
        ['18%',  '#2b1810'],
        ['50%',  '#321d12'],
        ['82%',  '#2b1810'],
        ['100%', '#1c0e09'],
    ].forEach(([offset, color]) => {
        const stop = svgEl('stop', { offset });
        stop.style.stopColor = color;
        neckGrad.appendChild(stop);
    });
    defs.appendChild(neckGrad);

    // Glow filters
    FB_COLORS.markerColors.forEach((color, i) => {
        const filter = svgEl('filter', { id: `fbGlow${i}`, x: '-60%', y: '-60%', width: '220%', height: '220%' });
        const blur = svgEl('feGaussianBlur', { in: 'SourceGraphic', stdDeviation: '4.5', result: 'blur' });
        const flood = svgEl('feFlood', { 'flood-color': color, 'flood-opacity': '1', result: 'color' });
        const comp  = svgEl('feComposite', { in: 'color', in2: 'blur', operator: 'in', result: 'glow' });
        const merge = svgEl('feMerge');
        merge.appendChild(svgEl('feMergeNode', { in: 'glow' }));
        merge.appendChild(svgEl('feMergeNode', { in: 'SourceGraphic' }));
        filter.append(blur, flood, comp, merge);
        defs.appendChild(filter);
    });

    svg.appendChild(defs);

    // ── Neck body ────────────────────────────────────────────────────────────
    svg.appendChild(svgEl('rect', {
        x: FB_LEFT_MARGIN, y: 0,
        width: FB_SVG_WIDTH - FB_LEFT_MARGIN,
        height: FB_SVG_HEIGHT,
        fill: 'url(#neckGrad)',
        rx: '4',
    }));

    // Subtle side-binding strips
    svg.appendChild(svgEl('rect', {
        x: FB_LEFT_MARGIN, y: 0,
        width: FB_SVG_WIDTH - FB_LEFT_MARGIN, height: 3,
        fill: 'rgba(255,220,150,0.12)', rx: '2',
    }));
    svg.appendChild(svgEl('rect', {
        x: FB_LEFT_MARGIN, y: FB_SVG_HEIGHT - 3,
        width: FB_SVG_WIDTH - FB_LEFT_MARGIN, height: 3,
        fill: 'rgba(255,220,150,0.12)', rx: '2',
    }));

    // ── Fret wires ───────────────────────────────────────────────────────────
    for (let f = 1; f <= FB_FRETS; f++) {
        const x = fretX(f);
        svg.appendChild(svgEl('line', {
            x1: x, y1: FB_TOP_MARGIN - 8,
            x2: x, y2: stringY(FB_STRINGS - 1) + 8,
            stroke: 'rgba(210,200,175,0.6)',
            'stroke-width': f % 12 === 0 ? 2.8 : 1.8,
        }));
        // Highlight edge
        svg.appendChild(svgEl('line', {
            x1: x + 0.6, y1: FB_TOP_MARGIN - 8,
            x2: x + 0.6, y2: stringY(FB_STRINGS - 1) + 8,
            stroke: 'rgba(255,248,220,0.18)',
            'stroke-width': '0.8',
        }));
    }

    // ── Nut ──────────────────────────────────────────────────────────────────
    svg.appendChild(svgEl('rect', {
        x: FB_LEFT_MARGIN,
        y: FB_TOP_MARGIN - 8,
        width: FB_NUT_WIDTH,
        height: (FB_STRINGS - 1) * FB_STRING_SPACING + 16,
        fill: '#f0e8d0',
        rx: '1.5',
    }));

    // ── Inlay dots ───────────────────────────────────────────────────────────
    SINGLE_DOTS.forEach(fret => {
        const x = markerX(fret);
        const y = FB_TOP_MARGIN + ((FB_STRINGS - 1) * FB_STRING_SPACING) / 2;
        svg.appendChild(svgEl('circle', {
            cx: x, cy: y, r: '8',
            fill: FB_COLORS.dotFill, stroke: FB_COLORS.dotStroke, 'stroke-width': '1',
        }));
    });
    DOUBLE_DOTS.forEach(fret => {
        const x = markerX(fret);
        const midY = FB_TOP_MARGIN + ((FB_STRINGS - 1) * FB_STRING_SPACING) / 2;
        [-14, 14].forEach(dy => {
            svg.appendChild(svgEl('circle', {
                cx: x, cy: midY + dy, r: '7',
                fill: FB_COLORS.dotFill, stroke: FB_COLORS.dotStroke, 'stroke-width': '1',
            }));
        });
    });

    // ── Strings ──────────────────────────────────────────────────────────────
    for (let s = 0; s < FB_STRINGS; s++) {
        const y = stringY(s);
        const th = STRING_THICKNESS[s];
        // Shadow
        svg.appendChild(svgEl('line', {
            x1: FB_LEFT_MARGIN + FB_NUT_WIDTH, y1: y + th / 2 + 0.6,
            x2: FB_SVG_WIDTH - FB_RIGHT_MARGIN, y2: y + th / 2 + 0.6,
            stroke: 'rgba(0,0,0,0.5)', 'stroke-width': th * 1.3,
        }));
        // String body
        svg.appendChild(svgEl('line', {
            x1: FB_LEFT_MARGIN + FB_NUT_WIDTH, y1: y,
            x2: FB_SVG_WIDTH - FB_RIGHT_MARGIN, y2: y,
            stroke: STRING_COLORS[s], 'stroke-width': th, 'stroke-linecap': 'round',
        }));
        // Highlight
        svg.appendChild(svgEl('line', {
            x1: FB_LEFT_MARGIN + FB_NUT_WIDTH, y1: y - th * 0.3,
            x2: FB_SVG_WIDTH - FB_RIGHT_MARGIN, y2: y - th * 0.3,
            stroke: 'rgba(255,255,255,0.25)', 'stroke-width': th * 0.28,
        }));
    }

    // ── String labels ────────────────────────────────────────────────────────
    for (let s = 0; s < FB_STRINGS; s++) {
        const y = stringY(s);
        const label = svgEl('text', {
            x: FB_LEFT_MARGIN - 10, y: y + 4.5,
            'text-anchor': 'end',
            fill: FB_COLORS.stringLabel,
            'font-size': '13',
            'font-family': 'Inter, Poppins, sans-serif',
            'font-weight': '600',
        });
        label.textContent = STRING_LABELS[s];
        svg.appendChild(label);
    }

    // ── Fret number labels ───────────────────────────────────────────────────
    [1, 3, 5, 7, 9, 12, 15].forEach(f => {
        if (f > FB_FRETS) return;
        const x = markerX(f);
        const y = stringY(FB_STRINGS - 1) + 22;
        const label = svgEl('text', {
            x, y, 'text-anchor': 'middle',
            fill: FB_COLORS.fretLabel,
            'font-size': '11',
            'font-family': 'Inter, Poppins, sans-serif',
        });
        label.textContent = f;
        svg.appendChild(label);
    });

    // ── Live marker group ────────────────────────────────────────────────────
    svg.appendChild(svgEl('g', { id: 'fb-markers' }));

    // ── Outer border ─────────────────────────────────────────────────────────
    svg.appendChild(svgEl('rect', {
        x: FB_LEFT_MARGIN, y: 0,
        width: FB_SVG_WIDTH - FB_LEFT_MARGIN, height: FB_SVG_HEIGHT,
        fill: 'none', stroke: 'rgba(255,220,160,0.08)', 'stroke-width': '1', rx: '4',
    }));

    container.innerHTML = '';
    container.appendChild(svg);
    return svg;
}

// ─────────────────────────────────────────────────────────────────────────────
//  LIVE MARKER RENDERING
// ─────────────────────────────────────────────────────────────────────────────

let _fbAnimId  = null;
let _fbPulse   = 0;

const FB_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
function midiShortName(midi) { return FB_NOTE_NAMES[midi % 12]; }

function renderFretboardMarkers(positions) {
    const group = document.getElementById('fb-markers');
    if (!group) return;
    group.innerHTML = '';
    if (!positions || positions.length === 0) return;

    positions.forEach((pos, i) => {
        const colorIdx = i % FB_COLORS.markerColors.length;
        const color    = FB_COLORS.markerColors[colorIdx];
        const cx = markerX(pos.fret);
        const cy = stringY(pos.string);

        // Outer pulse ring
        const ring = svgEl('circle', {
            cx, cy, r: '19',
            fill: 'none',
            stroke: color,
            'stroke-width': '2',
            opacity: '0.35',
            class: 'fb-ring',
        });
        group.appendChild(ring);

        // Main dot
        const dot = svgEl('circle', {
            cx, cy, r: '13',
            fill: color,
            opacity: '0.92',
            filter: `url(#fbGlow${colorIdx})`,
        });
        group.appendChild(dot);

        // Note name
        const txt = svgEl('text', {
            x: cx, y: cy + 4,
            'text-anchor': 'middle',
            fill: '#fff',
            'font-size': '9.5',
            'font-weight': '800',
            'font-family': 'Inter, Poppins, sans-serif',
            'pointer-events': 'none',
        });
        txt.textContent = pos.noteName || '';
        group.appendChild(txt);
    });
}

function _fbAnimLoop() {
    _fbPulse += 0.065;
    const rings = document.querySelectorAll('#fb-markers .fb-ring');
    rings.forEach(ring => {
        const s = Math.sin(_fbPulse);
        ring.setAttribute('opacity', (0.2 + 0.3 * (0.5 + 0.5 * s)).toFixed(3));
        ring.setAttribute('r',       (18 + 4 * (0.5 + 0.5 * s)).toFixed(2));
    });
    _fbAnimId = requestAnimationFrame(_fbAnimLoop);
}

// ─────────────────────────────────────────────────────────────────────────────
//  PUBLIC API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Initialize the fretboard inside `containerId`. Call once on DOMContentLoaded.
 */
function initFretboard(containerId) {
    buildFretboard(containerId);
    if (!_fbAnimId) _fbAnimLoop();
    console.log('🎸 Fretboard visualizer ready');
}

/**
 * Update fingering markers to reflect currently active MIDI pitches.
 * @param {number[]} midiNotes
 */
function updateFretboard(midiNotes) {
    if (!midiNotes || midiNotes.length === 0) {
        renderFretboardMarkers([]);
        return;
    }
    const positions = chooseBestPositions(midiNotes);
    const enriched  = positions.map(pos => ({
        ...pos,
        noteName: midiShortName(pos.midiSource),
    }));
    renderFretboardMarkers(enriched);
}

/** Clear all fingering markers. */
function clearFretboard() {
    renderFretboardMarkers([]);
}
