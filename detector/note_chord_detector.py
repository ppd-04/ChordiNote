"""
note_chord_detector.py  -  Symbolic Grid Hybrid Chord Detector
==============================================================

Pipeline
--------
1.  Parse V4 Polyphonic note events (already overtone-free).
2.  Map notes onto a fixed time grid (e.g., 0.5-second sliding windows).
3.  Accumulate symbolic MIDI notes into a Pitch Class Profile (PCP) per window.
4.  Track the lowest MIDI note in each window as the Bass note.
5.  Score diatonic chord candidates against the PCP using cosine similarity,
    applying strong bonuses/penalties based on the Bass note.
6.  Smooth the sequence using a Viterbi HMM.
7.  Merge consecutive identical chords.
"""

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

_ENHARMONIC = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}

_CHORD_TEMPLATES = {
    'maj':  [0, 4, 7],
    'min':  [0, 3, 7],
    'dim':  [0, 3, 6],
    'maj7': [0, 4, 7, 11],
    '7':    [0, 4, 7, 10],
    'min7': [0, 3, 7, 10],
    'm7b5': [0, 3, 6, 10],
}

_MAJOR_SCALE   = [0, 2, 4, 5, 7, 9, 11]
_MINOR_SCALE   = [0, 2, 3, 5, 7, 8, 10]
_MAJOR_QUALITY = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim']
_MINOR_QUALITY = ['min', 'dim', 'maj', 'min', 'min', 'maj', 'maj']
_SEVENTH_MAP   = {'maj': 'maj7', 'min': 'min7', 'dim': 'm7b5'}

_WINDOW_SIZE_SEC = 0.5  # Fixed time grid resolution
_SELF_TRANSITION = 0.95 # HMM self-loop prior (increased to hold chords longer)


# ---------------------------------------------------------------------------
# KEY PARSING & CANDIDATES
# ---------------------------------------------------------------------------

def _parse_key(key_str: str):
    if not key_str or key_str.strip().lower() in ('unknown', ''):
        return 0, True
    parts = key_str.strip().split()
    root_name = _ENHARMONIC.get(parts[0], parts[0])
    try:
        root_pc = PITCH_CLASSES.index(root_name)
    except ValueError:
        root_pc = 0
    is_major = len(parts) < 2 or 'minor' not in parts[1].lower()
    return root_pc, is_major


def _make_template_vector(root_pc: int, offsets: list) -> np.ndarray:
    vec = np.zeros(12)
    for off in offsets:
        vec[(root_pc + off) % 12] = 1.0
    if len(offsets) >= 2:
        fifth_pc = (root_pc + 7) % 12
        if vec[fifth_pc] > 0:
            vec[fifth_pc] = 0.7
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _build_candidates(key_str: str):
    root_pc, is_major = _parse_key(key_str)
    scale     = _MAJOR_SCALE   if is_major else _MINOR_SCALE
    qualities = _MAJOR_QUALITY if is_major else _MINOR_QUALITY

    candidates = []
    seen_labels = set()

    def _add(chord_root, quality, prior):
        label = PITCH_CLASSES[chord_root]
        if quality != 'maj':
            label += quality
        if label in seen_labels:
            return
        seen_labels.add(label)
        vec = _make_template_vector(chord_root, _CHORD_TEMPLATES[quality])
        candidates.append((label, chord_root, vec, prior))

    # Diatonic triads & 7ths
    for interval, quality in zip(scale, qualities):
        chord_root = (root_pc + interval) % 12
        _add(chord_root, quality, prior=0.1)  # Boost diatonic triads
        q7 = _SEVENTH_MAP.get(quality)
        if q7:
            _add(chord_root, q7, prior=-0.1)  # Penalize 7ths (simplicity bias)

    # Secondary dominants
    for interval in scale:
        sd_root = (root_pc + interval - 7) % 12
        _add(sd_root, '7', prior=-0.2)

    # All other major/minor triads as fallbacks (penalized slightly vs diatonic)
    for c_root in range(12):
        _add(c_root, 'maj', prior=-0.1)
        _add(c_root, 'min', prior=-0.1)

    return candidates


# ---------------------------------------------------------------------------
# SYMBOLIC GRID WINDOWING
# ---------------------------------------------------------------------------

def _parse_timestamp(ts_str):
    parts = ts_str.replace("s", "").split("-")
    return float(parts[0].strip()), float(parts[1].strip())


def _build_symbolic_windows(detected_notes, window_size_sec=_WINDOW_SIZE_SEC):
    max_end = 0.0
    valid_notes = []
    
    for ev in detected_notes:
        if ev.get("note") == "REST" or not ev.get("midi_notes"):
            continue
        try:
            start, end = _parse_timestamp(ev["timestamp"])
            max_end = max(max_end, end)
            valid_notes.append({
                "start": start,
                "end": end,
                "midi_notes": ev["midi_notes"],
                "velocities": ev.get("velocities", [0.7] * len(ev["midi_notes"]))
            })
        except:
            pass

    if not valid_notes or max_end == 0:
        return []

    n_windows = int(np.ceil(max_end / window_size_sec))
    windows = [{"start": i * window_size_sec, 
                "end": (i + 1) * window_size_sec, 
                "pcp": np.zeros(12), 
                "bass_midi": 128} for i in range(n_windows)]

    # Distribute notes into windows
    for note_ev in valid_notes:
        start_idx = int(note_ev["start"] // window_size_sec)
        end_idx   = int(note_ev["end"] // window_size_sec)
        
        for w_idx in range(start_idx, min(end_idx + 1, n_windows)):
            w = windows[w_idx]
            for midi, vel in zip(note_ev["midi_notes"], note_ev["velocities"]):
                w["pcp"][midi % 12] += vel
                if midi < w["bass_midi"]:
                    w["bass_midi"] = midi

    return windows


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def _score_windows(windows, candidates):
    results = []
    for w in windows:
        pcp = w["pcp"]
        bass_pc = w["bass_midi"] % 12 if w["bass_midi"] < 128 else -1
        
        norm = np.linalg.norm(pcp)
        if norm > 0:
            pcp = pcp / norm
            
        scores = {}
        for label, root_pc, tmpl, prior in candidates:
            # Base cosine similarity + prior bias
            score = float(np.dot(pcp, tmpl)) + prior
            
            # Bass Heuristics
            if bass_pc != -1:
                if bass_pc == root_pc:
                    score += 0.3  # Strong bonus for root inversion
                elif tmpl[bass_pc] > 0:
                    score += 0.1  # Slight bonus for valid inversion (e.g. 3rd in bass)
                else:
                    score -= 0.1  # Relaxed penalty for clashing bass note (was -0.5)
                    
            scores[label] = score
            
        best_label = max(scores, key=scores.__getitem__) if scores else "N/A"
        results.append((w["start"], w["end"], best_label, scores))
        
    return results


# ---------------------------------------------------------------------------
# VITERBI HMM
# ---------------------------------------------------------------------------

def _viterbi_smooth(raw_results, candidates):
    if not raw_results:
        return []

    chord_labels = [c[0] for c in candidates]
    n_chords = len(chord_labels)
    T = len(raw_results)

    STAY = _SELF_TRANSITION
    MOVE = (1.0 - STAY) / max(1, n_chords - 1)

    log_stay = np.log(STAY + 1e-12)
    log_move = np.log(MOVE + 1e-12)

    log_emit = np.zeros((n_chords, T))
    for t, (_, _, _, scores) in enumerate(raw_results):
        score_vec  = np.array([scores.get(lbl, 0.0) for lbl in chord_labels])
        score_vec  = score_vec * 15.0  # sharpen
        score_vec -= score_vec.max()
        prob_vec   = np.exp(score_vec)
        prob_vec  /= (prob_vec.sum() + 1e-12)
        log_emit[:, t] = np.log(prob_vec + 1e-12)

    viterbi = np.full((n_chords, T), -np.inf)
    backptr = np.zeros((n_chords, T), dtype=int)

    viterbi[:, 0] = np.log(1.0 / n_chords) + log_emit[:, 0]

    for t in range(1, T):
        for s in range(n_chords):
            trans = viterbi[:, t - 1] + np.where(
                np.arange(n_chords) == s, log_stay, log_move
            )
            best_prev     = int(np.argmax(trans))
            viterbi[s, t] = trans[best_prev] + log_emit[s, t]
            backptr[s, t] = best_prev

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(viterbi[:, -1]))
    for t in range(T - 2, -1, -1):
        path[t] = backptr[path[t + 1], t + 1]

    return [chord_labels[path[t]] for t in range(T)]


def _merge_chord_sequence(raw_results, smooth_labels):
    if not smooth_labels:
        return []

    merged = []
    current_label = smooth_labels[0]
    current_start = raw_results[0][0]
    current_end   = raw_results[0][1]

    for i in range(1, len(smooth_labels)):
        ws, we = raw_results[i][0], raw_results[i][1]
        if smooth_labels[i] == current_label:
            current_end = we
        else:
            merged.append({'start': round(current_start, 3),
                           'end':   round(current_end,   3),
                           'chord': current_label})
            current_label = smooth_labels[i]
            current_start = ws
            current_end   = we

    merged.append({'start': round(current_start, 3),
                   'end':   round(current_end,   3),
                   'chord': current_label})
    return merged


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def detect_chords_from_notes(detected_notes, musical_key: str = 'C Major'):
    """
    Main entry point for the Symbolic Grid Hybrid approach.
    """
    try:
        windows = _build_symbolic_windows(detected_notes)
        if not windows:
            return []

        candidates = _build_candidates(musical_key)
        if not candidates:
            return []

        raw_results   = _score_windows(windows, candidates)
        smooth_labels = _viterbi_smooth(raw_results, candidates)
        return _merge_chord_sequence(raw_results, smooth_labels)

    except Exception as exc:
        print(f'[note_chord_detector] Error: {exc}')
        import traceback
        traceback.print_exc()
        return []

def detect_chords_from_audio(audio_path: str, musical_key: str = 'C Major', beats_per_window: int = 2):
    """Deprecated shim for audio-based fallback."""
    return []
