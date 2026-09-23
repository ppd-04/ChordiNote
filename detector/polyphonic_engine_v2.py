"""
polyphonic_engine_v2.py  -  Enhanced CQT + NMF Transcription Engine

Key improvements over v1:

1. LEARNED W TEMPLATES (Semi-Supervised NMF)
   Instead of pure math sine-wave templates, we compute pitch-accurate
   spectral fingerprints directly from the target audio, then freeze W.

2. INHARMONICITY-CORRECTED TEMPLATES
   Real piano strings are stiff metal - their overtones are slightly sharp.
   We model this with a stretch coefficient B (Fletcher 1964 inharmonicity model).

3. OCTAVE-DEPENDENT HARMONICS
   Low notes (C2) get 14 harmonics, high notes (C6+) get only 2-3.
   Prevents high-octave templates bleeding energy across octaves.

4. ADAPTIVE DYNAMIC THRESHOLDING (Local Percentile)
   Per-frame threshold based on local activation percentiles.
   Loud frames auto-raise threshold (suppress ghosts), quiet frames lower it.

5. L1 SPARSITY PENALTY IN NMF
   Adds lambda in NMF denominator, forcing fewer active notes per frame.
   Most mathematically principled ghost suppression method.

6. HMM-BASED NOTE SMOOTHING (2-state Viterbi)
   Eliminates single-frame ghost blips and brief silence holes in real notes.

7. DUAL ONSET DETECTION (energy + spectral flux combined)
   Better detection of breath attacks (flute) and hammer strikes (piano).
"""

import numpy as np
import librosa


# ===========================================================================
# PROFILES FOR V2
# ===========================================================================

V2_PROFILES = {
    "piano_v2": {
        "label": "Piano V2 - Learned Templates (Best for Piano Covers)",
        "fmin": "A1",
        "fmax": "C8",
        "hop_length": 512,
        "bins_per_octave": 36,
        "nmf_iterations": 100,
        "l1_lambda": 0.03,
        "inharmonicity_B": 2e-4,
        "adaptive_threshold_pct": 75,
        "global_threshold_floor": 0.05,
        "min_note_duration": 0.06,
        "min_rest_duration": 0.08,
        "max_polyphony": 6,
        "hmm_transition_stay": 0.90,
        "onset_merge_ms": 40,
    },
    "flute_v2": {
        "label": "Flute/Wind V2 - Learned Templates (Best for Flute/Woodwind Covers)",
        "fmin": "C4",
        "fmax": "C8",
        "hop_length": 256,
        "bins_per_octave": 48,
        "nmf_iterations": 80,
        "l1_lambda": 0.05,
        "inharmonicity_B": 5e-6,
        "adaptive_threshold_pct": 70,
        "global_threshold_floor": 0.08,
        "min_note_duration": 0.05,
        "min_rest_duration": 0.06,
        "max_polyphony": 2,
        "hmm_transition_stay": 0.88,
        "onset_merge_ms": 30,
    },
    "guitar_v2": {
        "label": "Guitar V2 - Learned Templates (Best for Acoustic Guitar Covers)",
        "fmin": "E2",
        "fmax": "E6",
        "hop_length": 512,
        "bins_per_octave": 36,
        "nmf_iterations": 100,
        "l1_lambda": 0.025,
        "inharmonicity_B": 4e-4,
        "adaptive_threshold_pct": 72,
        "global_threshold_floor": 0.06,
        "min_note_duration": 0.07,
        "min_rest_duration": 0.08,
        "max_polyphony": 6,
        "hmm_transition_stay": 0.88,
        "onset_merge_ms": 35,
    },
}


# ===========================================================================
# STEP 1 - OCTAVE-DEPENDENT HARMONICS
# ===========================================================================

def _n_harmonics_for_midi(midi_note):
    """Return number of harmonics based on pitch register."""
    if midi_note <= 48:    # C3 and below
        return 14
    elif midi_note <= 60:  # C3-C4
        return 10
    elif midi_note <= 72:  # C4-C5
        return 7
    elif midi_note <= 84:  # C5-C6
        return 4
    else:                  # Above C6
        return 2


# ===========================================================================
# STEP 2 - INHARMONIC TEMPLATE MATRIX (Physics-corrected)
# ===========================================================================

def _build_inharmonic_templates(n_bins, fmin_hz, bins_per_octave, B=2e-4):
    """
    Build physically-accurate harmonic template matrix W.

    Inharmonicity model (Fletcher 1964):
        f_n = n * f0 * sqrt(1 + B * n^2)

    Amplitude roll-off: A_n proportional to 1 / n^0.75
    """
    fmin_midi = int(np.floor(librosa.hz_to_midi(fmin_hz)))
    fmax_midi = int(np.ceil(fmin_midi + n_bins * (12.0 / bins_per_octave)))
    fmax_midi = min(fmax_midi, 108)

    note_midi_list = list(range(fmin_midi, fmax_midi))
    n_notes = len(note_midi_list)

    W = np.zeros((n_bins, n_notes))

    for note_idx, midi_note in enumerate(note_midi_list):
        f0 = librosa.midi_to_hz(midi_note)
        n_harmonics = _n_harmonics_for_midi(midi_note)

        for h in range(1, n_harmonics + 1):
            # Inharmonic frequency with physical stretch factor
            f_h = h * f0 * np.sqrt(1.0 + B * h * h)

            if f_h < fmin_hz:
                continue

            amp = 1.0 / (h ** 0.75)
            bin_float = bins_per_octave * np.log2(f_h / fmin_hz)
            bin_center = int(round(bin_float))

            if bin_center >= n_bins:
                break

            for offset, spread in [(-1, 0.25), (0, 1.0), (1, 0.25)]:
                b = bin_center + offset
                if 0 <= b < n_bins:
                    W[b, note_idx] += amp * spread

    col_norms = np.linalg.norm(W, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1.0
    W = W / col_norms

    return W, note_midi_list


# ===========================================================================
# STEP 3 - SEMI-SUPERVISED W ADAPTATION
# ===========================================================================

def _adapt_templates_to_audio(V, W_init, n_iter_adapt=20):
    """
    Let W drift by up to 10% from physics-based initialization to match
    the specific recording's timbre (room, mic, instrument EQ).
    """
    n_bins, n_notes = W_init.shape
    n_frames = V.shape[1]
    eps = 1e-10

    W = W_init.copy()
    H = np.random.rand(n_notes, n_frames) * 0.1 + 0.01

    for _ in range(n_iter_adapt):
        # Update H
        WtV = W.T @ V
        WtWH = (W.T @ W) @ H
        H *= WtV / (WtWH + eps)

        # Update W gently (10% drift max)
        VHt = V @ H.T
        WHHt = W @ (H @ H.T)
        W_new = W * (VHt / (WHHt + eps))
        W = 0.90 * W + 0.10 * W_new

    col_norms = np.linalg.norm(W, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1.0
    W = W / col_norms

    return W


# ===========================================================================
# STEP 4 - SPARSE NMF WITH L1 PENALTY
# ===========================================================================

def _nmf_sparse(V, W, n_iterations=100, l1_lambda=0.03):
    """
    NMF with L1 sparsity: H update denominator includes lambda,
    forcing weak ghost activations toward zero.
    """
    n_notes = W.shape[1]
    n_frames = V.shape[1]
    eps = 1e-10

    np.random.seed(42)
    H = np.random.rand(n_notes, n_frames) * 0.1 + 0.01

    Wt = W.T
    WtW = Wt @ W

    for _ in range(n_iterations):
        numerator = Wt @ V
        denominator = WtW @ H + l1_lambda + eps
        H *= numerator / denominator

    return H


# ===========================================================================
# STEP 5 - ADAPTIVE FRAME-WISE THRESHOLDING
# ===========================================================================

def _apply_adaptive_threshold(H, percentile=75, floor=0.05):
    """
    Per-frame threshold based on local activation peak, using Hysteresis 
    (dual-thresholds for legato) and Skyline Melody Protection.
    Returns binary active_mask and globally-normalized H_norm.
    """
    n_notes, n_frames = H.shape

    h_max = np.max(H)
    if h_max <= 0:
        return np.zeros_like(H, dtype=bool), H
    H_norm = H / h_max

    active_mask = np.zeros((n_notes, n_frames), dtype=bool)
    
    # 1. Hysteresis Thresholding (Legato Sustain)
    # Require a strong peak to START a note, but a lower threshold to KEEP it alive
    for f in range(n_frames):
        col = H_norm[:, f]
        col_max = np.max(col)
        if col_max > 0:
            attack_thresh = max(floor, col_max * 0.15)
            release_thresh = max(floor * 0.5, col_max * 0.05)
            
            if f == 0:
                active_mask[:, f] = col > attack_thresh
            else:
                was_active = active_mask[:, f-1]
                survives_attack = col > attack_thresh
                survives_sustain = (col > release_thresh) & was_active
                active_mask[:, f] = survives_attack | survives_sustain

    # 2. Skyline Protector (Top-note melody priority)
    # The highest pitch active is almost always the melody. Ensure it doesn't get 
    # chopped off if it's weaker than thick bass chords.
    skyline_floor = max(0.01, floor * 0.4)
    for f in range(n_frames):
        col = H_norm[:, f]
        candidates = np.where(col > skyline_floor)[0]
        if len(candidates) > 0:
            highest_candidate = candidates[-1]
            active_mask[highest_candidate, f] = True

    return active_mask, H_norm


# ===========================================================================
# STEP 6 - HMM SMOOTHING (2-state Viterbi per note)
# ===========================================================================

def _hmm_smooth(active_mask, stay_prob=0.90):
    """
    2-state HMM (Off/On) per note using Viterbi decoding.
    Removes single-frame ghost blips and fills tiny silence holes.
    """
    n_notes, n_frames = active_mask.shape
    smoothed = np.zeros_like(active_mask)

    log_trans = np.log(np.array([
        [stay_prob,       1 - stay_prob],
        [1 - stay_prob,   stay_prob    ],
    ]) + 1e-10)

    log_emit = np.log(np.array([
        [0.90, 0.10],
        [0.10, 0.90],
    ]) + 1e-10)

    for note_idx in range(n_notes):
        obs = active_mask[note_idx, :].astype(int)

        viterbi = np.full((2, n_frames), -np.inf)
        backptr = np.zeros((2, n_frames), dtype=int)

        viterbi[:, 0] = np.log(0.5) + log_emit[:, obs[0]]

        for t in range(1, n_frames):
            for s in range(2):
                scores = viterbi[:, t-1] + log_trans[:, s] + log_emit[s, obs[t]]
                best = np.argmax(scores)
                viterbi[s, t] = scores[best]
                backptr[s, t] = best

        path = np.zeros(n_frames, dtype=int)
        path[-1] = np.argmax(viterbi[:, -1])
        for t in range(n_frames - 2, -1, -1):
            path[t] = backptr[path[t + 1], t + 1]

        smoothed[note_idx, :] = path.astype(bool)

    return smoothed


# ===========================================================================
# STEP 7 - DUAL ONSET DETECTION
# ===========================================================================

def _detect_onsets(y, sr, hop_length, merge_ms=40):
    """Combine energy + spectral flux onset detectors and de-duplicate."""
    onset_1 = librosa.frames_to_time(
        librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True, units='frames'),
        sr=sr, hop_length=hop_length
    )
    flux_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median, fmax=8000)
    onset_2 = librosa.frames_to_time(
        librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, onset_envelope=flux_env, backtrack=True, units='frames'),
        sr=sr, hop_length=hop_length
    )

    all_onsets = np.sort(np.concatenate([onset_1, onset_2]))
    if len(all_onsets) == 0:
        return np.array([])

    merge_sec = merge_ms / 1000.0
    keep = [True]
    for i in range(1, len(all_onsets)):
        keep.append(all_onsets[i] - all_onsets[i - 1] >= merge_sec)

    return all_onsets[np.array(keep)]


# ===========================================================================
# STEP 8 - EVENTS FROM SMOOTHED MASK + ONSET SPLITTING
# ===========================================================================

def _extract_events(smoothed_mask, note_midi_list, times, profile, onset_times):
    """Convert smoothed binary mask to note event tuples, splitting at onsets."""
    n_notes, n_frames = smoothed_mask.shape
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.02
    min_frames = max(1, int(profile["min_note_duration"] / frame_duration))

    raw_events = []
    for note_idx in range(n_notes):
        in_note = False
        start_frame = 0
        for f in range(n_frames):
            if smoothed_mask[note_idx, f] and not in_note:
                in_note = True
                start_frame = f
            elif not smoothed_mask[note_idx, f] and in_note:
                in_note = False
                if f - start_frame >= min_frames:
                    raw_events.append((note_midi_list[note_idx], start_frame, f))
        if in_note and n_frames - start_frame >= min_frames:
            raw_events.append((note_midi_list[note_idx], start_frame, n_frames))

    # Onset-based repeated-note splitting
    if onset_times is not None and len(onset_times) > 0:
        split_events = []
        margin = 0.03
        for midi, start, end in raw_events:
            t_start = times[min(start, len(times) - 1)]
            t_end = times[min(end - 1, len(times) - 1)] + frame_duration
            internal = [ot for ot in onset_times if t_start + margin < ot < t_end - margin]
            if not internal:
                split_events.append((midi, start, end))
            else:
                pts = [start]
                for ot in internal:
                    of = int(np.argmin(np.abs(times - ot)))
                    if start < of < end:
                        pts.append(of)
                pts.append(end)
                for k in range(len(pts) - 1):
                    if pts[k+1] - pts[k] >= min_frames:
                        split_events.append((midi, pts[k], pts[k+1]))
        raw_events = split_events

    # Merge tiny same-pitch gaps (< 30ms)
    min_gap = max(1, int(0.03 / frame_duration))
    raw_events.sort(key=lambda x: (x[1], x[0]))
    merged = []
    for midi, start, end in raw_events:
        if merged and merged[-1][0] == midi and 0 < (start - merged[-1][2]) <= min_gap:
            merged[-1] = (midi, merged[-1][1], end)
        else:
            merged.append((midi, start, end))

    return merged


# ===========================================================================
# STEP 9 - HARMONIC GHOST REMOVAL (inharmonicity-aware)
# ===========================================================================

def _remove_harmonic_ghosts(events, H_norm, note_midi_list, times, B=2e-4):
    """Remove ghost notes that are inharmonic overtones of stronger lower notes."""
    if not events:
        return events

    n_frames = len(times)
    midi_to_row = {midi: row for row, midi in enumerate(note_midi_list)}
    
    # Precompute frequencies for all MIDI notes used to avoid librosa.midi_to_hz in a tight loop
    midi_to_hz = {midi: librosa.midi_to_hz(midi) for midi in set(ev[0] for ev in events)}

    def strength(midi, start, end):
        row = midi_to_row.get(midi)
        if row is not None and row < H_norm.shape[0]:
            s, e = max(0, start), min(n_frames, end)
            if e > s:
                return float(np.mean(H_norm[row, s:e]))
        return 0.0

    def is_overtone(midi_high, midi_low):
        # Quick heuristic filter: an overtone is at least 12 semitones up, at most ~48
        diff = midi_high - midi_low
        if diff < 11 or diff > 49:
            return False
            
        f0 = midi_to_hz[midi_low]
        f_high = midi_to_hz[midi_high]
        for n in range(2, 10):
            f_n = n * f0 * np.sqrt(1 + B * n * n)
            if abs(f_high - f_n) / f_n < 0.04:
                return True
        return False

    def overlap(ev1, ev2):
        ov = max(0, min(ev1[2], ev2[2]) - max(ev1[1], ev2[1]))
        shorter = min(ev1[2] - ev1[1], ev2[2] - ev2[1])
        return ov / shorter if shorter > 0 else 0.0

    sorted_ev = sorted(events, key=lambda x: x[0])
    remove = set()
    
    # Precalculate strengths for all events
    strengths = [strength(ev[0], ev[1], ev[2]) for ev in sorted_ev]

    for i, ev_i in enumerate(sorted_ev):
        if i in remove:
            continue
        str_i = strengths[i]
        
        for j in range(i + 1, len(sorted_ev)):
            if j in remove:
                continue
            ev_j = sorted_ev[j]
            
            # Quick overlap check (avoid function call)
            if min(ev_i[2], ev_j[2]) - max(ev_i[1], ev_j[1]) <= 0:
                continue
                
            if overlap(ev_i, ev_j) < 0.3:
                continue
                
            if is_overtone(ev_j[0], ev_i[0]):
                str_j = strengths[j]
                if str_j < str_i * 0.70:
                    remove.add(j)
                elif str_j < str_i * 1.1:
                    ratio = midi_to_hz[ev_j[0]] / midi_to_hz[ev_i[0]]
                    if ratio > 3.0:
                        remove.add(j)

    return [ev for idx, ev in enumerate(sorted_ev) if idx not in remove]


# ===========================================================================
# STEP 10 - MAX POLYPHONY ENFORCEMENT
# ===========================================================================

def _enforce_polyphony(events, H_norm, note_midi_list, times, max_poly=6):
    """Remove weakest notes at frames where polyphony exceeds max_poly."""
    if not events:
        return events

    n_frames = len(times)
    midi_to_row = {midi: row for row, midi in enumerate(note_midi_list)}

    def strength(midi, start, end):
        row = midi_to_row.get(midi)
        if row is not None and row < H_norm.shape[0]:
            s, e = max(0, start), min(n_frames, end)
            if e > s:
                return float(np.mean(H_norm[row, s:e]))
        return 0.0

    event_str = [strength(*ev) for ev in events]
    active_at = [[] for _ in range(n_frames)]
    for idx, (m, s, e) in enumerate(events):
        for f in range(max(0, s), min(n_frames, e)):
            active_at[f].append(idx)

    votes = np.zeros(len(events))
    for f in range(n_frames):
        active = active_at[f]
        if len(active) <= max_poly:
            continue
        ranked = sorted(active, key=lambda i: event_str[i], reverse=True)
        for rank, idx in enumerate(ranked):
            if rank >= max_poly:
                votes[idx] += 1

    return [ev for idx, ev in enumerate(events) if votes[idx] / max(1, ev[2] - ev[1]) < 0.30]


# ===========================================================================
# STEP 11 - FORMAT OUTPUT (identical to v1 format for full compatibility)
# ===========================================================================

def _format_output(events, times):
    """Format raw events to output dicts identical to v1 polyphonic_engine."""
    if not events:
        return []

    frame_duration = times[1] - times[0] if len(times) > 1 else 0.02
    events.sort(key=lambda x: (x[1], x[0]))

    used = set()
    note_groups = []

    for i, (midi_i, start_i, end_i) in enumerate(events):
        if i in used:
            continue
        group = [(midi_i, start_i, end_i)]
        used.add(i)
        for j in range(i + 1, len(events)):
            if j in used:
                continue
            midi_j, start_j, end_j = events[j]
            ov = max(0, min(end_i, end_j) - max(start_i, start_j))
            if ov * frame_duration >= 0.05:
                group.append((midi_j, start_j, end_j))
                used.add(j)
        note_groups.append(group)

    out = []
    last_end = 0.0

    for group in note_groups:
        midi_notes = sorted(set(g[0] for g in group))
        start_t = times[min(min(g[1] for g in group), len(times) - 1)]
        end_t = times[min(max(g[2] for g in group), len(times) - 1)] + frame_duration

        if start_t - last_end > 0.15:
            out.append({"timestamp": f"{last_end:.2f}s - {start_t:.2f}s", "note": "REST", "frequency": "-"})

        names = [librosa.midi_to_note(m) for m in midi_notes]
        freqs = [librosa.midi_to_hz(m) for m in midi_notes]

        if len(midi_notes) > 1:
            out.append({
                "timestamp": f"{start_t:.2f}s - {end_t:.2f}s",
                "note": " + ".join(names),
                "frequency": " + ".join(f"{f:.1f} Hz" for f in freqs),
                "is_chord": True,
                "midi_notes": midi_notes,
            })
        else:
            out.append({
                "timestamp": f"{start_t:.2f}s - {end_t:.2f}s",
                "note": names[0],
                "frequency": f"{freqs[0]:.1f} Hz",
                "is_chord": False,
                "midi_notes": midi_notes,
            })

        last_end = end_t

    return out


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def process_audio_polyphonic_v2(file_path, profile_name="piano_v2"):
    """
    V2 pipeline: CQT -> dual onset detection -> inharmonic W (physics)
    -> semi-supervised W adaptation -> sparse NMF (L1) -> adaptive threshold
    -> HMM smoothing -> event extraction + onset splitting
    -> harmonic ghost removal -> polyphony enforcement -> format output
    """
    profile = V2_PROFILES.get(profile_name, V2_PROFILES["piano_v2"])

    SR = 22050
    y, sr = librosa.load(file_path, sr=SR, mono=True)

    fmin_hz = librosa.note_to_hz(profile["fmin"])
    fmax_hz = librosa.note_to_hz(profile["fmax"])
    bpo = profile["bins_per_octave"]
    hop = profile["hop_length"]
    n_bins = int(np.ceil(bpo * np.log2(fmax_hz / fmin_hz)))

    C = np.abs(librosa.cqt(
        y, sr=sr, hop_length=hop, fmin=fmin_hz,
        n_bins=n_bins, bins_per_octave=bpo, filter_scale=1.0
    ))
    times = librosa.times_like(C, sr=sr, hop_length=hop)

    onset_times = _detect_onsets(y, sr, hop, merge_ms=profile["onset_merge_ms"])

    W_init, note_midi_list = _build_inharmonic_templates(
        n_bins, fmin_hz, bpo, B=profile["inharmonicity_B"]
    )
    W = _adapt_templates_to_audio(C, W_init, n_iter_adapt=20)

    H = _nmf_sparse(C, W, n_iterations=profile["nmf_iterations"], l1_lambda=profile["l1_lambda"])

    active_mask, H_norm = _apply_adaptive_threshold(
        H, percentile=profile["adaptive_threshold_pct"], floor=profile["global_threshold_floor"]
    )

    smoothed_mask = _hmm_smooth(active_mask, stay_prob=profile["hmm_transition_stay"])

    raw_events = _extract_events(smoothed_mask, note_midi_list, times, profile, onset_times)

    if not raw_events:
        return []

    clean_events = _remove_harmonic_ghosts(
        raw_events, H_norm, note_midi_list, times, B=profile["inharmonicity_B"]
    )

    final_events = _enforce_polyphony(
        clean_events, H_norm, note_midi_list, times, max_poly=profile["max_polyphony"]
    )

    return _format_output(final_events, times)


def get_v2_profile_display_info(profile_name):
    """Helper for the UI."""
    profile = V2_PROFILES.get(profile_name, V2_PROFILES["piano_v2"])
    return {"label": profile["label"], "fmin": profile["fmin"], "fmax": profile["fmax"]}
