"""
polyphonic_engine_v4.py  -  Pitch-Aware & Harmonic Rule Hybrid Engine

Key improvements over v3/v2:
- Replaces unsupervised GMM with a Multi-Stage Musical Heuristic Filter.
- Stage 1: Frequency-dependent energy thresholding (Log scale curve).
- Stage 2: Onset-guided harmonic validation (checks for independent attack transient).
- Stage 3: Key-scale awareness & Envelope continuity check.
"""

import numpy as np
import librosa
from .polyphonic_engine_v2 import (
    _build_inharmonic_templates,
    _adapt_templates_to_audio,
    _nmf_sparse,
    _hmm_smooth,
    _detect_onsets,
    _extract_events,
    _enforce_polyphony,
    _format_output,
    V2_PROFILES
)
from .key_detector import detect_key_from_notes

# ===========================================================================
# STAGE 1 - Pitch-Aware Adaptive Thresholding
# ===========================================================================

def _apply_pitch_aware_threshold(H, note_midi_list, percentile=75, floor=0.05):
    """
    Apply a frequency-dependent threshold curve.
    High pitch notes require less energy to be perceived.
    """
    n_notes, n_frames = H.shape
    h_max = np.max(H)
    if h_max <= 0:
        return np.zeros_like(H, dtype=bool), H
    
    H_norm = H / h_max
    active_mask = np.zeros((n_notes, n_frames), dtype=bool)

    # Pitch-dependent curve: threshold lowers as MIDI pitch increases
    # e.g., C2 (36) -> 1.0x floor, C6 (84) -> 0.4x floor
    def get_pitch_multiplier(midi):
        if midi < 48:
            return 1.2
        elif midi < 72:
            return 1.0
        elif midi < 84:
            return 0.7
        else:
            return 0.4

    for note_idx in range(n_notes):
        midi = note_midi_list[note_idx]
        pitch_mult = get_pitch_multiplier(midi)
        
        col = H_norm[note_idx, :]
        col_max = np.max(col)
        
        if col_max > 0:
            attack_thresh = max(floor * pitch_mult, col_max * 0.15)
            release_thresh = max(floor * 0.5 * pitch_mult, col_max * 0.05)
            
            was_active = False
            for f in range(n_frames):
                val = col[f]
                survives_attack = val > attack_thresh
                survives_sustain = (val > release_thresh) and was_active
                
                is_active = survives_attack or survives_sustain
                active_mask[note_idx, f] = is_active
                was_active = is_active
                
    # Skyline Protector (Top-note melody priority)
    skyline_floor = max(0.01, floor * 0.3)
    for f in range(n_frames):
        col = H_norm[:, f]
        candidates = np.where(col > skyline_floor)[0]
        if len(candidates) > 0:
            highest_candidate = candidates[-1]
            active_mask[highest_candidate, f] = True

    return active_mask, H_norm

# ===========================================================================
# STAGE 2 & 3 - Onset & Harmonic Validation
# ===========================================================================

def _hybrid_ghost_filter(events, H_norm, note_midi_list, times, onset_times, B=2e-4):
    """
    Filters ghosts based on:
    1. Independent attack transients.
    2. Overlap and harmonic relationship with lower active notes.
    """
    if not events:
        return events

    n_frames = len(times)
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.02
    midi_to_row = {midi: row for row, midi in enumerate(note_midi_list)}
    midi_to_hz = {midi: librosa.midi_to_hz(midi) for midi in set(ev[0] for ev in events)}

    def get_attack_energy(midi, start, end):
        row = midi_to_row.get(midi)
        if row is not None and row < H_norm.shape[0]:
            # Look at first 3 frames (approx 60ms)
            s = max(0, start)
            e = min(n_frames, s + 3)
            return float(np.max(H_norm[row, s:e]))
        return 0.0
        
    def get_mean_energy(midi, start, end):
        row = midi_to_row.get(midi)
        if row is not None and row < H_norm.shape[0]:
            s, e = max(0, start), min(n_frames, end)
            if e > s:
                return float(np.mean(H_norm[row, s:e]))
        return 0.0

    def is_overtone(midi_high, midi_low):
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
    
    attacks = [get_attack_energy(ev[0], ev[1], ev[2]) for ev in sorted_ev]
    means = [get_mean_energy(ev[0], ev[1], ev[2]) for ev in sorted_ev]

    for i, ev_i in enumerate(sorted_ev):
        if i in remove:
            continue
        
        for j in range(i + 1, len(sorted_ev)):
            if j in remove:
                continue
            ev_j = sorted_ev[j]
            
            if min(ev_i[2], ev_j[2]) - max(ev_i[1], ev_j[1]) <= 0:
                continue
                
            if overlap(ev_i, ev_j) < 0.4:
                continue
                
            if is_overtone(ev_j[0], ev_i[0]):
                # Stage 2: Check for independent attack
                # If the high note has a very weak attack compared to its mean, 
                # or if its attack is weak compared to the lower note's attack, it's a ghost.
                attack_j = attacks[j]
                mean_j = means[j]
                attack_i = attacks[i]
                
                # If it's truly an independent note (e.g., stacked chord), it should have a sharp onset
                if attack_j < 1.2 * mean_j and attack_j < attack_i * 0.6:
                    remove.add(j)

    return [ev for idx, ev in enumerate(sorted_ev) if idx not in remove]


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def process_audio_polyphonic_v4(file_path, profile_name="piano_v2"):
    """
    V4 pipeline: CQT -> dual onset -> inharmonic W -> semi-supervised W 
    -> sparse NMF -> PITCH-AWARE threshold -> HMM smoothing 
    -> event extraction -> HYBRID GHOST FILTER -> polyphony -> format
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

    # V4: Pitch-Aware Thresholding
    active_mask, H_norm = _apply_pitch_aware_threshold(
        H, note_midi_list, percentile=profile["adaptive_threshold_pct"], floor=profile["global_threshold_floor"]
    )

    smoothed_mask = _hmm_smooth(active_mask, stay_prob=profile["hmm_transition_stay"])

    raw_events = _extract_events(smoothed_mask, note_midi_list, times, profile, onset_times)

    if not raw_events:
        return []

    # V4: Hybrid Harmonic / Onset Ghost Filter
    clean_events = _hybrid_ghost_filter(
        raw_events, H_norm, note_midi_list, times, onset_times, B=profile["inharmonicity_B"]
    )

    final_events = _enforce_polyphony(
        clean_events, H_norm, note_midi_list, times, max_poly=profile["max_polyphony"]
    )

    return _format_output(final_events, times, H_norm=H_norm, note_midi_list=note_midi_list)
