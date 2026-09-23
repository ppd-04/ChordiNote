import numpy as np
import librosa


POLYPHONIC_PROFILES = {
    "piano_poly": {
        "label": "Piano Polyphonic (Best for chords & bass)",
        "fmin": "A2",          # 55 Hz - covers low bass notes
        "fmax": "C7",          # 2093 Hz - covers most piano range
        "hop_length": 512,
        "bins_per_octave": 36, # 3 bins per semitone for high resolution
        "nmf_iterations": 80,  # More iterations = more accurate but slower
        "activation_threshold": 0.20,  # Min activation to count as a note
        "min_note_duration": 0.06,     # Ignore notes shorter than 60ms
        "min_rest_duration": 0.08,     # Merge notes separated by tiny gaps
        "n_harmonics": 6,      # How many harmonics per note template
    },
    "general_poly": {
        "label": "General Polyphonic",
        "fmin": "C2",
        "fmax": "C7",
        "hop_length": 512,
        "bins_per_octave": 36,
        "nmf_iterations": 60,
        "activation_threshold": 0.2,
        "min_note_duration": 0.08,
        "min_rest_duration": 0.1,
        "n_harmonics": 5,
    },
}


def _build_harmonic_template_matrix(n_bins, fmin_hz, bins_per_octave, n_harmonics=6):

    # V = W*H ekhane W matrix ta banabo
    fmin_midi = int(np.floor(librosa.hz_to_midi(fmin_hz)))
    fmax_midi = int(np.ceil(fmin_midi + n_bins * (12.0 / bins_per_octave)))
    fmax_midi = min(fmax_midi, 108)  # C8 e cap

    note_midi_list = list(range(fmin_midi, fmax_midi))
    n_notes = len(note_midi_list)

    # Initialize template matrix with zeros
    W = np.zeros((n_bins, n_notes))


    harmonic_amplitudes = [1.0, 0.50, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02]

    for note_idx, midi_note in enumerate(note_midi_list):
        fundamental_hz = librosa.midi_to_hz(midi_note)

        for h in range(1, n_harmonics + 1):
            harmonic_hz = fundamental_hz * h


            if harmonic_hz < fmin_hz:
                continue


            # formula: bin = bins_per_octave * log2(freq / fmin)
            bin_float = bins_per_octave * np.log2(harmonic_hz / fmin_hz)
            bin_center = int(round(bin_float))

            if bin_center >= n_bins:
                break 


            amp = harmonic_amplitudes[h - 1] if h <= len(harmonic_amplitudes) else 0.02

            for offset in range(-1, 2):
                b = bin_center + offset
                if 0 <= b < n_bins:
                    # central bin full amplitude pabe ashepasher gula kom kom pabe
                    spread_factor = 1.0 if offset == 0 else 0.3
                    W[b, note_idx] += amp * spread_factor

    # normalize korle kono loud gula boss hoyejabena
    col_norms = np.linalg.norm(W, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1.0  # Avoid division by zero
    W = W / col_norms

    return W, note_midi_list


def _nmf_decompose(V, W, n_iterations=80):
    """

        H ← H * (W^T × V) / (W^T × W × H + ε)

    Args:
        V: (n_bins × n_frames) observed CQT spectrogram
        W: (n_bins × n_notes) fixed harmonic templates
        n_iterations: number of multiplicative update steps
    """
    n_notes = W.shape[1]
    n_frames = V.shape[1]
    eps = 1e-10  # Small constant to prevent division by zero

    # Initialize H with small random positive values
    np.random.seed(42)  # Fixed seed for reproducible results
    H = np.random.rand(n_notes, n_frames) * 0.1 + 0.01

    # Precompute W^T for efficiency (doesn't change during iterations)
    Wt = W.T
    WtW = Wt @ W  # (n_notes × n_notes)

    for iteration in range(n_iterations):
        # Lee & Seung multiplicative update rule for H
        numerator = Wt @ V          # (n_notes × n_frames)
        denominator = WtW @ H + eps # (n_notes × n_frames)
        H *= (numerator / denominator)

    return H

def _enforce_max_polyphony(H, max_poly=4):
    """
    Enforce a maximum number of simultaneous notes at each time frame.
    
    When NMF detects 6 notes at once, the weakest 2 are almost certainly
    harmonics of stronger notes or noise. This function keeps only the
    top N strongest activations at each frame and zeros out the rest.
    
    This is called "sparsity-constrained NMF" in the literature.
    
    How it works:
    1. For each time frame (column of H), rank all note activations
    2. Keep only the top `max_poly` activations
    3. Zero out everything else
    
    Example:
      Before (6 notes active at frame t):
        C4: 0.8, E4: 0.6, G4: 0.5, C5: 0.3, E5: 0.15, G5: 0.08
      
      After max_poly=4:
        C4: 0.8, E4: 0.6, G4: 0.5, C5: 0.3, E5: 0.0, G5: 0.0
                                          ↑ zeroed   ↑ zeroed
    
    Args:
        H: (n_notes × n_frames) activation matrix
        max_poly: maximum number of simultaneous notes allowed
    
    Returns:
        H_sparse: same shape as H, but with weak activations zeroed out
    """
    H_sparse = H.copy()
    n_notes, n_frames = H.shape
    
    for frame_idx in range(n_frames):
        column = H_sparse[:, frame_idx]
        
        # Count how many notes are significantly active
        active_count = np.sum(column > 0.01)
        
        if active_count > max_poly:
            # Find the indices of the top N strongest activations
            top_indices = np.argsort(column)[-max_poly:]
            
            # Zero out everything that's NOT in the top N
            mask = np.ones(n_notes, dtype=bool)
            mask[top_indices] = False
            column[mask] = 0
            
            H_sparse[:, frame_idx] = column
    
    return H_sparse


def _activations_to_note_events_raw(H, note_midi_list, times, profile, onset_times=None):
    """
    Extract raw note events from activation matrix WITHOUT inserting RESTs
    or grouping into chords. This is used internally for the two-pass system.
    
    Key improvement: Uses onset detection to split repeated notes.
    When the same note is played twice (e.g., C-C in Für Elise),
    the onset detector finds the second strike and splits the
    continuous activation into two separate note events.
    
    Returns a list of (midi_note, start_frame, end_frame) tuples.
    """
    n_notes, n_frames = H.shape
    threshold = profile["activation_threshold"]
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.02

    # Normalize H
    h_max = np.max(H)
    if h_max > 0:
        H_norm = H / h_max
    else:
        return []

    active = H_norm > threshold
    raw_events = []

    for note_idx in range(n_notes):
        in_note = False
        start_frame = 0

        for frame_idx in range(n_frames):
            if active[note_idx, frame_idx] and not in_note:
                # Note just started
                in_note = True
                start_frame = frame_idx
            elif not active[note_idx, frame_idx] and in_note:
                # Note just ended
                in_note = False
                raw_events.append((note_midi_list[note_idx], start_frame, frame_idx))

        if in_note:
            raw_events.append((note_midi_list[note_idx], start_frame, n_frames))

    # ============================================
    # SPLIT REPEATED NOTES USING ONSET DETECTION
    # ============================================
    # This is the critical fix for the "two strokes becoming one" problem.
    #
    # When you play C-C quickly, the NMF activation shows one continuous
    # block. But the onset detector hears two distinct strikes.
    # We use the onset times to split the continuous block into two events.
    
    if onset_times is not None and len(onset_times) > 0:
        split_events = []
        
        for midi, start, end in raw_events:
            note_start_time = times[min(start, len(times) - 1)]
            note_end_time = times[min(end - 1, len(times) - 1)] + frame_duration
            
            # Find all onsets that fall WITHIN this note's time range
            # We add a small margin (30ms) to avoid splitting at the exact edges
            margin = 0.03
            internal_onsets = [
                ot for ot in onset_times
                if note_start_time + margin < ot < note_end_time - margin
            ]
            
            if len(internal_onsets) == 0:
                # No internal onsets — this is a single note, keep as-is
                split_events.append((midi, start, end))
            else:
                # There are onsets inside this note — split at each onset
                # This turns one long C into multiple C notes
                split_points = [start]
                
                for ot in internal_onsets:
                    # Convert onset time to frame index
                    onset_frame = np.argmin(np.abs(times - ot))
                    # Only split if the onset frame is actually inside the event
                    if start < onset_frame < end:
                        split_points.append(onset_frame)
                
                split_points.append(end)
                
                # Create separate events for each segment
                for k in range(len(split_points) - 1):
                    seg_start = split_points[k]
                    seg_end = split_points[k + 1]
                    seg_duration = (seg_end - seg_start) * frame_duration
                    
                    # Only keep segments that are long enough to be real notes
                    if seg_duration >= profile["min_note_duration"]:
                        split_events.append((midi, seg_start, seg_end))
        
        raw_events = split_events

    # ============================================
    # MERGE TINY GAPS (but be much less aggressive)
    # ============================================
    # Only merge if the gap is VERY small (< 30ms) AND the notes are the same pitch.
    # Previously this was too aggressive and was merging separate notes.
    min_gap_frames = max(1, int(0.03 / frame_duration))  # 30ms max gap for merging
    merged = []
    raw_events.sort(key=lambda x: (x[1], x[0]))

    for event in raw_events:
        midi, start, end = event
        duration = (end - start) * frame_duration
        if duration < profile["min_note_duration"]:
            continue
        if merged:
            last_midi, last_start, last_end = merged[-1]
            gap = start - last_end
            # Only merge if same pitch AND gap is tiny (< 30ms)
            if last_midi == midi and 0 < gap <= min_gap_frames:
                merged[-1] = (midi, last_start, end)
                continue
        merged.append((midi, start, end))

    min_midi = profile.get("min_midi", 45)
    merged = [(midi, start, end) for midi, start, end in merged if midi >= min_midi]

    return merged


def _merge_two_passes(events_pass1, events_pass2, times):
    """
    Merge note events from two passes.
    
    RULE: Pass 1 notes are NEVER modified. Pass 2 notes only fill
    the silent gaps between pass 1 notes. If a pass 2 note overlaps
    with a pass 1 note, the overlapping portion is trimmed away.
    
    Think of it like this:
      Pass 1 = the melody (untouchable)
      Pass 2 = the bass (only audible when melody is silent)
    """
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.02
    n_frames = len(times)

    # Step 1: Build a coverage map from pass 1
    # This tells us which time frames already have notes from pass 1
    covered = np.zeros(n_frames, dtype=bool)
    for midi, start, end in events_pass1:
        s = max(0, start)
        e = min(n_frames, end)
        covered[s:e] = True

    # Step 2: Find the gaps (uncovered regions) in pass 1
    # These are the only places where pass 2 notes are allowed
    gaps = []
    in_gap = False
    gap_start = 0

    for frame_idx in range(n_frames):
        if not covered[frame_idx] and not in_gap:
            in_gap = True
            gap_start = frame_idx
        elif covered[frame_idx] and in_gap:
            in_gap = False
            gaps.append((gap_start, frame_idx))

    if in_gap:
        gaps.append((gap_start, n_frames))

    # Step 3: Clip pass 2 events to fit within the gaps
    # A pass 2 event is only kept if it falls inside a gap
    clipped_pass2 = []

    for midi2, start2, end2 in events_pass2:
        # Find which gaps this pass 2 event overlaps with
        for gap_start, gap_end in gaps:
            # Calculate the overlap between this event and this gap
            clip_start = max(start2, gap_start)
            clip_end = min(end2, gap_end)

            if clip_start < clip_end:
                # There is a valid overlap — keep this clipped portion
                clipped_duration = (clip_end - clip_start) * frame_duration
                if clipped_duration >= 0.05:  # At least 50ms to be meaningful
                    clipped_pass2.append((midi2, clip_start, clip_end))

    # Step 4: Merge pass 2 clipped events that are the same note and adjacent
    clipped_pass2.sort(key=lambda x: (x[0], x[1]))
    merged_pass2 = []

    for event in clipped_pass2:
        midi, start, end = event
        if merged_pass2:
            last_midi, last_start, last_end = merged_pass2[-1]
            # Merge if same pitch and gap between them is tiny (< 80ms)
            if last_midi == midi and (start - last_end) * frame_duration < 0.08:
                merged_pass2[-1] = (midi, last_start, end)
                continue
        merged_pass2.append((midi, start, end))

    # Step 5: Combine pass 1 (untouched) + clipped pass 2
    all_events = list(events_pass1) + merged_pass2

    # Sort by start time, then by pitch (low notes first for chords)
    all_events.sort(key=lambda x: (x[1], x[0]))

    # Step 6: Group simultaneous notes into chords and build output
    output_sequence = []
    simultaneous_threshold = 0.05  # 50ms overlap = simultaneous
    used = set()
    note_groups = []

    for i, (midi_i, start_i, end_i) in enumerate(all_events):
        if i in used:
            continue

        group = [(midi_i, start_i, end_i)]
        used.add(i)

        for j in range(i + 1, len(all_events)):
            if j in used:
                continue
            midi_j, start_j, end_j = all_events[j]

            overlap_start = max(start_i, start_j)
            overlap_end = min(end_i, end_j)
            overlap_duration = max(0, overlap_end - overlap_start) * frame_duration

            if overlap_duration >= simultaneous_threshold:
                group.append((midi_j, start_j, end_j))
                used.add(j)

        note_groups.append(group)

    # Step 7: Build final output with RESTs
    last_end_time = 0.0

    for group in note_groups:
        midi_notes = sorted(set(g[0] for g in group))
        group_start_frame = min(g[1] for g in group)
        group_end_frame = max(g[2] for g in group)

        start_t = times[min(group_start_frame, len(times) - 1)]
        end_t = times[min(group_end_frame, len(times) - 1)] + frame_duration

        # Insert REST if there's a gap
        if start_t - last_end_time > 0.15:
            output_sequence.append({
                "timestamp": f"{last_end_time:.2f}s - {start_t:.2f}s",
                "note": "REST",
                "frequency": "-",
            })

        note_names = [librosa.midi_to_note(m) for m in midi_notes]
        freqs = [librosa.midi_to_hz(m) for m in midi_notes]

        if len(midi_notes) > 1:
            output_sequence.append({
                "timestamp": f"{start_t:.2f}s - {end_t:.2f}s",
                "note": " + ".join(note_names),
                "frequency": " + ".join([f"{f:.1f} Hz" for f in freqs]),
                "is_chord": True,
                "midi_notes": midi_notes,
            })
        else:
            output_sequence.append({
                "timestamp": f"{start_t:.2f}s - {end_t:.2f}s",
                "note": note_names[0],
                "frequency": f"{freqs[0]:.1f} Hz",
                "is_chord": False,
                "midi_notes": midi_notes,
            })

        last_end_time = end_t

    return output_sequence


def _filter_excess_notes(events, H, note_midi_list, times, max_poly=4):
    """
    Post-processing filter: remove excess notes when too many overlap.
    
    This runs AFTER both NMF passes are complete. It scans the final
    event list and at any moment where more than `max_poly` notes are
    active simultaneously, it removes the weakest ones.
    
    "Weakest" is determined by the average NMF activation energy in H.
    Real notes (melody, bass) have high activation. Artifacts and
    misdetected harmonics have low activation. So the weakest ones
    are almost always the wrong ones.
    
    Example:
      6 notes detected at time 2.5s:
        C4: activation 0.85  ← keep (melody)
        E4: activation 0.72  ← keep (chord tone)
        G4: activation 0.61  ← keep (chord tone)
        C5: activation 0.45  ← keep (melody octave)
        E5: activation 0.12  ← REMOVE (harmonic artifact)
        G5: activation 0.08  ← REMOVE (harmonic artifact)
    
    Args:
        events: list of (midi_note, start_frame, end_frame) tuples
        H: (n_notes × n_frames) combined activation matrix
        note_midi_list: MIDI note numbers for each row of H
        times: time stamps for each frame
        max_poly: maximum simultaneous notes allowed (default 4)
    
    Returns:
        Filtered list of (midi_note, start_frame, end_frame) tuples
    """
    if not events:
        return events
    
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.02
    n_frames = len(times)
    
    # Step 1: Compute the average activation strength for each event
    # This tells us how "real" each detected note is
    event_strengths = []
    
    # Build a lookup: midi_note → row index in H
    midi_to_row = {}
    for row_idx, midi in enumerate(note_midi_list):
        if midi not in midi_to_row:
            midi_to_row[midi] = row_idx
    
    for midi, start, end in events:
        row_idx = midi_to_row.get(midi)
        if row_idx is not None and row_idx < H.shape[0]:
            # Average activation over the note's duration
            s = max(0, start)
            e = min(n_frames, end)
            if e > s:
                avg_activation = np.mean(H[row_idx, s:e])
            else:
                avg_activation = 0.0
        else:
            avg_activation = 0.0
        
        event_strengths.append(avg_activation)
    
    # Step 2: For each frame, find which events are active
    # and remove the weakest ones if there are too many
    events_active_at_frame = [[] for _ in range(n_frames)]
    
    for event_idx, (midi, start, end) in enumerate(events):
        for f in range(max(0, start), min(n_frames, end)):
            events_active_at_frame[f].append(event_idx)
    
    # Step 3: Mark events for removal
    # An event is removed if, at ANY frame during its lifetime,
    # it is among the weakest notes when polyphony exceeds max_poly
    removal_votes = np.zeros(len(events))
    
    for f in range(n_frames):
        active = events_active_at_frame[f]
        
        if len(active) <= max_poly:
            continue  # Fine, no filtering needed at this frame
        
        # Too many notes at this frame — rank by strength
        active_with_strength = [
            (idx, event_strengths[idx]) for idx in active
        ]
        active_with_strength.sort(key=lambda x: x[1], reverse=True)
        
        # The notes beyond max_poly are the weakest — vote to remove them
        for rank, (idx, strength) in enumerate(active_with_strength):
            if rank >= max_poly:
                removal_votes[idx] += 1
    
    # Step 4: Remove events that got too many removal votes
    # An event is removed if it was marked as excess in more than 30%
    # of its frames. This prevents removing a note just because of a
    # brief momentary overlap.
    filtered_events = []
    
    for event_idx, (midi, start, end) in enumerate(events):
        duration_frames = max(1, end - start)
        removal_ratio = removal_votes[event_idx] / duration_frames
        
        if removal_ratio < 0.3:
            # Keep this note — it was only occasionally in excess
            filtered_events.append((midi, start, end))
        else:
            # Remove this note — it was consistently the weakest
            # in over-polyphonic regions
            pass
    
    return filtered_events


def _remove_harmonic_artifacts(events, H, note_midi_list, times):
    """
    Remove ghost notes that are actually harmonics of lower real notes.
    
    This is the most important filter in the entire system. It solves the
    problem where playing a simple C3 chord produces phantom C5, E5, G5 notes.
    
    How it works:
    1. Sort all detected notes from lowest pitch to highest
    2. For each high note, check if its frequency is an integer multiple
       of any lower note's frequency (i.e., is it a harmonic?)
    3. If yes, AND both notes overlap in time, AND the high note is weaker
       than the low note → the high note is a harmonic ghost → DELETE IT
    
    Example with C3 chord (C3-E3-G3):
      Detected: C3, E3, G3, C4, G4, C5, E5
                                    ↑   ↑   ↑   ↑
                                    harmonics of C3!
      
      C4 (262 Hz) / C3 (131 Hz) = 2.0 → exact 2nd harmonic → REMOVE
      G4 (392 Hz) / C3 (131 Hz) = 3.0 → exact 3rd harmonic → REMOVE
      C5 (523 Hz) / C3 (131 Hz) = 4.0 → exact 4th harmonic → REMOVE
      E5 (659 Hz) / C3 (131 Hz) = 5.0 → exact 5th harmonic → REMOVE
      
      Final: C3, E3, G3 ← clean!
    
    The tolerance is 3% because real piano strings are slightly inharmonic
    (stiffness causes harmonics to be slightly sharp).
    """
    if not events:
        return events
    
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.02
    
    # Build midi → row index lookup for activation strength
    midi_to_row = {}
    for row_idx, midi in enumerate(note_midi_list):
        if midi not in midi_to_row:
            midi_to_row[midi] = row_idx
    
    def get_strength(midi, start, end):
        """Get average NMF activation for a note event."""
        row = midi_to_row.get(midi)
        if row is not None and row < H.shape[0]:
            s = max(0, start)
            e = min(len(times), end)
            if e > s:
                return np.mean(H[row, s:e])
        return 0.0
    
    def is_harmonic_of(higher_midi, lower_midi, tolerance=0.03):
        """
        Check if higher_midi is a harmonic of lower_midi.
        
        A note is a harmonic if its frequency is approximately an integer
        multiple of the lower note's frequency.
        
        tolerance=0.03 means 3% deviation allowed (piano inharmonicity)
        """
        freq_high = librosa.midi_to_hz(higher_midi)
        freq_low = librosa.midi_to_hz(lower_midi)
        
        if freq_low <= 0:
            return False
        
        ratio = freq_high / freq_low
        
        # Check if ratio is close to an integer (2, 3, 4, 5, 6, 7, 8)
        # We only check up to the 8th harmonic
        for harmonic_number in range(2, 9):
            if abs(ratio - harmonic_number) / harmonic_number < tolerance:
                return True
        
        return False
    
    def events_overlap(ev1, ev2):
        """Check if two events overlap in time by at least 30%."""
        start1, end1 = ev1[1], ev1[2]
        start2, end2 = ev2[1], ev2[2]
        overlap = max(0, min(end1, end2) - max(start1, start2))
        shorter = min(end1 - start1, end2 - start2)
        if shorter <= 0:
            return False
        return (overlap / shorter) > 0.3
    
    # Sort events by pitch (lowest first)
    sorted_events = sorted(events, key=lambda x: x[0])
    
    # Mark events for removal
    remove_set = set()
    
    for i in range(len(sorted_events)):
        if i in remove_set:
            continue
        
        midi_i, start_i, end_i = sorted_events[i]
        strength_i = get_strength(midi_i, start_i, end_i)
        
        # Check all HIGHER notes
        for j in range(i + 1, len(sorted_events)):
            if j in remove_set:
                continue
            
            midi_j, start_j, end_j = sorted_events[j]
            strength_j = get_strength(midi_j, start_j, end_j)
            
            # Skip if not overlapping in time
            if not events_overlap(sorted_events[i], sorted_events[j]):
                continue
            
            # Check if the higher note is a harmonic of the lower note
            if is_harmonic_of(midi_j, midi_i):
                # The higher note is a harmonic ghost if:
                # 1. It's weaker than the lower note, OR
                # 2. It's close in strength but the interval is a perfect
                #    octave/5th (most common harmonic confusion)
                freq_ratio = librosa.midi_to_hz(midi_j) / librosa.midi_to_hz(midi_i)
                
                if strength_j < strength_i * 0.8:
                    # Clearly weaker → definitely a harmonic ghost
                    remove_set.add(j)
                elif strength_j < strength_i * 1.2 and freq_ratio < 5.0:
                    # Similar strength but within 4th harmonic range
                    # and the higher note is not much stronger
                    # → likely a harmonic, remove it
                    remove_set.add(j)
    
    # Build filtered list
    filtered = [
        ev for idx, ev in enumerate(sorted_events)
        if idx not in remove_set
    ]
    
    return filtered

def _format_events_to_output(events, times):
    """
    Convert raw (midi, start_frame, end_frame) events to the final
    output format with chords, RESTs, and proper timestamps.
    """
    if not events:
        return []
    
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.02
    simultaneous_threshold = 0.05  # 50ms
    
    # Sort by start time, then by pitch
    events.sort(key=lambda x: (x[1], x[0]))
    
    # Group simultaneous notes into chords
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
            
            overlap_start = max(start_i, start_j)
            overlap_end = min(end_i, end_j)
            overlap_duration = max(0, overlap_end - overlap_start) * frame_duration
            
            if overlap_duration >= simultaneous_threshold:
                group.append((midi_j, start_j, end_j))
                used.add(j)
        
        note_groups.append(group)
    
    # Build output with RESTs
    output_sequence = []
    last_end_time = 0.0
    
    for group in note_groups:
        midi_notes = sorted(set(g[0] for g in group))
        group_start_frame = min(g[1] for g in group)
        group_end_frame = max(g[2] for g in group)
        
        start_t = times[min(group_start_frame, len(times) - 1)]
        end_t = times[min(group_end_frame, len(times) - 1)] + frame_duration
        
        # Insert REST if there's a gap
        if start_t - last_end_time > 0.15:
            output_sequence.append({
                "timestamp": f"{last_end_time:.2f}s - {start_t:.2f}s",
                "note": "REST",
                "frequency": "-",
            })
        
        note_names = [librosa.midi_to_note(m) for m in midi_notes]
        freqs = [librosa.midi_to_hz(m) for m in midi_notes]
        
        if len(midi_notes) > 1:
            output_sequence.append({
                "timestamp": f"{start_t:.2f}s - {end_t:.2f}s",
                "note": " + ".join(note_names),
                "frequency": " + ".join([f"{f:.1f} Hz" for f in freqs]),
                "is_chord": True,
                "midi_notes": midi_notes,
            })
        else:
            output_sequence.append({
                "timestamp": f"{start_t:.2f}s - {end_t:.2f}s",
                "note": note_names[0],
                "frequency": f"{freqs[0]:.1f} Hz",
                "is_chord": False,
                "midi_notes": midi_notes,
            })
        
        last_end_time = end_t
    
    return output_sequence


# def process_audio_polyphonic(file_path, profile_name="piano_poly"):
#     """
#     Main function: detect notes in polyphonic audio using CQT + NMF.

#     Pipeline:
#     1. Load audio and compute CQT spectrogram (V matrix)
#     2. Build harmonic template matrix (W matrix)
#     3. Run NMF to get activation matrix (H matrix)
#     4. Convert activations to note events
#     """
#     profile = POLYPHONIC_PROFILES.get(profile_name, POLYPHONIC_PROFILES["piano_poly"])

#     # Step 1: Load audio
#     sample_rate = 22050
#     y, sr = librosa.load(file_path, sr=sample_rate, mono=True)

#     fmin_hz = librosa.note_to_hz(profile["fmin"])
#     fmax_hz = librosa.note_to_hz(profile["fmax"])
#     bins_per_octave = profile["bins_per_octave"]

#     # Calculate number of CQT bins
#     n_bins = int(np.ceil(bins_per_octave * np.log2(fmax_hz / fmin_hz)))

#     # Step 2: Compute CQT spectrogram (this is our V matrix)
#     # V shape: (n_bins, n_frames)
#     C = np.abs(librosa.cqt(
#         y, sr=sr,
#         hop_length=profile["hop_length"],
#         fmin=fmin_hz,
#         n_bins=n_bins,
#         bins_per_octave=bins_per_octave,
#         filter_scale=1.0
#     ))

#     times = librosa.times_like(C, sr=sr, hop_length=profile["hop_length"])

#     # Step 3: Build harmonic template matrix (W matrix)
#     # W shape: (n_bins, n_notes)
#     W, note_midi_list = _build_harmonic_template_matrix(
#         n_bins, fmin_hz, bins_per_octave, profile["n_harmonics"]
#     )

#     # Step 4: Run NMF decomposition to get activation matrix (H matrix)
#     # V ≈ W × H, so H shape: (n_notes, n_frames)
#     H = _nmf_decompose(C, W, n_iterations=profile["nmf_iterations"])

#     # Step 5: Convert activations to note events
#     output_sequence = _activations_to_note_events(H, note_midi_list, times, profile)

#     return output_sequence


# changed for iterative overlap smth

# def process_audio_polyphonic(file_path, profile_name="piano_poly"):
#     """
#     Main function: detect notes in polyphonic audio using CQT + NMF.
    
#     Uses a TWO-PASS approach:
#       Pass 1: Run NMF on the original CQT → detect loud/prominent notes
#       Pass 2: Subtract detected notes from CQT → run NMF on residual → detect hidden notes
#       Merge: Combine both passes for complete note coverage
    
#     This iterative residual approach prevents soft notes (like bass lines
#     in Für Elise) from being masked by louder melody notes.
#     """
#     profile = POLYPHONIC_PROFILES.get(profile_name, POLYPHONIC_PROFILES["piano_poly"])

#     # Step 1: Load audio
#     sample_rate = 22050
#     y, sr = librosa.load(file_path, sr=sample_rate, mono=True)

#     fmin_hz = librosa.note_to_hz(profile["fmin"])
#     fmax_hz = librosa.note_to_hz(profile["fmax"])
#     bins_per_octave = profile["bins_per_octave"]

#     # Calculate number of CQT bins
#     n_bins = int(np.ceil(bins_per_octave * np.log2(fmax_hz / fmin_hz)))

#     # Step 2: Compute CQT spectrogram (V matrix)
#     C = np.abs(librosa.cqt(
#         y, sr=sr,
#         hop_length=profile["hop_length"],
#         fmin=fmin_hz,
#         n_bins=n_bins,
#         bins_per_octave=bins_per_octave,
#         filter_scale=1.0
#     ))

#     times = librosa.times_like(C, sr=sr, hop_length=profile["hop_length"])

#     # Step 3: Build harmonic template matrix (W matrix)
#     W, note_midi_list = _build_harmonic_template_matrix(
#         n_bins, fmin_hz, bins_per_octave, profile["n_harmonics"]
#     )

#     # ============================================
#     # PASS 1: Detect prominent notes
#     # ============================================
#     H1 = _nmf_decompose(C, W, n_iterations=profile["nmf_iterations"])
    
#     # Use a HIGHER threshold for pass 1 — only catch the confident notes
#     pass1_threshold = profile["activation_threshold"] * 1.2
#     pass1_profile = dict(profile)
#     pass1_profile["activation_threshold"] = pass1_threshold
    
#     events_pass1 = _activations_to_note_events_raw(H1, note_midi_list, times, pass1_profile)

#     # ============================================
#     # PASS 2: Subtract pass 1 notes, detect hidden notes
#     # ============================================
#     # Reconstruct what pass 1 found: V_reconstructed = W × H1
#     # But only keep the strong activations (the ones we actually detected)
#     H1_strong = H1.copy()
#     h1_max = np.max(H1) if np.max(H1) > 0 else 1.0
#     H1_strong[H1_strong / h1_max < pass1_threshold] = 0
    
#     # Reconstruct the spectral contribution of detected notes
#     C_reconstructed = W @ H1_strong
    
#     # Subtract from original to get the residual
#     # Use soft subtraction: max(0, original - reconstructed)
#     # This prevents negative values (NMF requires non-negative input)
#     C_residual = np.maximum(0, C - C_reconstructed)
    
#     # Check if there's enough energy left in the residual to bother
#     residual_energy = np.sum(C_residual)
#     original_energy = np.sum(C)
    
#     events_pass2 = []
#     if original_energy > 0 and (residual_energy / original_energy) > 0.05:
#         # More than 5% energy remaining — there are hidden notes to find
        
#         # Run NMF on the residual with a LOWER threshold
#         # (the remaining notes are quieter, so we need to be more sensitive)
#         H2 = _nmf_decompose(C_residual, W, n_iterations=profile["nmf_iterations"])
        
#         pass2_threshold = profile["activation_threshold"] * 0.7
#         pass2_profile = dict(profile)
#         pass2_profile["activation_threshold"] = pass2_threshold
        
#         events_pass2 = _activations_to_note_events_raw(H2, note_midi_list, times, pass2_profile)

#     # ============================================
#     # MERGE: Combine both passes
#     # ============================================
#     merged_events = _merge_two_passes(events_pass1, events_pass2, times)
    
#     return merged_events



def process_audio_polyphonic(file_path, profile_name="piano_poly"):
    """
    Main function: detect notes in polyphonic audio using CQT + NMF.
    
    Uses a TWO-PASS approach with onset-based note splitting:
      Pass 1: Run NMF on the original CQT → detect loud/prominent notes
      Pass 2: Subtract detected notes from CQT → run NMF on residual → detect hidden notes
      Merge: Combine both passes, filling only silent gaps
    
    Onset detection is used to split repeated notes (e.g., C-C in Für Elise)
    that would otherwise be merged into one continuous note.
    """
    profile = POLYPHONIC_PROFILES.get(profile_name, POLYPHONIC_PROFILES["piano_poly"])

    # Step 1: Load audio
    sample_rate = 22050
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)

    fmin_hz = librosa.note_to_hz(profile["fmin"])
    fmax_hz = librosa.note_to_hz(profile["fmax"])
    bins_per_octave = profile["bins_per_octave"]

    # Calculate number of CQT bins
    n_bins = int(np.ceil(bins_per_octave * np.log2(fmax_hz / fmin_hz)))

    # Step 2: Compute CQT spectrogram (V matrix)
    C = np.abs(librosa.cqt(
        y, sr=sr,
        hop_length=profile["hop_length"],
        fmin=fmin_hz,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        filter_scale=1.0
    ))

    times = librosa.times_like(C, sr=sr, hop_length=profile["hop_length"])

    # Step 3: Detect onsets from the original audio
    # This is critical for splitting repeated notes like C-C
    # We use multiple onset detection methods and combine them for robustness
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        hop_length=profile["hop_length"],
        backtrack=True,
        units='frames'
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=profile["hop_length"])
    
    # Also detect onsets using spectral flux (more sensitive to repeated notes)
    onset_frames_flux = librosa.onset.onset_detect(
        y=y, sr=sr,
        hop_length=profile["hop_length"],
        onset_envelope=librosa.onset.onset_strength(
            y=y, sr=sr,
            hop_length=profile["hop_length"],
            aggregate=np.median
        ),
        backtrack=True,
        units='frames'
    )
    onset_times_flux = librosa.frames_to_time(onset_frames_flux, sr=sr, hop_length=profile["hop_length"])
    
    # Combine both onset detections and remove duplicates within 40ms
    all_onsets = np.sort(np.concatenate([onset_times, onset_times_flux]))
    if len(all_onsets) > 0:
        # Remove onsets that are too close together (< 40ms)
        keep = [True]
        for i in range(1, len(all_onsets)):
            if all_onsets[i] - all_onsets[i-1] >= 0.04:
                keep.append(True)
            else:
                keep.append(False)
        onset_times = all_onsets[keep]
    else:
        onset_times = np.array([])

    # Step 4: Build harmonic template matrix (W matrix)
    W, note_midi_list = _build_harmonic_template_matrix(
        n_bins, fmin_hz, bins_per_octave, profile["n_harmonics"]
    )

    # ============================================
    # PASS 1: Detect prominent notes
    # ============================================
    H1 = _nmf_decompose(C, W, n_iterations=profile["nmf_iterations"])

    max_poly = profile.get("max_polyphony", 4)
    H1 = _enforce_max_polyphony(H1, max_poly=max_poly)
    
    # Use a HIGHER threshold for pass 1 — only catch the confident notes
    pass1_threshold = profile["activation_threshold"] * 1.2
    pass1_profile = dict(profile)
    pass1_profile["activation_threshold"] = pass1_threshold
    
    events_pass1 = _activations_to_note_events_raw(
        H1, note_midi_list, times, pass1_profile, onset_times=onset_times
    )

    # ============================================
    # PASS 2: Subtract pass 1 notes, detect hidden notes
    # ============================================
    H1_strong = H1.copy()
    h1_max = np.max(H1) if np.max(H1) > 0 else 1.0
    H1_strong[H1_strong / h1_max < pass1_threshold] = 0
    
    # Reconstruct the spectral contribution of detected notes
    C_reconstructed = W @ H1_strong
    
    # Subtract from original to get the residual
    C_residual = np.maximum(0, C - C_reconstructed)
    
    # Check if there's enough energy left in the residual
    residual_energy = np.sum(C_residual)
    original_energy = np.sum(C)
    
    events_pass2 = []
    if original_energy > 0 and (residual_energy / original_energy) > 0.05:
        H2 = _nmf_decompose(C_residual, W, n_iterations=profile["nmf_iterations"])
        H2 = _enforce_max_polyphony(H2, max_poly=max_poly)

        
        pass2_threshold = profile["activation_threshold"] * 0.7
        pass2_profile = dict(profile)
        pass2_profile["activation_threshold"] = pass2_threshold
        
        events_pass2 = _activations_to_note_events_raw(
            H2, note_midi_list, times, pass2_profile, onset_times=onset_times
        )

    # ============================================
    # MERGE: Combine both passes (pass 2 fills gaps only)
    # ============================================
    # merged_events = _merge_two_passes(events_pass1, events_pass2, times)
    
    # return merged_events

    all_raw_events = list(events_pass1)
    
    # Clip pass 2 to gaps only (same logic as before)
    covered = np.zeros(len(times), dtype=bool)
    for midi, start, end in events_pass1:
        s = max(0, start)
        e = min(len(times), end)
        covered[s:e] = True
    
    for midi2, start2, end2 in events_pass2:
        is_duplicate = False
        for midi1, start1, end1 in events_pass1:
            if midi1 == midi2:
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                if max(0, overlap_end - overlap_start) / max(1, end2 - start2) > 0.5:
                    is_duplicate = True
                    break
        if not is_duplicate:
            # Only keep the parts that fall in gaps
            for f in range(max(0, start2), min(len(times), end2)):
                if not covered[f]:
                    all_raw_events.append((midi2, start2, end2))
                    break
    
    # Remove exact duplicates
    all_raw_events = list(set(all_raw_events))
    all_raw_events.sort(key=lambda x: (x[1], x[0]))
    
    # ============================================
    # POST-FILTER: Remove excess notes
    # ============================================
    # Combine H matrices from both passes for strength evaluation
        # ============================================
    # POST-FILTER 1: Remove harmonic ghost notes
    # ============================================
    # This is the most important filter. It removes notes like C5, E5
    # that are actually just harmonics of C3.
    H_combined = np.maximum(H1, H2) if len(events_pass2) > 0 else H1
    
    filtered_events = _remove_harmonic_artifacts(
        all_raw_events, H_combined, note_midi_list, times
    )
    
    # ============================================
    # POST-FILTER 2: Remove excess polyphony
    # ============================================
    # After removing harmonics, if there are still too many notes,
    # remove the weakest ones.
    max_poly = profile.get("max_polyphony", 4)
    filtered_events = _filter_excess_notes(
        filtered_events, H_combined, note_midi_list, times, max_poly=max_poly
    )
    
    # ============================================
    # FORMAT: Convert filtered events to output
    # ============================================
    merged_events = _format_events_to_output(filtered_events, times)
    
    return merged_events


def get_profile_display_info(profile_name):
    """Helper for the UI"""
    profile = POLYPHONIC_PROFILES.get(profile_name, POLYPHONIC_PROFILES["piano_poly"])
    return {
        "label": profile["label"],
        "fmin": profile["fmin"],
        "fmax": profile["fmax"],
    }