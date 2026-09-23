import numpy as np
import librosa


POLYPHONIC_PROFILES = {
    "piano_poly": {
        "label": "Piano Polyphonic (Best for chords & bass)",
        "fmin": "A1",          # 55 Hz - covers low bass notes
        "fmax": "C7",          # 2093 Hz - covers most piano range
        "hop_length": 512,
        "bins_per_octave": 36, # 3 bins per semitone for high resolution
        "nmf_iterations": 80,  # More iterations = more accurate but slower
        "activation_threshold": 0.15,  # Min activation to count as a note
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


def _activations_to_note_events(H, note_midi_list, times, profile):
    """
    Convert the NMF activation matrix H into a list of note events.

    H tells us the "strength" of each note at each time frame.
    We threshold H, find connected regions, and merge short gaps.

    Args:
        H: (n_notes × n_frames) activation matrix from NMF
        note_midi_list: MIDI note numbers for each row of H
        times: time stamps for each frame
        profile: analysis profile dict

    Returns:
        List of note event dicts (same format as pYIN engine)
    """
    n_notes, n_frames = H.shape
    threshold = profile["activation_threshold"]
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.02

    # Normalize H so the maximum activation is 1.0
    h_max = np.max(H)
    if h_max > 0:
        H_norm = H / h_max
    else:
        return []

    # Step 1: Threshold the activation matrix to get binary note presence
    active = H_norm > threshold

    # Step 2: Extract note events from the binary matrix
    raw_events = []  # List of (midi_note, start_frame, end_frame)

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

        # Handle note that extends to the end
        if in_note:
            raw_events.append((note_midi_list[note_idx], start_frame, n_frames))

    # Step 3: Merge events that are very close together (tiny gaps)
    min_rest_frames = max(1, int(profile["min_rest_duration"] / frame_duration))
    merged_events = []

    # Sort events by start time, then by pitch
    raw_events.sort(key=lambda x: (x[1], x[0]))

    for event in raw_events:
        midi, start, end = event
        duration = (end - start) * frame_duration

        # Skip very short events (likely noise)
        if duration < profile["min_note_duration"]:
            continue

        # Try to merge with the last event of the same pitch
        if merged_events:
            last_midi, last_start, last_end = merged_events[-1]
            gap = start - last_end
            if last_midi == midi and gap <= min_rest_frames:
                # Merge: extend the previous event
                merged_events[-1] = (midi, last_start, end)
                continue

        merged_events.append((midi, start, end))

    # Step 4: Group simultaneous notes into chords
    # Two notes are "simultaneous" if they overlap in time by more than 50%
    simultaneous_threshold = 0.05  # 50ms overlap minimum

    # Sort by start time
    merged_events.sort(key=lambda x: x[1])

    # Build time-sorted output
    output_sequence = []
    used = set()

    for i, (midi_i, start_i, end_i) in enumerate(merged_events):
        if i in used:
            continue

        start_t = times[min(start_i, len(times) - 1)]
        end_t = times[min(end_i, len(times) - 1)] + frame_duration

        # Find all notes that overlap with this one
        chord_group = [(midi_i, start_i, end_i)]
        used.add(i)

        for j in range(i + 1, len(merged_events)):
            if j in used:
                continue
            midi_j, start_j, end_j = merged_events[j]

            # Check temporal overlap
            overlap_start = max(start_i, start_j)
            overlap_end = min(end_i, end_j)
            overlap_duration = max(0, overlap_end - overlap_start) * frame_duration

            if overlap_duration >= simultaneous_threshold:
                chord_group.append((midi_j, start_j, end_j))
                used.add(j)

        # Build the output entry
        if len(chord_group) > 1:
            # Multiple simultaneous notes = chord
            midi_notes = sorted(set(g[0] for g in chord_group))
            note_names = [librosa.midi_to_note(m) for m in midi_notes]
            freqs = [librosa.midi_to_hz(m) for m in midi_notes]

            # Use the widest time span of the chord group
            group_start = times[min(min(g[1] for g in chord_group), len(times) - 1)]
            group_end = times[min(max(g[2] for g in chord_group), len(times) - 1)] + frame_duration

            output_sequence.append({
                "timestamp": f"{group_start:.2f}s - {group_end:.2f}s",
                "note": " + ".join(note_names),
                "frequency": " + ".join([f"{f:.1f} Hz" for f in freqs]),
                "is_chord": True,
                "midi_notes": midi_notes,
            })
        else:
            # Single note
            midi_note = chord_group[0][0]
            note_name = librosa.midi_to_note(midi_note)
            freq = librosa.midi_to_hz(midi_note)

            output_sequence.append({
                "timestamp": f"{start_t:.2f}s - {end_t:.2f}s",
                "note": note_name,
                "frequency": f"{freq:.1f} Hz",
                "is_chord": False,
                "midi_notes": [midi_note],
            })

    # Sort final output by timestamp
    output_sequence.sort(key=lambda x: float(x["timestamp"].split("s")[0]))

    # Insert REST events between notes
    final_sequence = []
    last_end = 0.0

    for event in output_sequence:
        event_start = float(event["timestamp"].split("s")[0])
        if event_start - last_end > 0.15:
            final_sequence.append({
                "timestamp": f"{last_end:.2f}s - {event_start:.2f}s",
                "note": "REST",
                "frequency": "-",
            })
        final_sequence.append(event)
        event_end_str = event["timestamp"].split(" - ")[1].replace("s", "")
        last_end = float(event_end_str)

    return final_sequence


def process_audio_polyphonic(file_path, profile_name="piano_poly"):
    """
    Main function: detect notes in polyphonic audio using CQT + NMF.

    Pipeline:
    1. Load audio and compute CQT spectrogram (V matrix)
    2. Build harmonic template matrix (W matrix)
    3. Run NMF to get activation matrix (H matrix)
    4. Convert activations to note events
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

    # Step 2: Compute CQT spectrogram (this is our V matrix)
    # V shape: (n_bins, n_frames)
    C = np.abs(librosa.cqt(
        y, sr=sr,
        hop_length=profile["hop_length"],
        fmin=fmin_hz,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        filter_scale=1.0
    ))

    times = librosa.times_like(C, sr=sr, hop_length=profile["hop_length"])

    # Step 3: Build harmonic template matrix (W matrix)
    # W shape: (n_bins, n_notes)
    W, note_midi_list = _build_harmonic_template_matrix(
        n_bins, fmin_hz, bins_per_octave, profile["n_harmonics"]
    )

    # Step 4: Run NMF decomposition to get activation matrix (H matrix)
    # V ≈ W × H, so H shape: (n_notes, n_frames)
    H = _nmf_decompose(C, W, n_iterations=profile["nmf_iterations"])

    # Step 5: Convert activations to note events
    output_sequence = _activations_to_note_events(H, note_midi_list, times, profile)

    return output_sequence


def get_profile_display_info(profile_name):
    """Helper for the UI"""
    profile = POLYPHONIC_PROFILES.get(profile_name, POLYPHONIC_PROFILES["piano_poly"])
    return {
        "label": profile["label"],
        "fmin": profile["fmin"],
        "fmax": profile["fmax"],
    }