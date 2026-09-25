import numpy as np
import librosa
from scipy.signal import medfilt

def process_audio_voice(file_path, noise_gate_db=-45.0, min_duration_sec=0.04):
    """
    Monophonic pitch tracker designed specifically for vocal tracks.
    Uses pYIN to track pitch, applies an energy noise gate to ignore instrument bleed,
    and quantizes the melody to discrete MIDI notes.
    """
    SR = 22050
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)
    except Exception as e:
        print(f"[voice_engine] Error loading {file_path}: {e}")
        return []

    # 1. Monophonic Pitch Tracking using pYIN
    # fmin=65 (C2), fmax=1046 (C6) - standard vocal range
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, sr=sr, fmin=65, fmax=1046, frame_length=2048, hop_length=512
    )

    times = librosa.times_like(f0, sr=sr, hop_length=512)

    # 2. Noise Gate (Ignore faint instrument bleed during vocal rests)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    
    # Check if audio is completely silent
    if np.max(rms) == 0:
        return []
        
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Trust pYIN's HMM-backed voiced_flag instead of manually thresholding probability
    valid_mask = voiced_flag & (rms_db >= noise_gate_db)
    f0_gated = np.where(valid_mask, f0, np.nan)

    # 3. Vibrato Smoothing (Median Filter)
    # Fill NaNs temporarily for smoothing, then re-apply mask
    f0_interp = np.copy(f0_gated)
    nan_idx = np.isnan(f0_interp)
    if np.any(nan_idx) and not np.all(nan_idx):
        f0_interp[nan_idx] = np.interp(np.flatnonzero(nan_idx), np.flatnonzero(~nan_idx), f0_interp[~nan_idx])
    
    # 7-frame median filter (~160ms) to aggressively flatten vibrato
    f0_smoothed = medfilt(f0_interp, kernel_size=7)
    f0_smoothed = np.where(valid_mask, f0_smoothed, np.nan)

    # 4. Quantize to discrete MIDI semitones
    midi_continuous = librosa.hz_to_midi(f0_smoothed)
    midi_quantized = np.where(valid_mask, np.round(midi_continuous).astype(int), -1)

    # 5. Extract contiguous Note Events
    events = []
    current_note = -1
    start_time = 0.0

    for i, midi in enumerate(midi_quantized):
        if midi != current_note:
            # Save the previous note event if it existed
            if current_note != -1:
                end_time = times[i]
                duration = end_time - start_time
                if duration >= min_duration_sec:
                    events.append({
                        "start": start_time,
                        "end": end_time,
                        "midi": current_note
                    })
            # Start new note
            current_note = midi
            start_time = times[i]

    # Handle the very last note
    if current_note != -1:
        end_time = times[-1]
        duration = end_time - start_time
        if duration >= min_duration_sec:
            events.append({
                "start": start_time,
                "end": end_time,
                "midi": current_note
            })

    # 6. Format Output to match Visual Piano schema
    out = []
    last_end = 0.0

    for ev in events:
        start_t = ev["start"]
        end_t = ev["end"]
        midi_val = ev["midi"]
        
        # Insert REST if there's a significant gap
        if start_t - last_end > 0.05:
            out.append({
                "timestamp": f"{last_end:.2f}s - {start_t:.2f}s",
                "note": "REST"
            })
            
        note_name = librosa.midi_to_note(midi_val)
        freq = librosa.midi_to_hz(midi_val)
        
        # Velocity proxy based on mean RMS in the note's window
        # Get frame indices for this note
        start_idx = np.searchsorted(times, start_t)
        end_idx = np.searchsorted(times, end_t)
        if end_idx > start_idx:
            mean_rms = np.mean(rms[start_idx:end_idx])
            # Scale velocity from 0.4 to 1.0 based on RMS
            vel = 0.4 + 0.6 * (mean_rms / np.max(rms))
        else:
            vel = 0.7

        out.append({
            "timestamp": f"{start_t:.2f}s - {end_t:.2f}s",
            "note": note_name,
            "frequency": f"{freq:.1f} Hz",
            "is_chord": False,
            "midi_notes": [int(midi_val)],
            "velocities": [min(1.0, float(vel))],
        })
        
        last_end = end_t
        
    return out
