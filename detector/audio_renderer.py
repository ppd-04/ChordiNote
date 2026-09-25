import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

#eta ai ke bolsi array theke audio banaite ami kisu pori nai egula
_TIMESTAMP_PATTERN = re.compile(r"([0-9.]+)s\s*-\s*([0-9.]+)s")
_FREQUENCY_PATTERN = re.compile(r"([0-9.]+)\s*Hz")


def _parse_event(event):
    timestamp = _TIMESTAMP_PATTERN.fullmatch(event["timestamp"])
    if timestamp is None:
        return None

    start, end = map(float, timestamp.groups())
    if end <= start:
        return None
    if event["note"] == "REST":
        return start, end, [], []

    # Extract velocities if available, default to 0.7
    event_velocities = event.get("velocities", [])

    # Robust parsing: use midi_notes directly if available (handles chords properly)
    if "midi_notes" in event and event["midi_notes"]:
        pitches = [float(librosa.midi_to_hz(m)) for m in event["midi_notes"]]
        velocities = event_velocities if len(event_velocities) == len(pitches) else [0.7] * len(pitches)
        return start, end, pitches, velocities

    # Fallback parsing for V1 compatibility
    freq_strs = _FREQUENCY_PATTERN.findall(event.get("frequency", ""))
    if freq_strs:
        pitches = [float(f) for f in freq_strs]
    else:
        note_names = event["note"].split(" + ")
        pitches = [float(librosa.note_to_hz(n.strip())) for n in note_names if n.strip() != "REST"]

    velocities = event_velocities if len(event_velocities) == len(pitches) else [0.7] * len(pitches)
    return start, end, pitches, velocities


def _synth_note(frequencies, velocities, duration, sample_rate):
    if not isinstance(frequencies, list):
        frequencies = [frequencies]
    if not isinstance(velocities, list) or len(velocities) != len(frequencies):
        velocities = [0.7] * len(frequencies)
        
    sample_count = max(1, round(duration * sample_rate))
    time = np.arange(sample_count) / sample_rate
    
    # Mix all frequencies (chords) with individual velocity weighting & register-adaptive timbre
    signal = np.zeros(sample_count)
    for freq, vel in zip(frequencies, velocities):
        phase = 2 * np.pi * freq * time
        
        # Register-dependent harmonic rolloff
        if freq < 250:  # Bass notes: richer overtone spectrum
            harmonics = (
                np.sin(phase)
                + 0.65 * np.sin(2 * phase)
                + 0.40 * np.sin(3 * phase)
                + 0.25 * np.sin(4 * phase)
                + 0.15 * np.sin(5 * phase)
                + 0.08 * np.sin(6 * phase)
            )
        elif freq < 600:  # Mid-range notes
            harmonics = (
                np.sin(phase)
                + 0.45 * np.sin(2 * phase)
                + 0.20 * np.sin(3 * phase)
                + 0.08 * np.sin(4 * phase)
            )
        else:  # High / Treble notes: cleaner fundamental
            harmonics = (
                np.sin(phase)
                + 0.25 * np.sin(2 * phase)
                + 0.05 * np.sin(3 * phase)
            )
            
        signal += vel * harmonics
        
    # Normalize chord volume
    if len(frequencies) > 0:
        signal /= np.sqrt(len(frequencies))

    # Piano envelope: sharp attack, exponential decay (mimicking string damping/sustain)
    attack = min(round(0.015 * sample_rate), sample_count // 2)
    release = min(round(0.050 * sample_rate), sample_count // 2)
    
    envelope = np.ones(sample_count)
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack, endpoint=False)
        
    # Exponential decay throughout the note (sustain pedal effect)
    decay_rate = 1.5
    if sample_count > attack:
        envelope[attack:] = np.exp(-decay_rate * time[:sample_count - attack])
    
    if release > 0:
        release_env = np.linspace(1, 0, release, endpoint=False)
        envelope[-release:] *= release_env
        
    return 0.35 * signal * envelope


def render_reconstructed_audio(
    notes,
    output_path,
    sample_rate=22050,
    bridge_gap_seconds=0.16,
):
    events = [_parse_event(event) for event in notes]
    events = [event for event in events if event is not None]
    if not events:
        return False

    # Short detector dropouts bridging
    for index in range(1, len(events) - 1):
        previous = events[index - 1]
        current = events[index]
        following = events[index + 1]
        if (
            len(current[2]) == 0
            and current[1] - current[0] <= bridge_gap_seconds
            and len(previous[2]) > 0
            and len(following[2]) > 0
        ):
            events[index - 1] = (previous[0], current[1], previous[2], previous[3])
            events[index] = (current[0], current[1], following[2], following[3])

    duration = max(end for _, end, _, _ in events)
    audio = np.zeros(max(1, round(duration * sample_rate)), dtype=np.float32)
    for start, end, frequencies, velocities in events:
        if not frequencies:
            continue
        start_sample = max(0, round(start * sample_rate))
        end_sample = min(len(audio), round(end * sample_rate))
        if end_sample > start_sample:
            note = _synth_note(frequencies, velocities, (end_sample - start_sample) / sample_rate, sample_rate)
            audio[start_sample:end_sample] += note[:end_sample - start_sample]

    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio *= 0.95 / peak
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    return True


def calculate_reconstruction_error(original_path, reconstructed_path, sample_rate=22050):
    """
    Calculate signal reconstruction error metrics between original and reconstructed audio.
    Metrics calculated:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Log-Spectral Distance (Spectral Error)
    - Signal Match Score (%)
    """
    try:
        y_orig, _ = librosa.load(original_path, sr=sample_rate, mono=True)
        y_rec, _ = librosa.load(reconstructed_path, sr=sample_rate, mono=True)

        if len(y_orig) == 0 or len(y_rec) == 0:
            return None

        # Align audio array lengths
        target_len = max(len(y_orig), len(y_rec))
        y_orig = np.pad(y_orig, (0, target_len - len(y_orig)))
        y_rec = np.pad(y_rec, (0, target_len - len(y_rec)))

        # Normalize peak amplitudes for fair signal comparison
        p_orig = np.max(np.abs(y_orig))
        p_rec = np.max(np.abs(y_rec))
        if p_orig > 0:
            y_orig = y_orig / p_orig
        if p_rec > 0:
            y_rec = y_rec / p_rec

        # 1. Time Domain Mean Squared Error (MSE) & RMSE
        mse = float(np.mean((y_orig - y_rec) ** 2))
        rmse = float(np.sqrt(mse))

        # 2. Phase-Invariant Spectral Cosine Similarity (Match Score %)
        # Human ears hear pitch & frequency energy over time (spectrogram magnitude),
        # not raw sample phase alignment. Time-domain dot product cancels out on tiny ms phase shifts.
        stft_orig = np.abs(librosa.stft(y_orig))
        stft_rec = np.abs(librosa.stft(y_rec))

        flat_orig = stft_orig.flatten()
        flat_rec = stft_rec.flatten()

        dot_spec = np.dot(flat_orig, flat_rec)
        norm_orig = np.linalg.norm(flat_orig)
        norm_rec = np.linalg.norm(flat_rec)

        if norm_orig > 0 and norm_rec > 0:
            cos_sim = float(dot_spec / (norm_orig * norm_rec))
            match_score = float(np.clip(cos_sim, 0.0, 1.0) * 100.0)
        else:
            match_score = 0.0

        # 3. Harmonic Pitch Match Score (Chromagram Cosine Similarity)
        # Chroma features group energy into the 12 musical pitch classes (C, C#, D... B),
        # making it resilient to timber/overtone differences and focusing strictly on pitch accuracy.
        chroma_orig = librosa.feature.chroma_stft(y=y_orig, sr=sample_rate)
        chroma_rec = librosa.feature.chroma_stft(y=y_rec, sr=sample_rate)

        flat_chroma_orig = chroma_orig.flatten()
        flat_chroma_rec = chroma_rec.flatten()

        dot_chroma = np.dot(flat_chroma_orig, flat_chroma_rec)
        norm_chroma_orig = np.linalg.norm(flat_chroma_orig)
        norm_chroma_rec = np.linalg.norm(flat_chroma_rec)

        if norm_chroma_orig > 0 and norm_chroma_rec > 0:
            chroma_sim = float(dot_chroma / (norm_chroma_orig * norm_chroma_rec))
            pitch_match_score = float(np.clip(chroma_sim, 0.0, 1.0) * 100.0)
        else:
            pitch_match_score = 0.0

        # 4. Spectral Distance / Log-Spectral Distance (LSD)
        log_diff = np.log10(stft_orig + 1e-4) - np.log10(stft_rec + 1e-4)
        spectral_error = float(np.sqrt(np.mean(log_diff ** 2)))

        return {
            "mse": round(mse, 6),
            "rmse": round(rmse, 4),
            "spectral_error": round(spectral_error, 4),
            "match_score": round(match_score, 1),
            "pitch_match_score": round(pitch_match_score, 1)
        }
    except Exception as err:
        print(f"Failed to calculate reconstruction metrics: {err}")
        return None