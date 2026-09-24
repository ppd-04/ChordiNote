import librosa
import numpy as np
#vabtesilam shobkichuke generalize korar bodole user ke choice dibo bivinnovabe analyze kora. taile hoito ektu easier hobe i dunno. option o barbe.xD
ANALYSIS_PROFILES = {
    "clean_melody": {
        "label": "Clean Melody",
        "fmin": "C2",
        "fmax": "C7",
        "hop_length": 128,
        "frame_length": 4096,
        "confidence": 0.30,
        "onset_spacing": 0.08,
        "max_gap": 0.12,
        "reconstruction_gap": 0.16,
        "transition_rate": 12,
    },
    "fast_melody": {
        "label": "Fast / Ornamented Melody",
        "fmin": "C2",
        "fmax": "D7",
        "hop_length": 64,
        "frame_length": 2048,
        "confidence": 0.08,
        "onset_spacing": 0.025,
        "max_gap": 0.06,
        "reconstruction_gap": 0.45,
        "transition_rate": 24,
    },
    "noisy_melody": {
        "label": "Noisy / YouTube Recording",
        "fmin": "C2",
        "fmax": "D7",
        "hop_length": 128,
        "frame_length": 4096,
        "confidence": 0.45,
        "onset_spacing": 0.08,
        "max_gap": 0.10,
        "reconstruction_gap": 0.20,
        "transition_rate": 12,
    },
    "piano": {
        "label": "Piano Music",
        "fmin": "A0",          # Lowest piano key (27.5 Hz)
        "fmax": "C8",          # Highest piano key (4186 Hz)
        "hop_length": 256,     # Larger hop = better low-freq resolution
        "frame_length": 8192,  # Larger frame = better frequency resolution for bass
        "confidence": 0.25,    # Slightly lower to catch bass notes
        "onset_spacing": 0.06,
        "max_gap": 0.15,
        "reconstruction_gap": 0.18,
        "transition_rate": 10, # Lower = smoother pitch tracking
    },
}


def process_audio_file(file_path, profile_name="clean_melody"):
    profile = ANALYSIS_PROFILES.get(profile_name, ANALYSIS_PROFILES["clean_melody"])
    sample_rate = 22050
    hop_length = profile["hop_length"]#bishesh droshtobbo: function er vetor parameter eshober maane ami jani na
    frame_length = profile["frame_length"]
    duration = librosa.get_duration(path=file_path)
    chunk_seconds = 30
    context_seconds = 2
    frame_times = []
    frame_frequencies = []
    frame_midi = []
    frame_rms = []
    frame_onsets = []

    for core_start in np.arange(0, duration, chunk_seconds):
        core_end = min(core_start + chunk_seconds, duration)
        analysis_start = max(0, core_start - context_seconds)
        analysis_duration = min(duration, core_end + context_seconds) - analysis_start#onek boro gaan er jonno array banaite gele array pura prithibi ultai boro hoye jai jormungandr er moto tai choto choto kore vangsi.
        y, sr = librosa.load(
            file_path,
            sr=sample_rate,
            mono=True,
            offset=float(analysis_start),
            duration=float(analysis_duration),
        )
        y, trim_indices = librosa.effects.trim(y, top_db=50)
        analysis_start += trim_indices[0] / sr
        if len(y) < frame_length:
            continue

        # f0, voiced_flag, probabilities = librosa.pyin(
        #     y,
        #     fmin=librosa.note_to_hz(profile["fmin"]),
        #     fmax=librosa.note_to_hz(profile["fmax"]),
        #     sr=sr,
        #     frame_length=frame_length,
        #     hop_length=hop_length,
        #     fill_na=None,
        #     max_transition_rate=profile["transition_rate"],
        # )

        # changing here

                # Adaptive frame length: use larger window for better bass detection
        # Low frequencies need longer windows to resolve properly
        fmin_hz = librosa.note_to_hz(profile["fmin"])
        adaptive_frame_length = frame_length
        
        # If we're looking for notes below C3 (130 Hz), increase frame length
        # Rule: need at least 2 full cycles of the lowest frequency in each frame
        min_frame_for_fmin = int(2.0 * sr / fmin_hz)
        # Round up to nearest power of 2 for FFT efficiency
        min_frame_power2 = int(2 ** np.ceil(np.log2(min_frame_for_fmin)))
        adaptive_frame_length = max(frame_length, min_frame_power2)

        f0, voiced_flag, probabilities = librosa.pyin(
            y,
            fmin=fmin_hz,
            fmax=librosa.note_to_hz(profile["fmax"]),
            sr=sr,
            frame_length=adaptive_frame_length,
            hop_length=hop_length,
            fill_na=None,
            max_transition_rate=profile["transition_rate"],
        )

        times = librosa.times_like(f0, sr=sr, hop_length=hop_length) + analysis_start
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        rms = np.pad(rms, (0, max(0, len(f0) - len(rms))), mode='edge')[:len(f0)]
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=hop_length,
            backtrack=False,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        onset_times += analysis_start
        onset_times = onset_times[np.r_[True, np.diff(onset_times) >= profile["onset_spacing"]]]

        midi = np.full(len(f0), np.nan)
        valid = voiced_flag & (probabilities >= profile["confidence"]) & ~np.isnan(f0) & (f0 > 0)#confidence er amount ekta threshold theke beshi hole nisi cause naile onek ghost note peye jai. ja shunse vabse but shune nai.
        midi[valid] = np.rint(librosa.hz_to_midi(f0[valid]))
        energy_floor = np.max(rms) * 10 ** (-55 / 20)
        valid &= rms > energy_floor #to eliminate kom energy er frame jegulake voice vabse but ashole dhor background noise ba nishshash
        midi[~valid] = np.nan

        for index in range(2, len(midi) - 2):
            if np.isnan(midi[index]):
                continue
            neighbors = midi[index - 2:index].tolist() + midi[index + 1:index + 3].tolist()
            neighbors = [value for value in neighbors if not np.isnan(value)]
            if len(neighbors) < 2:
                continue
            neighbor_median = float(np.median(neighbors))
            if abs(midi[index] - neighbor_median) >= 11:  # Octave error correction
                octave_candidate = midi[index] + (12 if midi[index] < neighbor_median else -12)
                if abs(octave_candidate - neighbor_median) < abs(midi[index] - neighbor_median):
                    midi[index] = octave_candidate

        corrected_frequencies = f0.copy()
        corrected = ~np.isnan(midi)
        corrected_frequencies[corrected] = librosa.midi_to_hz(midi[corrected])

        core_mask = (times >= core_start) & (times < core_end)
        chunk_onsets = np.zeros(len(f0), dtype=bool)
        for onset_time in onset_times:
            nearest_frame = np.argmin(np.abs(times - onset_time))
            chunk_onsets[nearest_frame] = True

        frame_times.extend(times[core_mask])
        frame_frequencies.extend(corrected_frequencies[core_mask])
        frame_midi.extend(midi[core_mask])
        frame_rms.extend(rms[core_mask])
        frame_onsets.extend(chunk_onsets[core_mask])

    if not frame_times:
        return []

    times = np.asarray(frame_times)
    frequencies = np.asarray(frame_frequencies)
    midi = np.asarray(frame_midi)
    rms = np.asarray(frame_rms)
    onsets = np.asarray(frame_onsets)
    frame_duration = hop_length / sample_rate
    states = np.full(len(times), 'REST', dtype=object)
    states[~np.isnan(midi)] = midi[~np.isnan(midi)]

    #majhemajhe emni emni gap diye de ekta contunuous note er moddhe, tokhon ektu gap gula tackle korar try kora, eta arektu better kora dorkar, apatoto choto gap er duipashe same note thakle fill kore di
    max_gap_frames = max(1, round(profile["max_gap"] / frame_duration))
    for index in range(1, len(states) - 1):
        if states[index] == 'REST':
            end = index
            while end < len(states) and states[end] == 'REST':
                end += 1
            if end - index <= max_gap_frames and index > 0 and end < len(states):
                if states[index - 1] == states[end] and isinstance(states[index - 1], (int, float, np.integer)):
                    states[index:end] = states[index - 1]
            index = end

    events = []
    start = 0
    for index in range(1, len(states)):
        split_for_onset = (
            onsets[index]
            and states[index] == states[index - 1]
            and states[index] != 'REST'
        )
        if states[index] != states[start] or split_for_onset:
            events.append((start, index))
            start = index
    events.append((start, len(states)))

    grouped_sequence = []#simply sequence reconstruct kora
    for start, end in events:
        state = states[start]
        if state == 'REST':
            note = 'REST'#rest maan jokhon kichu baaje na vabe, ekta marker diye rakha jaate reconstruct korte shubidha hoi
            frequency = '-'
        else:
            stable_frequencies = frequencies[start:end]
            stable_frequencies = stable_frequencies[~np.isnan(stable_frequencies)]
            frequency_value = float(np.median(stable_frequencies)) if len(stable_frequencies) else float(librosa.midi_to_hz(state))
            note = librosa.midi_to_note(state)
            frequency = f"{frequency_value:.1f} Hz"
        grouped_sequence.append({
            "timestamp": f"{times[start]:.2f}s - {times[end - 1] + frame_duration:.2f}s",
            "note": note,
            "frequency": frequency,
        })

    return grouped_sequence