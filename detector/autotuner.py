import numpy as np
import librosa
import soundfile as sf
import psola

def autotune_audio(file_path, output_path, scale_notes=None, pull=1.0):
    SR = 22050
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)
    except Exception as e:
        print(f"[autotuner] Error loading {file_path}: {e}")
        return False

    hop_length = 512
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, sr=sr, fmin=65, fmax=1046, frame_length=2048, hop_length=hop_length
    )

    target_f0 = np.copy(f0)

    valid_idx = ~np.isnan(f0)
    if not np.any(valid_idx):
        sf.write(output_path, y, sr)
        return True

    f0_valid = f0[valid_idx]
    midi_valid = librosa.hz_to_midi(f0_valid)

    if scale_notes is None:
        midi_target = np.round(midi_valid)
    else:
        scale_notes = set(scale_notes)
        scale_extended = []
        for octave in range(2, 7):
            for n in scale_notes:
                scale_extended.append(n + 12 * (octave + 1))

        midi_target = []
        for midi in midi_valid:
            closest = min(scale_extended, key=lambda x: abs(x - midi))
            midi_target.append(closest)

        midi_target = np.array(midi_target)

    midi_corrected = midi_valid + pull * (midi_target - midi_valid)
    target_f0[valid_idx] = librosa.midi_to_hz(midi_corrected)

    try:
        y_shifted = psola.vocode(y, sample_rate=sr, target_pitch=target_f0)
        sf.write(output_path, y_shifted, sr)
        return True
    except Exception as e:
        print(f"[autotuner] PSOLA error: {e}")
        return False
