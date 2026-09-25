import numpy as np
import librosa
import soundfile as sf
import psola

def autotune_audio(file_path, output_path, scale_notes=None, pull=1.0):
    """
    Autotunes an audio file using pYIN and PSOLA.
    
    file_path: path to the input audio file
    output_path: path to save the pitch-corrected audio
    scale_notes: list of allowed MIDI notes (pitch classes 0-11). If None, snaps to the nearest semitone (chromatic).
    pull: 0.0 to 1.0. How strongly to pull the pitch to the target note. 1.0 is hard tuning, 0.0 is no tuning.
    """
    SR = 22050
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)
    except Exception as e:
        print(f"[autotuner] Error loading {file_path}: {e}")
        return False
        
    # 1. Pitch Tracking
    hop_length = 512
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, sr=sr, fmin=65, fmax=1046, frame_length=2048, hop_length=hop_length
    )
    
    # 2. Determine target pitches
    target_f0 = np.copy(f0)
    
    valid_idx = ~np.isnan(f0)
    if not np.any(valid_idx):
        # No voiced frames found, just save the original
        sf.write(output_path, y, sr)
        return True
        
    f0_valid = f0[valid_idx]
    midi_valid = librosa.hz_to_midi(f0_valid)
    
    if scale_notes is None:
        # Snap to nearest semitone (Chromatic)
        midi_target = np.round(midi_valid)
    else:
        # Snap to nearest note in scale
        scale_notes = set(scale_notes)
        scale_extended = []
        for octave in range(2, 7):
            for n in scale_notes:
                scale_extended.append(n + 12 * (octave + 1)) # C2 is midi 36 (12 * 3), octave starting at C-1 (0) 
                
        # Better way to map to pitch classes
        # Find the closest pitch class from the scale
        midi_target = []
        for midi in midi_valid:
            # find closest note in all octaves 
            closest = min(scale_extended, key=lambda x: abs(x - midi))
            midi_target.append(closest)
            
        midi_target = np.array(midi_target)
        
    # Apply pull factor (0.0 = original, 1.0 = target)
    midi_corrected = midi_valid + pull * (midi_target - midi_valid)
    
    target_f0[valid_idx] = librosa.midi_to_hz(midi_corrected)
    
    # 3. Pitch shifting using PSOLA
    try:
        y_shifted = psola.vocode(y, sample_rate=sr, target_pitch=target_f0)
        sf.write(output_path, y_shifted, sr)
        return True
    except Exception as e:
        print(f"[autotuner] PSOLA error: {e}")
        return False
