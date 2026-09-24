# pyrefly: ignore [missing-import]
import numpy as np

# Krumhansl-Schmuckler key profiles
# These profiles represent the statistical distribution of pitch classes in major and minor keys.
# The 12 values correspond to the 12 pitch classes starting from the tonic (0 = Tonic, 1 = minor 2nd, etc.)

MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def detect_key_from_notes(detected_events):
    """
    Detect the musical key of a song based on the detected note events.
    Uses the Krumhansl-Schmuckler key-finding algorithm.
    """
    if not detected_events:
        return "Unknown"

    # 1. Build a Pitch Class Profile (Chroma vector) from the detected notes
    # We sum the duration of each pitch class played.
    pitch_class_durations = np.zeros(12)

    for event in detected_events:
        if event.get("note") == "REST" or "midi_notes" not in event:
            continue
            
        # Parse timestamp to get duration
        # Example timestamp: "0.00s - 1.50s"
        try:
            time_parts = event["timestamp"].replace("s", "").split("-")
            start = float(time_parts[0].strip())
            end = float(time_parts[1].strip())
            duration = max(0, end - start)
        except Exception:
            duration = 1.0 # fallback

        # Add duration to the respective pitch classes
        for midi_note in event["midi_notes"]:
            pitch_class = midi_note % 12
            pitch_class_durations[pitch_class] += duration

    # If no valid notes found
    if np.sum(pitch_class_durations) == 0:
        return "Unknown"

    # Normalize the profile
    pitch_class_durations = pitch_class_durations / np.max(pitch_class_durations)

    # 2. Correlate with all 24 possible keys (12 major, 12 minor)
    best_key = "Unknown"
    max_correlation = -1.0

    for i in range(12):
        # Shift profiles to match the current root note 'i'
        shifted_major = np.roll(MAJOR_PROFILE, i)
        shifted_minor = np.roll(MINOR_PROFILE, i)

        # Pearson correlation for Major
        corr_major = np.corrcoef(pitch_class_durations, shifted_major)[0, 1]
        if corr_major > max_correlation:
            max_correlation = corr_major
            best_key = f"{PITCH_CLASSES[i]} Major"

        # Pearson correlation for Minor
        corr_minor = np.corrcoef(pitch_class_durations, shifted_minor)[0, 1]
        if corr_minor > max_correlation:
            max_correlation = corr_minor
            best_key = f"{PITCH_CLASSES[i]} Minor"

    return best_key
