"""
Chord Audio Generator
Synthesizes chord audio using additive synthesis (sine waves).

How chords work musically:
- A chord = 3 or more notes played together
- Each note has a specific frequency
- We generate sine waves for each note and mix them together
- We add harmonics (overtones) to make it sound more natural

For example, C Major = C4 + E4 + G4 = 261.6Hz + 329.6Hz + 392.0Hz
"""

import io
import numpy as np
import wave
import struct
import base64


# ============================================
# CHORD DEFINITIONS
# Each chord is defined by its notes (MIDI numbers)
# MIDI 60 = C4 (Middle C), each +1 = one semitone up
# ============================================

# Helper: convert note name to MIDI number
NOTE_TO_MIDI = {
    'C': 60, 'C#': 61, 'Db': 61,
    'D': 62, 'D#': 63, 'Eb': 63,
    'E': 64, 'Fb': 64,
    'F': 65, 'F#': 66, 'Gb': 66,
    'G': 67, 'G#': 68, 'Ab': 68,
    'A': 69, 'A#': 70, 'Bb': 70,
    'B': 71, 'Cb': 71,
}


def _build_chord_midi(root_name, intervals):
    """
    Build MIDI note numbers for a chord.

    Parameters:
        root_name: e.g., 'C', 'F#', 'Bb'
        intervals: list of semitone intervals from root
                   Major = [0, 4, 7] (root, major 3rd, perfect 5th)
                   Minor = [0, 3, 7] (root, minor 3rd, perfect 5th)
    Returns:
        List of MIDI note numbers
    """
    root = NOTE_TO_MIDI.get(root_name, 60)
    return [root + interval for interval in intervals]


# Interval patterns for different chord types
CHORD_TYPES = {
    '':      [0, 4, 7],         # Major (C, D, E, etc.)
    'm':     [0, 3, 7],         # Minor (Cm, Dm, Em, etc.)
    '7':     [0, 4, 7, 10],     # Dominant 7th
    'm7':    [0, 3, 7, 10],     # Minor 7th
    'maj7':  [0, 4, 7, 11],     # Major 7th
    'dim':   [0, 3, 6],         # Diminished
    'aug':   [0, 4, 8],         # Augmented
    'sus2':  [0, 2, 7],         # Suspended 2nd
    'sus4':  [0, 5, 7],         # Suspended 4th
}

# Normal mode chords (common, beginner-friendly)
NORMAL_CHORDS = [
    ('C', ''), ('D', ''), ('E', ''), ('F', ''), ('G', ''), ('A', ''), ('B', ''),
    ('C', 'm'), ('D', 'm'), ('E', 'm'), ('F', 'm'), ('G', 'm'), ('A', 'm'), ('B', 'm'),
]

# Hard mode chords (adds sharps, flats, and complex types)
HARD_CHORDS = NORMAL_CHORDS + [
    ('C#', ''), ('D#', 'm'), ('F#', ''), ('G#', 'm'), ('Bb', ''), ('Eb', ''),
    ('Ab', ''), ('F#', 'm'), ('Bb', 'm'), ('Eb', 'm'),
    ('C', '7'), ('D', '7'), ('E', '7'), ('G', '7'), ('A', '7'),
    ('A', 'm7'), ('D', 'm7'), ('E', 'm7'),
    ('C', 'maj7'), ('F', 'maj7'), ('G', 'maj7'),
    ('B', 'dim'), ('C', 'dim'),
    ('C', 'sus2'), ('D', 'sus4'), ('G', 'sus4'),
    ('C', 'aug'),
]


def get_chord_name(root, chord_type):
    """Return the display name of a chord. e.g., ('C', 'm') → 'Cm'"""
    return f"{root}{chord_type}"


def midi_to_frequency(midi_note):
    """
    Convert MIDI note number to frequency in Hz.
    Formula: f = 440 * 2^((midi - 69) / 12)
    MIDI 69 = A4 = 440 Hz (standard tuning reference)
    """
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def generate_chord_audio(root, chord_type, duration=2.5, sample_rate=22050):
    """
    Generate audio for a chord as a base64-encoded WAV string.

    Why base64?
    - We can embed the audio directly in HTML as a data URI
    - No need to save files to disk or set up URLs
    - The browser can play it directly: <audio src="data:audio/wav;base64,...">

    Parameters:
        root: note name ('C', 'F#', 'Bb', etc.)
        chord_type: '' for major, 'm' for minor, '7', etc.
        duration: length in seconds
        sample_rate: samples per second (22050 is CD-quality / 2)

    Returns:
        base64 encoded WAV string
    """
    intervals = CHORD_TYPES.get(chord_type, [0, 4, 7])
    midi_notes = _build_chord_midi(root, intervals)

    # Total number of audio samples
    num_samples = int(duration * sample_rate)

    # Time array: [0, 1/sr, 2/sr, ..., duration]
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Mix all notes together
    signal = np.zeros(num_samples)

    for midi_note in midi_notes:
        freq = midi_to_frequency(midi_note)

        # Fundamental frequency (the note itself)
        note_signal = np.sin(2 * np.pi * freq * t)

        # Add harmonics to make it sound richer (like a real instrument)
        # 2nd harmonic at half volume, 3rd at quarter, etc.
        note_signal += 0.4 * np.sin(2 * np.pi * 2 * freq * t)
        note_signal += 0.15 * np.sin(2 * np.pi * 3 * freq * t)
        note_signal += 0.06 * np.sin(2 * np.pi * 4 * freq * t)

        signal += note_signal

    # Normalize to prevent clipping (keep amplitude between -1 and 1)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7

    # Apply envelope (smooth fade in/out to avoid clicks)
    # Attack: first 50ms fades in
    attack_samples = min(int(0.05 * sample_rate), num_samples // 4)
    # Release: last 200ms fades out
    release_samples = min(int(0.2 * sample_rate), num_samples // 4)

    envelope = np.ones(num_samples)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)

    signal *= envelope

    # Convert float signal to 16-bit integer PCM
    # WAV files store audio as integers, not floats
    signal_int = np.int16(signal * 32767)

    # Write to WAV format in memory (not to a file)
    buffer = io.BytesIO()
    with wave.open(buffer, 'w') as wav_file:
        wav_file.setnchannels(1)           # Mono
        wav_file.setsampwidth(2)           # 2 bytes = 16 bit
        wav_file.setframerate(sample_rate)
        # struct.pack converts each integer to bytes
        wav_file.writeframes(struct.pack(f'<{len(signal_int)}h', *signal_int))
        # '<' = little-endian byte order
        # 'h' = signed short (16-bit integer)
        # We pack ALL samples at once

    # Convert the WAV bytes to base64 string
    wav_bytes = buffer.getvalue()
    return base64.b64encode(wav_bytes).decode('utf-8')
    # base64.b64encode converts binary → ASCII text
    # .decode('utf-8') converts bytes → string


def generate_reference_c():
    """Generate the reference C Major chord for practice."""
    return generate_chord_audio('C', '', duration=3.0)