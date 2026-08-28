import io
import numpy as np
import wave
import struct
import base64


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
    root = NOTE_TO_MIDI.get(root_name, 60)
    return [root + interval for interval in intervals]



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


NORMAL_CHORDS = [
    ('C', ''), ('D', ''), ('E', ''), ('F', ''), ('G', ''), ('A', ''), ('B', ''),
    ('C', 'm'), ('D', 'm'), ('E', 'm'), ('F', 'm'), ('G', 'm'), ('A', 'm'), ('B', 'm'),
]

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

    return f"{root}{chord_type}"


def midi_to_frequency(midi_note):
    # formula diye midi note number to frequency
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def generate_chord_audio(root, chord_type, duration=2.5, sample_rate=22050):

    intervals = CHORD_TYPES.get(chord_type, [0, 4, 7])
    midi_notes = _build_chord_midi(root, intervals)


    num_samples = int(duration * sample_rate)

    # time array r 
    t = np.linspace(0, duration, num_samples, endpoint=False)

    #shobgula mix
    signal = np.zeros(num_samples)

    for midi_note in midi_notes:
        freq = midi_to_frequency(midi_note)

        note_signal = np.sin(2 * np.pi * freq * t)

        # ken jani eshb habijabi korse. maybe valo shunate 
        note_signal += 0.4 * np.sin(2 * np.pi * 2 * freq * t)
        note_signal += 0.15 * np.sin(2 * np.pi * 3 * freq * t)
        note_signal += 0.06 * np.sin(2 * np.pi * 4 * freq * t)

        signal += note_signal

    # normalize korar jonno
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7

    # smooth hobe shunte. fade in fad eout
    attack_samples = min(int(0.05 * sample_rate), num_samples // 4)

    release_samples = min(int(0.2 * sample_rate), num_samples // 4)

    envelope = np.ones(num_samples)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)

    signal *= envelope


    signal_int = np.int16(signal * 32767)


    buffer = io.BytesIO()
    with wave.open(buffer, 'w') as wav_file:
        wav_file.setnchannels(1)           # Mono
        wav_file.setsampwidth(2)           # 2 bytes = 16 bit
        wav_file.setframerate(sample_rate)
        # struct.pack converts each integer to bytes
        wav_file.writeframes(struct.pack(f'<{len(signal_int)}h', *signal_int))


    wav_bytes = buffer.getvalue()
    return base64.b64encode(wav_bytes).decode('utf-8')



def generate_reference_c():
    return generate_chord_audio('C', '', duration=3.0)