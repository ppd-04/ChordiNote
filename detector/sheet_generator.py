"""
Sheet Music & Lead Sheet Generator Module for PitchDetect
Converts detected pitch/chord events into standard ABC Notation and Piano Lead Sheet format.
Generates non-overlapping Treble & Bass staves with explicit note annotations.
"""

import librosa
import numpy as np

# Mapping MIDI pitch to ABC notation note string
PITCH_NAMES_ABC = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B']

def midi_to_abc_note(midi_pitch):
    if midi_pitch is None:
        return 'z'
    
    octave = (int(midi_pitch) // 12) - 1
    pitch_idx = int(midi_pitch) % 12
    base_name = PITCH_NAMES_ABC[pitch_idx]
    
    if octave == 4:
        return base_name
    elif octave > 4:
        ticks = "'" * (octave - 5)
        name_lower = base_name.lower()
        return f"{name_lower}{ticks}"
    else:  # octave < 4
        commas = "," * (4 - octave)
        return f"{base_name}{commas}"

def midi_list_to_abc(midi_list):
    if not midi_list:
        return 'z'
    
    midi_list = sorted(list(set(int(m) for m in midi_list)))
    if len(midi_list) == 1:
        return midi_to_abc_note(midi_list[0])
    
    abc_notes = [midi_to_abc_note(m) for m in midi_list]
    return f"[{''.join(abc_notes)}]"

def parse_timestamp_to_seconds(ts):
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            clean_ts = ts.split('-')[0].replace('s', '').strip()
            return float(clean_ts)
        except Exception:
            return 0.0
    return 0.0

def parse_duration_to_seconds(event, default_sec=0.25):
    if 'duration' in event and isinstance(event['duration'], (int, float)):
        return float(event['duration'])
    ts = event.get('timestamp', '')
    if isinstance(ts, str) and '-' in ts:
        try:
            parts = ts.split('-')
            t1 = float(parts[0].replace('s', '').strip())
            t2 = float(parts[1].replace('s', '').strip())
            return max(0.1, t2 - t1)
        except Exception:
            pass
    return default_sec

def parse_note_string_to_midis(note_str):
    if not note_str or note_str == 'REST':
        return []
    midis = []
    parts = [p.strip() for p in note_str.replace('+', ' ').replace(',', ' ').split() if p.strip()]
    for p in parts:
        try:
            m = int(librosa.note_to_midi(p))
            midis.append(m)
        except Exception:
            pass
    return midis

def estimate_bpm(file_path):
    if not file_path:
        return 120
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=30.0)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item() if tempo.size > 0 else 120.0
        bpm = float(tempo)
        if bpm < 50 or bpm > 220:
            bpm = 120.0
        return round(bpm)
    except Exception as e:
        print(f"BPM estimation fallback: {e}")
        return 120

def sanitize_key_for_abc(musical_key):
    if not musical_key or musical_key == 'Unknown':
        return 'C'
    clean_key = musical_key.replace(' Major', '').replace(' Minor', 'm').strip()
    return clean_key if clean_key else 'C'

def generate_piano_abc(detected_notes, file_path=None, musical_key="C Major"):
    """
    Generates a dual-staff (Treble + Bass) ABC notation string for piano sheet music,
    annotating note names directly above stave notes and creating a lead sheet summary.
    """
    if not detected_notes:
        detected_notes = []

    bpm = estimate_bpm(file_path)
    abc_key = sanitize_key_for_abc(musical_key)
    sec_per_eighth = 30.0 / float(bpm)
    
    treble_events = [] # (units, abc_symbol, annotation)
    bass_events = []   # (units, abc_symbol, annotation)
    note_summary = []  # Detailed note lead sheet
    
    prev_time = 0.0
    current_units_count = 0
    
    for idx, event in enumerate(detected_notes):
        timestamp = parse_timestamp_to_seconds(event.get('timestamp', 0.0))
        note_str = str(event.get('note', '')).strip()
        
        midi_list = event.get('midi_notes', [])
        if not midi_list:
            midi_list = parse_note_string_to_midis(note_str)
            
        gap = timestamp - prev_time
        if gap > sec_per_eighth * 1.5:
            rest_units = max(1, int(round(gap / sec_per_eighth)))
            treble_events.append((rest_units, 'z', ''))
            bass_events.append((rest_units, 'z', ''))
            current_units_count += rest_units
            
        duration_sec = parse_duration_to_seconds(event, sec_per_eighth)
        num_units = max(1, min(16, int(round(duration_sec / sec_per_eighth))))

        if note_str == 'REST' or not midi_list:
            treble_events.append((num_units, 'z', ''))
            bass_events.append((num_units, 'z', ''))
            prev_time = timestamp + (num_units * sec_per_eighth)
            current_units_count += num_units
            continue
            
        treble_midis = [m for m in midi_list if m >= 60]
        bass_midis = [m for m in midi_list if m < 60]
        
        t_annotation = note_str if treble_midis else ''
        b_annotation = note_str if (bass_midis and not treble_midis) else ''
        
        if treble_midis:
            t_symbol = midi_list_to_abc(treble_midis)
            treble_events.append((num_units, t_symbol, t_annotation))
        else:
            treble_events.append((num_units, 'z', ''))
            
        if bass_midis:
            b_symbol = midi_list_to_abc(bass_midis)
            bass_events.append((num_units, b_symbol, b_annotation))
        else:
            bass_events.append((num_units, 'z', ''))
            
        note_summary.append({
            'num': idx + 1,
            'time': f"{timestamp:.2f}s",
            'note': note_str,
            'frequency': event.get('frequency', '-'),
            'clef': 'Treble (Right)' if (treble_midis and not bass_midis) else ('Bass (Left)' if (bass_midis and not treble_midis) else 'Grand Staff'),
            'measure': (current_units_count // 8) + 1
        })
            
        prev_time = timestamp + (num_units * sec_per_eighth)
        current_units_count += num_units

    if not treble_events:
        treble_events = [(8, 'z', '')]
    if not bass_events:
        bass_events = [(8, 'z', '')]
        
    def build_voice_abc(events, voice_id, clef_name, voice_title):
        header_line = f"V:{voice_id} clef={clef_name} name=\"{voice_title}\"\n"
        measure_tokens = []
        current_measure_units = 0
        measure_count = 0
        
        for units, symbol, annotation in events:
            duration_suffix = "" if units == 1 else str(units)
            ann_str = f"\"{annotation}\"" if annotation else ""
            token = f"{ann_str}{symbol}{duration_suffix}"
            measure_tokens.append(token)
            current_measure_units += units
            
            if current_measure_units >= 8:
                measure_tokens.append(" | ")
                current_measure_units = 0
                measure_count += 1
                if measure_count % 4 == 0:
                    measure_tokens.append("\n")
                    
        if current_measure_units > 0 and (not measure_tokens or measure_tokens[-1] != " | "):
            measure_tokens.append(" |")
            measure_count += 1
            
        return header_line + " ".join(measure_tokens), measure_count

    treble_abc, t_measures = build_voice_abc(treble_events, "1", "treble", "Treble (Right Hand)")
    bass_abc, b_measures = build_voice_abc(bass_events, "2", "bass", "Bass (Left Hand)")
    
    header = f"""X:1
T:Piano Sheet Music & Notes
C:PitchDetect AI Generator
M:4/4
L:1/8
Q:1/4={bpm}
K:{abc_key}
"""
    full_abc = f"{header}\n{treble_abc}\n\n{bass_abc}\n"
    
    return {
        'abc_string': full_abc,
        'bpm': bpm,
        'key': abc_key,
        'total_measures': max(t_measures, b_measures),
        'note_summary': note_summary
    }
