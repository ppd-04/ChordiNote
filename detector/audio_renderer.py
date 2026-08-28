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
        return start, end, None

    frequency = _FREQUENCY_PATTERN.search(event.get("frequency", ""))
    if frequency is not None:
        pitch = float(frequency.group(1))
    else:
        pitch = float(librosa.note_to_hz(event["note"]))
    return start, end, pitch


def _synth_note(frequency, duration, sample_rate):
    sample_count = max(1, round(duration * sample_rate))
    time = np.arange(sample_count) / sample_rate
    vibrato = 1 + 0.0025 * np.sin(2 * np.pi * 5.2 * time)
    phase = 2 * np.pi * np.cumsum(frequency * vibrato) / sample_rate
    signal = (
        np.sin(phase)
        + 0.32 * np.sin(2 * phase)
        + 0.14 * np.sin(3 * phase)
        + 0.06 * np.sin(4 * phase)
    )

    attack = min(round(0.008 * sample_rate), sample_count // 2)
    release = min(round(0.012 * sample_rate), sample_count // 2)
    envelope = np.ones(sample_count)
    if attack:
        envelope[:attack] = np.linspace(0, 1, attack, endpoint=False)
    if release:
        envelope[-release:] = np.linspace(1, 0, release, endpoint=False)
    return 0.32 * signal * envelope


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

    # Short detector dropouts are usually not intentional rests. Extend the
    # preceding note across them, but preserve longer musical pauses.
    for index in range(1, len(events) - 1):
        previous = events[index - 1]
        current = events[index]
        following = events[index + 1]
        if (
            current[2] is None
            and current[1] - current[0] <= bridge_gap_seconds
            and previous[2] is not None
            and following[2] is not None
        ):
            events[index - 1] = (previous[0], current[1], previous[2])
            events[index] = (current[0], current[1], following[2])

    duration = max(end for _, end, _ in events)
    audio = np.zeros(max(1, round(duration * sample_rate)), dtype=np.float32)
    for start, end, frequency in events:
        if frequency is None:
            continue
        start_sample = max(0, round(start * sample_rate))
        end_sample = min(len(audio), round(end * sample_rate))
        if end_sample > start_sample:
            note = _synth_note(frequency, (end_sample - start_sample) / sample_rate, sample_rate)
            audio[start_sample:end_sample] += note[:end_sample - start_sample]

    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio *= 0.95 / peak
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    return True