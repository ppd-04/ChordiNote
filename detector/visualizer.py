import io
import base64
import numpy as np
import librosa

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib 
matplotlib.use('Agg')
import librosa.display
import matplotlib as plt

def _to_list(arr):
    # python type e convert or smth

    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return list(arr)

def _freq_to_note_name(freq):

    if freq <= 0 or np.isnan(freq):
        return None
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi = 12 * np.log2(freq / 440.0) + 69
    midi_round = int(round(midi))
    if midi_round < 0 or midi_round > 127:
        return None
    return note_names[midi_round % 12] + str(midi_round // 12 - 1)

def generate_all_visualizations(file_path, max_duration=60):
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=max_duration)
    duration = float(len(y)/sr)
    return {
        'duration': round(duration, 2),
        'sample_rate': sr,
        'waveform': _generate_waveform(y, sr, duration),
        'fft': _generate_fft_spectrum(y, sr),
        'spectrogram': _generate_spectrogram_image(y, sr),
        'rms': _generate_rms_energy(y, sr, duration),
        'pitch_contour': _generate_pitch_contour(y, sr, duration),
    }

def _generate_waveform(y, sr, duration, num_points=2000):
    # prottek ta point er peak amp niye 
    window_size = max(1, len(y) // num_points)

    times = []
    amplitudes = []

    for i in range(0, len(y) - window_size, window_size):
        window = y[i:i + window_size]
        peak_idx = np.argmax(np.abs(window))
        times.append(round(float(i + peak_idx) / sr, 4))
        amplitudes.append(round(float(window[peak_idx]), 4))

    return {
        'times': times,
        'amplitudes': amplitudes,
    }


def _generate_fft_spectrum(y, sr, window_seconds=2.0, max_freq=4000):
    frame_length = sr // 10 
    hop = frame_length // 2
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]

    if len(rms) > 0:
        loudest_frame = np.argmax(rms)
        loudest_time = librosa.frames_to_time(loudest_frame, sr=sr, hop_length=hop)
    else:
        loudest_time = 0


    start = max(0, int((loudest_time - window_seconds / 2) * sr))
    end = min(len(y), start + int(window_seconds * sr))
    segment = y[start:end]

    if len(segment) < 256:
        segment = y[:min(len(y), int(window_seconds * sr))]

    window = np.hanning(len(segment))
    windowed = segment * window


    fft_result = np.fft.rfft(windowed)


    magnitudes = np.abs(fft_result)
    magnitudes_db = 20 * np.log10(magnitudes + 1e-10)


    frequencies = np.fft.rfftfreq(len(windowed), d=1.0 / sr)


    freq_mask = frequencies <= max_freq
    frequencies = frequencies[freq_mask]
    magnitudes_db = magnitudes_db[freq_mask]
    max_db = np.max(magnitudes_db)
    magnitudes_db = magnitudes_db - max_db


    step = max(1, len(frequencies) // 500)
    frequencies = frequencies[::step]
    magnitudes_db = magnitudes_db[::step]


    search_start = max(1, int(50 / (sr / len(windowed))))  
    if search_start < len(magnitudes_db):
        peak_idx = search_start + np.argmax(magnitudes_db[search_start:])
        peak_freq = round(float(frequencies[peak_idx]), 1)
    else:
        peak_freq = 0

    peak_note = _freq_to_note_name(peak_freq) if peak_freq > 0 else None


    harmonics = []
    if peak_freq > 50:
        for h in range(2, 6): 
            h_freq = peak_freq * h
            if h_freq <= max_freq:
                harmonics.append(round(h_freq, 1))

    return {
        'frequencies': _to_list(np.round(frequencies, 1)),
        'magnitudes': _to_list(np.round(magnitudes_db, 1)),
        'peak_freq': peak_freq,
        'peak_note': peak_note,
        'harmonics': harmonics,
    }



def _generate_spectrogram_image(y, sr):
    """
    Generate a Mel spectrogram image as a base64 string.
    Thread-safe and compatible with Django.
    """
    try:
        # 1. Compute Mel Spectrogram (128 frequency bins, up to 8000 Hz)
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000,
            n_fft=2048, hop_length=512
        )

        # Convert power to decibels (dB)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # 2. Create Figure directly (Thread-safe, avoids plt.subplots() crashes)
        fig = Figure(figsize=(10, 3.5), dpi=100)
        FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Dark theme styling
        fig.patch.set_facecolor('#0f0a1a')
        ax.set_facecolor('#0f0a1a')

        # 3. Draw spectrogram heatmap
        img = librosa.display.specshow(
            S_dB, sr=sr, hop_length=512,
            x_axis='time', y_axis='mel',
            fmax=8000, cmap='magma', ax=ax
        )

        ax.set_title('Mel Spectrogram', color='#c4b5fd', fontsize=12, pad=10)
        ax.set_xlabel('Time (s)', color='#a8a3b8', fontsize=9)
        ax.set_ylabel('Frequency (Hz)', color='#a8a3b8', fontsize=9)
        ax.tick_params(colors='#a8a3b8', labelsize=8)

        for spine in ax.spines.values():
            spine.set_color('#2a2a4a')

        # 4. Add colorbar safely
        cbar = fig.colorbar(img, ax=ax)
        cbar.ax.tick_params(colors='#a8a3b8', labelsize=7)
        cbar.set_label('Amplitude (dB)', color='#a8a3b8', fontsize=8)

        fig.tight_layout()

        # 5. Export directly to in-memory PNG buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return f"data:image/png;base64,{img_base64}"

    except Exception as e:
        # Print the exact error in your terminal so it never fails silently
        print(f"[Spectrogram Generation Error]: {e}")
        import traceback
        traceback.print_exc()
        return None

# genjam hoy ekhane

# def _generate_spectrogram_image(y, sr):
#     try:
#         S = librosa.feature.melspectrogram(
#             y=y, sr=sr, n_mels=128, fmax=8000,
#             n_fft=2048, hop_length=512
#         )

#         S_dB = librosa.power_to_db(S, ref=np.max)

#         fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)

#         fig.patch.set_facecolor('#0f0a1a')
#         ax.set_facecolor('#0f0a1a')


#         img = librosa.display.specshow(
#             S_dB, sr=sr, hop_length=512,
#             x_axis='time', y_axis='mel',
#             fmax=8000, cmap='magma', ax=ax
#         )

#         ax.set_title('Mel Spectrogram', color='#c4b5fd', fontsize=12, pad=10)
#         ax.set_xlabel('Time (s)', color='#a8a3b8', fontsize=9)
#         ax.set_ylabel('Frequency (Hz)', color='#a8a3b8', fontsize=9)
#         ax.tick_params(colors='#a8a3b8', labelsize=8)

#         for spine in ax.spines.values():
#             spine.set_color('#2a2a4a')

#         cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
#         cbar.ax.tick_params(colors='#a8a3b8', labelsize=7)
#         cbar.ax.set_ylabel('Amplitude', color='#a8a3b8', fontsize=8)

#         plt.tight_layout()


#         buf = io.BytesIO()
#         fig.savefig(buf, format='png', bbox_inches='tight',
#                     facecolor=fig.get_facecolor(), edgecolor='none')
#         plt.close(fig) 


#         buf.seek(0)
#         img_base64 = base64.b64encode(buf.read()).decode('utf-8')

#         return f"data:image/png;base64,{img_base64}"

#     except Exception as e:

#         return None


def _generate_rms_energy(y, sr, duration, num_points=500):

    rms = librosa.feature.rms(
        y=y, frame_length=2048, hop_length=512
    )[0]

    times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=512
    )

    max_rms = np.max(rms) if np.max(rms) > 0 else 1
    rms_normalized = rms / max_rms


    step = max(1, len(rms) // num_points)
    times = times[::step]
    rms_normalized = rms_normalized[::step]

    return {
        'times': _to_list(np.round(times, 3)),
        'rms': _to_list(np.round(rms_normalized, 4)),
    }

def _generate_pitch_contour(y, sr, duration):
   
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=2048,
        hop_length=256,
        fill_na=0,  
    )

    times = librosa.frames_to_time(
        np.arange(len(f0)), sr=sr, hop_length=256
    )

    notes = []
    for freq in f0:
        if freq > 0 and voiced_flag[len(notes)] if len(notes) < len(voiced_flag) else False:
            notes.append(_freq_to_note_name(float(freq)))
        else:
            notes.append(None)

    step = max(1, len(f0) // 800)
    times = times[::step]
    f0 = f0[::step]
    notes = notes[::step]

    return {
        'times': _to_list(np.round(times, 3)),
        'frequencies': _to_list(np.round(f0, 1)),
        'notes': notes,
    }


