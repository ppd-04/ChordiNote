import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


def remove_vocals_center_cancellation(input_path, output_path):
    # stereor jonno
    try:
        y, sr = librosa.load(input_path, sr=None, mono=False)

        if y.ndim == 1:

            return _fallback_mono_vocal_removal(y, sr, output_path)


        left = y[0]
        right = y[1]


        instrumental = left - right


        peak = np.max(np.abs(instrumental))
        if peak > 0:
            instrumental = instrumental / peak * 0.95

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, instrumental, sr, subtype='PCM_16')
        return True

    except Exception as e:
        print(f"Center cancellation error: {e}")
        return False


def _fallback_mono_vocal_removal(y, sr, output_path):
    n_fft = 2048
    hop_length = 512

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag, S_phase = librosa.magphase(S)


    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    vocal_mask = np.ones_like(S_mag)

    for i, f in enumerate(freqs):
        if 250 <= f <= 3500:

            vocal_mask[i, :] *= 0.5
        elif 150 <= f < 250 or 3500 < f <= 5000:

            vocal_mask[i, :] *= 0.75

    S_filtered = S_mag * vocal_mask * np.exp(1j * np.angle(S))


    y_filtered = librosa.istft(S_filtered, hop_length=hop_length, length=len(y))


    peak = np.max(np.abs(y_filtered))
    if peak > 0:
        y_filtered = y_filtered / peak * 0.95

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, y_filtered, sr, subtype='PCM_16')
    return True


def remove_vocals_nmf(input_path, output_path, n_components=12, n_iterations=50):
    try:
        y, sr = librosa.load(input_path, sr=22050, mono=True)

        n_fft = 2048
        hop_length = 512

        # Compute magnitude spectrogram
        S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S_mag = np.abs(S_complex)
        S_phase = np.angle(S_complex)

        n_freq_bins, n_frames = S_mag.shape

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        np.random.seed(42)
        W = np.random.rand(n_freq_bins, n_components) * 0.1 + 0.01
        H = np.random.rand(n_components, n_frames) * 0.1 + 0.01

        eps = 1e-10


        for iteration in range(n_iterations):

            numerator_H = W.T @ S_mag
            denominator_H = W.T @ W @ H + eps
            H *= (numerator_H / denominator_H)

   
            numerator_W = S_mag @ H.T
            denominator_W = W @ H @ H.T + eps
            W *= (numerator_W / denominator_W)


        vocal_components = []
        instrumental_components = []

        for k in range(n_components):
            template = W[:, k]


            template_norm = template / (np.max(template) + eps)



            low_mask = freqs < 200
            vocal_mask_freq = (freqs >= 200) & (freqs <= 3000)
            high_mask = freqs > 3000

            low_energy = np.sum(template_norm[low_mask]) if np.any(low_mask) else 0
            vocal_energy = np.sum(template_norm[vocal_mask_freq]) if np.any(vocal_mask_freq) else 0
            high_energy = np.sum(template_norm[high_mask]) if np.any(high_mask) else 0

            total_energy = low_energy + vocal_energy + high_energy + eps
            vocal_ratio = vocal_energy / total_energy
            low_ratio = low_energy / total_energy

            if vocal_ratio > 0.40 and low_ratio < 0.35:
                vocal_components.append(k)
            else:
                instrumental_components.append(k)


        if len(instrumental_components) == 0:

            ratios = []
            for k in range(n_components):
                template = W[:, k] / (np.max(W[:, k]) + eps)
                vocal_e = np.sum(template[(freqs >= 200) & (freqs <= 3000)])
                total_e = np.sum(template) + eps
                ratios.append((k, vocal_e / total_e))
            ratios.sort(key=lambda x: x[1])
            instrumental_components = [r[0] for r in ratios[:n_components // 2]]
            vocal_components = [r[0] for r in ratios[n_components // 2:]]

        H_instrumental = H.copy()
        for k in vocal_components:
            # ekbare remove korlamna
            H_instrumental[k, :] *= 0.1

        # istft
        S_instrumental_mag = W @ H_instrumental


        S_instrumental = S_instrumental_mag * np.exp(1j * S_phase)


        y_instrumental = librosa.istft(
            S_instrumental,
            hop_length=hop_length,
            length=len(y)
        )

        # Normalize
        peak = np.max(np.abs(y_instrumental))
        if peak > 0:
            y_instrumental = y_instrumental / peak * 0.95

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, y_instrumental, sr, subtype='PCM_16')
        return True

    except Exception as e:
        print(f"NMF vocal removal error: {e}")
        import traceback
        traceback.print_exc()
        return False