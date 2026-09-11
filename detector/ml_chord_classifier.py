import os
import pickle
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORD_TEMPLATES = {
    'C':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C  E  G
    'C#':   [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # C# F  G#
    'D':    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # D  F# A
    'D#':   [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # D# G  A#
    'E':    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # E  G# B
    'F':    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # F  A  C
    'F#':   [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # F# A# C#
    'G':    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # G  B  D
    'G#':   [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # G# C  D#
    'A':    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # A  C# E
    'A#':   [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # A# D  F
    'B':    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  # B  D# F#

    # Minor chords: root + minor 3rd (3 semitones) + perfect 5th (7 semitones)
    'Cm':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # C  Eb G
    'C#m':  [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # C# E  G#
    'Dm':   [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # D  F  A
    'D#m':  [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],  # D# F# A#
    'Em':   [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # E  G  B
    'Fm':   [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # F  Ab C
    'F#m':  [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # F# A  C#
    'Gm':   [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # G  Bb D
    'G#m':  [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],  # G# B  D#
    'Am':   [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # A  C  E
    'A#m':  [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # A# C# F
    'Bm':   [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # B  D  F#
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'knn_chord_model.pkl')

# training data generate kore knn classifier ke train kora hoy jeta ashe pasher 7 ta neighbour dekhe
# detect kore kon chord houar probabilit beshi

def generate_training_data(samples_per_chord=200, noise_level=0.15):
    X = []
    y = [] 

    for chord_name, template in CHORD_TEMPLATES.items():
        template = np.array(template, dtype=float)

        for _ in range(samples_per_chord):
            chroma = template.copy()

            # gausian noise add kora
            noise = np.random.normal(0, noise_level, 12)
            chroma += noise

            # randomly root node ke baray deya

            active_indices = np.where(template == 1)[0]
            for idx in active_indices:
                chroma[idx] *= np.random.uniform(0.6, 1.2)

            # kisu note ke weak kore deya
            inactive_indices = np.where(template == 0)[0]
            for idx in inactive_indices:
                if np.random.random() < 0.2: 
                    chroma[idx] += np.random.uniform(0.05, 0.25)

            #weak 
            if len(active_indices) > 0 and np.random.random() < 0.3:
                weak_idx = np.random.choice(active_indices)
                chroma[weak_idx] *= np.random.uniform(0.2, 0.5)

        
            chroma = np.clip(chroma, 0, 1)

            max_val = np.max(chroma)
            if max_val > 0:
                chroma = chroma / max_val

            X.append(chroma)
            y.append(chord_name)

    return np.array(X), np.array(y)




def train_and_save_model():
    print("Training Data Generating.................")
    X, y = generate_training_data(samples_per_chord=300, noise_level=0.15)
    print(f"   Created {len(X)} training examples for {len(CHORD_TEMPLATES)} chords")


    # metric cosine karon oi vector er moddhe angle dekhe distance er value r bodole
    print("Training KNN classifier...")
    knn = KNeighborsClassifier(
        n_neighbors=7,
        metric='cosine',
        weights='distance'
    )
    knn.fit(X, y)
    
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    print(f"   Cross-validation accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(knn, f)
    print(f"   Model saved to {MODEL_PATH}")

    return knn

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Train hoynai, Ebar hobe..")
        return train_and_save_model()
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def extract_chroma_segments(file_path, hop_seconds=1.0, max_duration=60):

    # cqt: constant q transform, fft r moto but logarithmic freq use kroe, gaaner jonno valo
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=max_duration)
    duration = float(len(y) / sr)


    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr,
        hop_length=int(sr * hop_seconds),
        n_chroma=12,
        norm=2  # L2 normalization
    )

    segments = []
    num_frames = chroma.shape[1]

    for i in range(num_frames):
        time_start = i * hop_seconds
        time_end = min((i + 1) * hop_seconds, duration)


        chroma_vector = chroma[:, i]

        max_val = np.max(chroma_vector)
        if max_val > 0:
            chroma_vector = chroma_vector / max_val

        segments.append({
            'time_start': round(time_start, 2),
            'time_end': round(time_end, 2),
            'chroma': chroma_vector.tolist(),
        })

    return segments

def predict_chord_template(chroma_vector):

    # normal method
    best_chord = None
    best_score = -1

    for chord_name, template in CHORD_TEMPLATES.items():
        template = np.array(template, dtype=float)
        chroma = np.array(chroma_vector, dtype=float)


        dot_product = np.dot(chroma, template)
        norm_chroma = np.linalg.norm(chroma)
        norm_template = np.linalg.norm(template)

        if norm_chroma > 0 and norm_template > 0:
            similarity = dot_product / (norm_chroma * norm_template)
        else:
            similarity = 0

        if similarity > best_score:
            best_score = similarity
            best_chord = chord_name

    return best_chord, round(float(best_score), 3)


def predict_chord_ml(chroma_vector, model):

    chroma = np.array(chroma_vector, dtype=float).reshape(1, -1)

    prediction = model.predict(chroma)[0]
    probabilities = model.predict_proba(chroma)[0]


    confidence = float(np.max(probabilities))

    class_names = model.classes_
    prob_dict = {}
    top_indices = np.argsort(probabilities)[-3:][::-1]
    for idx in top_indices:
        prob_dict[class_names[idx]] = round(float(probabilities[idx]), 3)

    return prediction, round(confidence, 3), prob_dict


def analyze_chords(file_path, hop_seconds=1.5, max_duration=30):

    model = load_model()


    segments = extract_chroma_segments(file_path, hop_seconds, max_duration)

    results = []
    agree_count = 0

    for seg in segments:
        chroma = seg['chroma']


        t_chord, t_conf = predict_chord_template(chroma)


        m_chord, m_conf, m_top3 = predict_chord_ml(chroma, model)


        agree = (t_chord == m_chord)
        if agree:
            agree_count += 1

        results.append({
            'time': f"{seg['time_start']:.1f}s - {seg['time_end']:.1f}s",
            'template_chord': t_chord,
            'template_confidence': t_conf,
            'ml_chord': m_chord,
            'ml_confidence': m_conf,
            'ml_top3': m_top3,
            'agree': agree,
        })


    total = len(results) if len(results) > 0 else 1
    agreement_rate = round(agree_count / total, 2)

    ml_avg = round(
        sum(r['ml_confidence'] for r in results) / total, 2
    ) if results else 0

    template_avg = round(
        sum(r['template_confidence'] for r in results) / total, 2
    ) if results else 0

    return {
        'segments': results,
        'agreement_rate': agreement_rate,
        'ml_avg_confidence': ml_avg,
        'template_avg_confidence': template_avg,
        'total_segments': len(results),
    }