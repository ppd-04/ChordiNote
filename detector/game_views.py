import json
import random

from django.shortcuts import render

from .chord_generator import (
    NORMAL_CHORDS,
    HARD_CHORDS,
    get_chord_name,
    generate_chord_audio,
    generate_reference_c,
)


def game_lobby(request):
    return render(request, 'detector/game_lobby.html')


def chord_quiz(request):

    mode = request.GET.get('mode', 'normal')

    if mode == 'hard':
        chord_pool = HARD_CHORDS
    else:
        chord_pool = NORMAL_CHORDS
        mode = 'normal'

    num_questions = 10
    selected_chords = random.sample(
        chord_pool,
        min(num_questions, len(chord_pool))
    )

    questions = []
    for i, (root, chord_type) in enumerate(selected_chords):
        correct_answer = get_chord_name(root, chord_type)

        wrong_options = set()
        while len(wrong_options) < 3:
            wrong_root, wrong_type = random.choice(chord_pool)
            wrong_name = get_chord_name(wrong_root, wrong_type)
            if wrong_name != correct_answer:
                wrong_options.add(wrong_name)

        all_options = list(wrong_options) + [correct_answer]
        random.shuffle(all_options)


        audio_data = generate_chord_audio(root, chord_type)

        questions.append({
            'id': i + 1,
            'correct': correct_answer,
            'options': all_options,
            'audio': audio_data,
            'root': root,
            'type': chord_type if chord_type else 'Major',
        })


    reference_audio = generate_reference_c()

    context = {
        'mode': mode,
        'mode_label': 'Normal Mode' if mode == 'normal' else 'Hard Mode',
        'questions_json': json.dumps(questions),
        'reference_audio': reference_audio,
        'total_questions': num_questions,
    }

    return render(request, 'detector/chord_quiz.html', context)