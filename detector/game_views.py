"""
Game Views — handles the chord quiz game logic.
Kept in a separate file to avoid touching the existing views.py
"""

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
    """
    The game selection page.
    Shows available games with descriptions.
    """
    return render(request, 'detector/game_lobby.html')


def chord_quiz(request):
    """
    The chord detection quiz game.
    Generates 10 questions with 4 options each.

    How it works:
    1. Pick a mode (normal/hard) from the URL parameter
    2. Select 10 random chords from the pool
    3. For each chord, generate 3 wrong options + 1 correct
    4. Generate audio for each correct chord
    5. Send everything to the template as JSON
    6. JavaScript handles the game flow (no page reloads during game)
    """

    mode = request.GET.get('mode', 'normal')
    # request.GET.get('mode') reads from URL: /game/chord-quiz/?mode=hard

    # Choose chord pool based on mode
    if mode == 'hard':
        chord_pool = HARD_CHORDS
    else:
        chord_pool = NORMAL_CHORDS
        mode = 'normal'

    # Select 10 random chords for this game
    num_questions = 10
    selected_chords = random.sample(
        chord_pool,
        min(num_questions, len(chord_pool))
    )
    # random.sample picks N unique items from the list (no repeats)

    # Build questions
    questions = []
    for i, (root, chord_type) in enumerate(selected_chords):
        correct_answer = get_chord_name(root, chord_type)

        # Generate 3 wrong options
        wrong_options = set()
        while len(wrong_options) < 3:
            wrong_root, wrong_type = random.choice(chord_pool)
            wrong_name = get_chord_name(wrong_root, wrong_type)
            if wrong_name != correct_answer:
                wrong_options.add(wrong_name)

        # Combine correct + wrong and shuffle
        all_options = list(wrong_options) + [correct_answer]
        random.shuffle(all_options)

        # Generate the chord audio (base64 encoded)
        audio_data = generate_chord_audio(root, chord_type)

        questions.append({
            'id': i + 1,
            'correct': correct_answer,
            'options': all_options,
            'audio': audio_data,
            'root': root,
            'type': chord_type if chord_type else 'Major',
        })

    # Generate reference C chord
    reference_audio = generate_reference_c()

    context = {
        'mode': mode,
        'mode_label': 'Normal Mode' if mode == 'normal' else 'Hard Mode',
        'questions_json': json.dumps(questions),
        # We pass questions as JSON so JavaScript can handle the game
        # without making new requests to the server for each question
        'reference_audio': reference_audio,
        'total_questions': num_questions,
    }

    return render(request, 'detector/chord_quiz.html', context)