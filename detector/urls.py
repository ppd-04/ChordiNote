from django.urls import path

from . import views
from . import game_views
# . mane current package

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_audio, name='upload'),

    path('games/', game_views.game_lobby, name='game_lobby'),
    path('games/chord-quiz/', game_views.chord_quiz, name='chord_quiz'),
    path('games/sing-the-note', game_views.sing_the_note, name='sing_the_note'),
    path('piano/', views.piano, name='piano'),
    path('audio-lab/', views.audio_lab, name='audio_lab'),
    path('api/autotune/', views.api_autotune, name='api_autotune'),
    path('tuner/', views.tuner, name='tuner'),
    path('metronome/', views.metronome, name='metronome'),
    path('guitar/', views.guitar, name='guitar'),
    path('stream-audio/<str:filename>/', views.serve_audio_ranged, name='stream_audio'),
]
    # '' mane root, keu root url call korle home function call hobe
    # upload url call korle upload_audio function call hobe