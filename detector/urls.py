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
]
    # '' mane root, keu root url call korle home function call hobe
    # upload url call korle upload_audio function call hobe