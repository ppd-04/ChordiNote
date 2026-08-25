from django.urls import path

from . import views
# . mane current package

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_audio, name='upload'),
    # '' mane root, keu root url call korle home function call hobe
    # upload url call korle upload_audio function call hobe
]