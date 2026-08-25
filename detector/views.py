from django.shortcuts import render

def home(request):
    return render(request, 'detector/home.html')

def upload_audio(request):
    if request.method=='POST':
        audio_file= request.FILES.get('audio_file')
        # context lage html e variable hishebe use korar jonno
        if audio_file:
            context = {
                'filename': audio_file.name,
                'filesize': round(audio_file.size/1024, 2),
                'uploaded': True,
            }
            return render(request, 'detector/upload.html', context)
    return render(request, 'detector/upload.html', {'uploaded':False})    
