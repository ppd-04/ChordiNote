import os
import json

from django.shortcuts import render
from django.conf import settings

from .dsp_engine import process_audio_file, ANALYSIS_PROFILES
from .audio_renderer import render_reconstructed_audio

from .visualizer import generate_all_visualizations
from .ml_chord_classifier import analyze_chords


def home(request):
    return render(request, 'detector/home.html')

def upload_audio(request):
    if request.method=='POST':
        audio_file= request.FILES.get('audio_file')
        profile_name = request.POST.get('profile', 'clearn_melody')

        if not audio_file:
            return render(request, 'detector/upload.html', {
                'upoloaded': False,
                'error' : 'Please select correct audio file',
                'profiles': ANALYSIS_PROFILES,
            })

        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, audio_file.name)

        with open(file_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        try:
            detected_notes = process_audio_file(file_path, profile_name)
            if not detected_notes:
                return render(request, 'detector/upload.html', {
                    'uploaded': False,
                    'error': 'Could not detect any notes. Try a different profile or a clearer audio file.',
                    'profiles': ANALYSIS_PROFILES,
                })
            reconstructed_filename = f"reconstructed_{audio_file.name.rsplit('.', 1)[0]}.wav"
            reconstructed_path = os.path.join(upload_dir, reconstructed_filename)

            render_success = render_reconstructed_audio(detected_notes, reconstructed_path,)
            total_notes = len([n for n in detected_notes if n['note'] != 'REST'])
            total_rests = len([n for n in detected_notes if n['note'] == 'REST'])
            unique_notes = len(set(n['note'] for n in detected_notes if n['note'] != 'REST'))

            profile_label = ANALYSIS_PROFILES.get(profile_name, {}).get('label', profile_name)

            try:
                # chord_analysis = analyze_chords(file_path)
                chord_analysis = analyze_chords(file_path, hop_seconds=3.0, max_duration=60)
            except Exception as chord_err:
                print(f"Chord analysis error: {chord_err}")
                chord_analysis = None

            try:
                viz_data = generate_all_visualizations(file_path)
            except Exception as viz_err:
                viz_data = None

            context = {
                'uploaded': True,
                'filename': audio_file.name,
                'filesize': round(audio_file.size / 1024, 2),
                'notes': detected_notes,
                'total_notes': total_notes,
                'total_rests': total_rests,
                'unique_notes': unique_notes,
                'profile_used': profile_label,
                'has_reconstruction': render_success,
                'reconstructed_url': f"/media/uploads/{reconstructed_filename}" if render_success else None,
                'original_url': f"/media/uploads/{audio_file.name}",
                'notes_json': json.dumps(detected_notes),
                'viz_data' : json.dumps(viz_data) if viz_data else None,
                'chord_analysis' : chord_analysis                
            }       
            return render(request, 'detector/results.html', context)  
        except Exception as e:
            return render(request, 'detector/upload.html', {
                'uploaded': False,
                'error': f'Processing error: {str(e)}',
                'profiles': ANALYSIS_PROFILES,
            })   
            

        
        # context lage html e variable hishebe use korar jonno
        
    return render(request, 'detector/upload.html', {
        'uploaded':False,
        'profiles': ANALYSIS_PROFILES,
        })    


def piano(request):
    return render(request, 'detector/piano.html')

def audio_lab(request):
    return render(request, 'detector/audio_lab.html')