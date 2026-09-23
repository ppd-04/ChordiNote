import os
import json
import librosa

from django.shortcuts import render
from django.conf import settings

from .dsp_engine import process_audio_file, ANALYSIS_PROFILES
from .polyphonic_engine import process_audio_polyphonic, POLYPHONIC_PROFILES, get_profile_display_info
from .audio_renderer import render_reconstructed_audio

from .visualizer import generate_all_visualizations
from .ml_chord_classifier import analyze_chords


def home(request):
    return render(request, 'detector/home.html')

# def upload_audio(request):
#     if request.method=='POST':
#         audio_file= request.FILES.get('audio_file')
#         profile_name = request.POST.get('profile', 'clearn_melody')

#         if not audio_file:
#             return render(request, 'detector/upload.html', {
#                 'upoloaded': False,
#                 'error' : 'Please select correct audio file',
#                 'profiles': ANALYSIS_PROFILES,
#             })

#         upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
#         os.makedirs(upload_dir, exist_ok=True)
#         file_path = os.path.join(upload_dir, audio_file.name)

#         with open(file_path, 'wb+') as destination:
#             for chunk in audio_file.chunks():
#                 destination.write(chunk)

#         try:
#             detected_notes = process_audio_file(file_path, profile_name)
#             if not detected_notes:
#                 return render(request, 'detector/upload.html', {
#                     'uploaded': False,
#                     'error': 'Could not detect any notes. Try a different profile or a clearer audio file.',
#                     'profiles': ANALYSIS_PROFILES,
#                 })
#             reconstructed_filename = f"reconstructed_{audio_file.name.rsplit('.', 1)[0]}.wav"
#             reconstructed_path = os.path.join(upload_dir, reconstructed_filename)

#             render_success = render_reconstructed_audio(detected_notes, reconstructed_path,)
#             total_notes = len([n for n in detected_notes if n['note'] != 'REST'])
#             total_rests = len([n for n in detected_notes if n['note'] == 'REST'])
#             unique_notes = len(set(n['note'] for n in detected_notes if n['note'] != 'REST'))

#             profile_label = ANALYSIS_PROFILES.get(profile_name, {}).get('label', profile_name)

#             try:
#                 # chord_analysis = analyze_chords(file_path)
#                 chord_analysis = analyze_chords(file_path, hop_seconds=3.0, max_duration=60)
#             except Exception as chord_err:
#                 print(f"Chord analysis error: {chord_err}")
#                 chord_analysis = None

#             try:
#                 viz_data = generate_all_visualizations(file_path)
#             except Exception as viz_err:
#                 viz_data = None

#             context = {
#                 'uploaded': True,
#                 'filename': audio_file.name,
#                 'filesize': round(audio_file.size / 1024, 2),
#                 'notes': detected_notes,
#                 'total_notes': total_notes,
#                 'total_rests': total_rests,
#                 'unique_notes': unique_notes,
#                 'profile_used': profile_label,
#                 'has_reconstruction': render_success,
#                 'reconstructed_url': f"/media/uploads/{reconstructed_filename}" if render_success else None,
#                 'original_url': f"/media/uploads/{audio_file.name}",
#                 'notes_json': json.dumps(detected_notes),
#                 'viz_data' : json.dumps(viz_data) if viz_data else None,
#                 'chord_analysis' : chord_analysis                
#             }       
#             return render(request, 'detector/results.html', context)  
#         except Exception as e:
#             return render(request, 'detector/upload.html', {
#                 'uploaded': False,
#                 'error': f'Processing error: {str(e)}',
#                 'profiles': ANALYSIS_PROFILES,
#             })   
            

        
#         # context lage html e variable hishebe use korar jonno
        
#     return render(request, 'detector/upload.html', {
#         'uploaded':False,
#         'profiles': ANALYSIS_PROFILES,
#         })    


def upload_audio(request):
    # Combine profiles for the UI dropdown
    combined_profiles = {}
    
    # Add pYIN profiles
    for key, profile in ANALYSIS_PROFILES.items():
        combined_profiles[f"pyin_{key}"] = {
            "label": f"[pYIN] {profile['label']}",
            "fmin": profile["fmin"],
            "fmax": profile["fmax"],
            "engine": "pyin",
        }
    
    # Add Polyphonic CQT profiles
    for key in POLYPHONIC_PROFILES.keys():
        info = get_profile_display_info(key)
        combined_profiles[f"poly_{key}"] = {
            "label": f"[Polyphonic] {info['label']}",
            "fmin": info["fmin"],
            "fmax": info["fmax"],
            "engine": "polyphonic",
        }

    if request.method == 'POST':
        audio_file = request.FILES.get('audio_file')
        profile_name = request.POST.get('profile', 'pyin_clean_melody')
        
        if not audio_file:
            return render(request, 'detector/upload.html', {
                'uploaded': False,
                'error': 'Please select a correct audio file',
                'profiles': combined_profiles,
            })

        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, audio_file.name)

        with open(file_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        try:
            #: # Route to the correct engine based on profile prefix
            if profile_name.startswith('poly_'):
                actual_profile = profile_name.replace('poly_', '')
                detected_notes = process_audio_polyphonic(file_path, actual_profile)
                profile_label = POLYPHONIC_PROFILES.get(actual_profile, {}).get('label', actual_profile)
                engine_used = "Polyphonic (CQT DSP)"
            else:
                actual_profile = profile_name.replace('pyin_', '')
                detected_notes = process_audio_file(file_path, actual_profile)
                profile_label = ANALYSIS_PROFILES.get(actual_profile, {}).get('label', actual_profile)
                engine_used = "Monophonic (pYIN)"
                
            if not detected_notes:
                return render(request, 'detector/upload.html', {
                    'uploaded': False,
                    'error': 'Could not detect any notes. Try a different profile or a clearer audio file.',
                    'profiles': combined_profiles,
                })

            reconstructed_filename = f"reconstructed_{audio_file.name.rsplit('.', 1)[0]}.wav"
            reconstructed_path = os.path.join(upload_dir, reconstructed_filename)

            # Flatten chords for the basic audio renderer
            reconstruction_notes = []
            for note_event in detected_notes:
                if note_event.get('is_chord', False):
                    for midi_note in note_event.get('midi_notes', []):
                        note_name = librosa.midi_to_note(midi_note)
                        freq = librosa.midi_to_hz(midi_note)
                        reconstruction_notes.append({
                            "timestamp": note_event["timestamp"],
                            "note": note_name,
                            "frequency": f"{freq:.1f} Hz",
                        })
                else:
                    reconstruction_notes.append(note_event)

            render_success = render_reconstructed_audio(reconstruction_notes, reconstructed_path)
            
            total_notes = len([n for n in detected_notes if n['note'] != 'REST'])
            total_rests = len([n for n in detected_notes if n['note'] == 'REST'])
            
            unique_notes_set = set()
            for n in detected_notes:
                if n['note'] != 'REST':
                    if ' + ' in n['note']:
                        for individual_note in n['note'].split(' + '):
                            unique_notes_set.add(individual_note.strip())
                    else:
                        unique_notes_set.add(n['note'])
            unique_notes = len(unique_notes_set)

            try:
                chord_analysis = analyze_chords(file_path, hop_seconds=3.0, max_duration=60)
            except Exception as chord_err:
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
                'engine_used': engine_used,
                'has_reconstruction': render_success,
                'reconstructed_url': f"/media/uploads/{reconstructed_filename}" if render_success else None,
                'original_url': f"/media/uploads/{audio_file.name}",
                'notes_json': json.dumps(detected_notes),
                'viz_data' : json.dumps(viz_data) if viz_data else None,
                'chord_analysis' : chord_analysis                
            }       
            return render(request, 'detector/results.html', context)  
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return render(request, 'detector/upload.html', {
                'uploaded': False,
                'error': f'Processing error: {str(e)}',
                'profiles': combined_profiles,
            })   
            
    return render(request, 'detector/upload.html', {
        'uploaded':False,
        'profiles': combined_profiles,
    })

def piano(request):
    return render(request, 'detector/piano.html')

def audio_lab(request):
    return render(request, 'detector/audio_lab.html')