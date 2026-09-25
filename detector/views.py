import os
import json
import librosa
import numpy as np

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .dsp_engine import process_audio_file, ANALYSIS_PROFILES
from .polyphonic_engine import process_audio_polyphonic, POLYPHONIC_PROFILES, get_profile_display_info
from .polyphonic_engine_v2 import process_audio_polyphonic_v2, V2_PROFILES, get_v2_profile_display_info
from .polyphonic_engine_v4 import process_audio_polyphonic_v4
from .voice_engine import process_audio_voice
from .audio_renderer import render_reconstructed_audio, calculate_reconstruction_error

from .visualizer import generate_all_visualizations
from .ml_chord_classifier import analyze_chords
from .key_detector import detect_key_from_notes
from .note_chord_detector import detect_chords_from_notes
from .sheet_generator import generate_piano_abc
from .karaoke import remove_vocals_center_cancellation, remove_vocals_nmf
from .autotuner import autotune_audio

def home(request):
    return render(request, 'detector/home.html')


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
        
    # Add Voice Engine profile
    combined_profiles["voice_acapella"] = {
        "label": "[Vocal] Acapella / Monophonic Voice (pYIN + Gate)",
        "fmin": "C2",
        "fmax": "C6",
        "engine": "voice",
    }
    
    # Add Polyphonic CQT v1 profiles
    for key in POLYPHONIC_PROFILES.keys():
        info = get_profile_display_info(key)
        combined_profiles[f"poly_{key}"] = {
            "label": f"[Polyphonic V1] {info['label']}",
            "fmin": info["fmin"],
            "fmax": info["fmax"],
            "engine": "polyphonic",
        }

    # Add V2 Enhanced profiles (Learned Templates + Sparsity + HMM)
    for key in V2_PROFILES.keys():
        info = get_v2_profile_display_info(key)
        combined_profiles[f"v2_{key}"] = {
            "label": f"[V2 Enhanced] {info['label']}",
            "fmin": info["fmin"],
            "fmax": info["fmax"],
            "engine": "polyphonic_v2",
        }

    # Add V4 Hybrid profiles
    # Piano sub-profiles (soft/balanced/fast) are deliberately excluded here —
    # they are presented as a single "Piano V4" card with a segmented switcher in the template.
    PIANO_SUBPROFILES = {"piano_v2_soft", "piano_v2", "piano_v2_fast"}
    for key in V2_PROFILES.keys():
        if key in PIANO_SUBPROFILES:
            continue  # handled by the segmented piano card in the template
        info = get_v2_profile_display_info(key)
        combined_profiles[f"v4_{key}"] = {
            "label": f"[V4 Hybrid] {info['label']} (Pitch-Aware Filter)",
            "fmin": info["fmin"],
            "fmax": info["fmax"],
            "engine": "polyphonic_v4",
        }

    # Single V4 Piano entry (the tempo switcher in the template picks the sub-profile)
    combined_profiles["v4_piano_v2"] = {
        "label": "[V4 Hybrid] Piano (Pitch-Aware Filter)",
        "fmin": "A1",
        "fmax": "C8",
        "engine": "polyphonic_v4",
        "is_piano_v4": True,   # flag read by template to render segmented switcher
    }

    # Separate profiles into Monophonic and Polyphonic groups
    monophonic_profiles = {}
    polyphonic_profiles = {}
    for key, profile in combined_profiles.items():
        if profile["engine"] in ["pyin", "voice"]:
            monophonic_profiles[key] = profile
        else:
            polyphonic_profiles[key] = profile

    if request.method == 'POST':
        audio_file = request.FILES.get('audio_file')
        profile_name = request.POST.get('profile', 'pyin_clean_melody')
        
        if not audio_file:
            return render(request, 'detector/upload.html', {
                'uploaded': False,
                'error': 'Please select a correct audio file',
                'profiles': combined_profiles,
                'monophonic_profiles': monophonic_profiles,
                'polyphonic_profiles': polyphonic_profiles,
            })

        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, audio_file.name)

        with open(file_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)


        try:
            # ============================================
            # KARAOKE: Optional Vocal Removal (Preprocessing)
            # ============================================
            # If the user checked "Remove vocals before detection",
            # we generate an instrumental.wav and a vocal.wav using either
            # center cancellation or NMF.
            # All subsequent detection steps use the instrumental file.
            remove_vocals = request.POST.get('remove_vocals') == 'on'
            karaoke_method = request.POST.get('karaoke_method', 'center')
            instrumental_url = None
            vocal_url = None
            karaoke_success = False

            if remove_vocals:
                instrumental_filename = f"instrumental_{audio_file.name.rsplit('.', 1)[0]}.wav"
                vocal_filename = f"vocal_{audio_file.name.rsplit('.', 1)[0]}.wav"
                
                instrumental_path = os.path.join(upload_dir, instrumental_filename)
                vocal_path = os.path.join(upload_dir, vocal_filename)

                if karaoke_method == 'nmf':
                    karaoke_success = remove_vocals_nmf(file_path, instrumental_path, vocal_path)
                else:
                    karaoke_success = remove_vocals_center_cancellation(file_path, instrumental_path, vocal_path)

                if karaoke_success:
                    instrumental_url = f"/media/uploads/{instrumental_filename}"
                    vocal_url = f"/media/uploads/{vocal_filename}"
                    # Route detection to the instrumental version instead of the original
                    detection_file = instrumental_path
                else:
                    # If vocal removal failed for any reason, fall back to original file
                    detection_file = file_path
            else:
                detection_file = file_path

            # Route to the correct engine based on profile prefix
            if profile_name.startswith('v4_'):
                actual_profile = profile_name.replace('v4_', '')
                detected_notes = process_audio_polyphonic_v4(detection_file, actual_profile)
                profile_label = V2_PROFILES.get(actual_profile, {}).get('label', actual_profile)
                engine_used = "Polyphonic V4 (Pitch-Aware Hybrid Filter)"
            elif profile_name.startswith('v2_'):
                actual_profile = profile_name.replace('v2_', '')
                detected_notes = process_audio_polyphonic_v2(detection_file, actual_profile)
                profile_label = V2_PROFILES.get(actual_profile, {}).get('label', actual_profile)
                engine_used = "Polyphonic V2 (CQT + Learned Templates + Sparse NMF + HMM)"
            elif profile_name.startswith('poly_'):
                actual_profile = profile_name.replace('poly_', '')
                detected_notes = process_audio_polyphonic(detection_file, actual_profile)
                profile_label = POLYPHONIC_PROFILES.get(actual_profile, {}).get('label', actual_profile)
                engine_used = "Polyphonic V1 (CQT DSP)"
            elif profile_name.startswith('voice_'):
                detected_notes = process_audio_voice(detection_file)
                profile_label = "Acapella / Monophonic Voice (pYIN + Gate)"
                engine_used = "Monophonic Voice Engine (pYIN + Noise Gate)"
            else:
                actual_profile = profile_name.replace('pyin_', '')
                detected_notes = process_audio_file(detection_file, actual_profile)
                profile_label = ANALYSIS_PROFILES.get(actual_profile, {}).get('label', actual_profile)
                engine_used = "Monophonic (pYIN)"
                
            if not detected_notes:
                return render(request, 'detector/upload.html', {
                    'uploaded': False,
                    'error': 'Could not detect any notes. Try a different profile or a clearer audio file.',
                    'profiles': combined_profiles,
                    'monophonic_profiles': monophonic_profiles,
                    'polyphonic_profiles': polyphonic_profiles,
                })

            # ============================================
            # RECONSTRUCTION 1: FULL (all notes)
            # ============================================
            reconstructed_filename = f"reconstructed_{audio_file.name.rsplit('.', 1)[0]}.wav"
            reconstructed_path = os.path.join(upload_dir, reconstructed_filename)

            render_success = render_reconstructed_audio(detected_notes, reconstructed_path)

            # ============================================
            # RECONSTRUCTION 2: CORE MELODY (single note)
            # ============================================
            # Pick only the highest note from each chord.
            # In piano music, the melody is almost always the highest note.
            # This gives a clean, single-note melody line.
            melody_filename = f"melody_{audio_file.name.rsplit('.', 1)[0]}.wav"
            melody_path = os.path.join(upload_dir, melody_filename)

            melody_notes = []
            for note_event in detected_notes:
                if note_event.get('is_chord', False):
                    # Pick the highest note (melody is on top) and its associated velocity
                    midi_list = note_event.get('midi_notes', [])
                    vel_list = note_event.get('velocities', [0.7] * len(midi_list))
                    if midi_list:
                        highest_idx = int(np.argmax(midi_list))
                        highest_midi = midi_list[highest_idx]
                        highest_vel = vel_list[highest_idx] if highest_idx < len(vel_list) else 0.7
                        note_name = librosa.midi_to_note(highest_midi)
                        freq = librosa.midi_to_hz(highest_midi)
                        melody_notes.append({
                            "timestamp": note_event["timestamp"],
                            "note": note_name,
                            "frequency": f"{freq:.1f} Hz",
                            "midi_notes": [highest_midi],
                            "velocities": [highest_vel],
                        })
                else:
                    # Single note — keep as-is
                    melody_notes.append(note_event)

            melody_success = render_reconstructed_audio(melody_notes, melody_path)
            
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
                musical_key = detect_key_from_notes(detected_notes)
            except Exception as e:
                print(f"Key detection error: {e}")
                musical_key = "Unknown"

            try:
                chord_analysis = analyze_chords(detection_file, hop_seconds=3.0, max_duration=60)
            except Exception as chord_err:
                chord_analysis = None

            try:
                viz_data = generate_all_visualizations(detection_file)
            except Exception as viz_err:
                viz_data = None

            # Symbolic Grid Hybrid Chord Detection (uses V4 note events mapped to time grid)
            note_chords = []
            try:
                note_chords = detect_chords_from_notes(detected_notes, musical_key)
            except Exception as ncd_err:
                print(f"Note chord detection error: {ncd_err}")

            reconstruction_metrics = None
            if render_success:
                try:
                    reconstruction_metrics = calculate_reconstruction_error(detection_file, reconstructed_path)
                except Exception as err:
                    print(f"Metrics computation error: {err}")

            try:
                sheet_data = generate_piano_abc(detected_notes, file_path=detection_file, musical_key=musical_key)
            except Exception as sheet_err:
                print(f"Sheet music generation error: {sheet_err}")
                sheet_data = None

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
                'reconstruction_metrics': reconstruction_metrics,
                'has_melody': melody_success,
                'melody_url': f"/media/uploads/{melody_filename}" if melody_success else None,
                'original_url': f"/media/uploads/{audio_file.name}",
                'has_karaoke': karaoke_success,
                'instrumental_url': instrumental_url,
                'vocal_url': vocal_url,
                'karaoke_method_used': karaoke_method if karaoke_success else None,
                'notes_json': json.dumps(detected_notes),
                'chords_json': json.dumps(note_chords),
                'viz_data' : json.dumps(viz_data) if viz_data else None,
                'chord_analysis' : chord_analysis,
                'musical_key': musical_key,
                'sheet_data': sheet_data,
                'sheet_abc': sheet_data['abc_string'] if sheet_data else None
            }       
            return render(request, 'detector/results.html', context)  
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return render(request, 'detector/upload.html', {
                'uploaded': False,
                'error': f'Processing error: {str(e)}',
                'profiles': combined_profiles,
                'monophonic_profiles': monophonic_profiles,
                'polyphonic_profiles': polyphonic_profiles,
            })   
            
    return render(request, 'detector/upload.html', {
        'uploaded':False,
        'profiles': combined_profiles,
        'monophonic_profiles': monophonic_profiles,
        'polyphonic_profiles': polyphonic_profiles,
    })

    
def piano(request):
    return render(request, 'detector/piano.html')

def audio_lab(request):
    return render(request, 'detector/audio_lab.html')

@csrf_exempt
def api_autotune(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        pull = float(request.POST.get('pull', 1.0))
        
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save original file
        file_path = os.path.join(upload_dir, audio_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)
                
        # Generate output path
        timestamp = os.path.basename(file_path).rsplit('.', 1)[0]
        output_filename = f"autotuned_{timestamp}.wav"
        output_path = os.path.join(upload_dir, output_filename)
        
        # Run autotuner
        success = autotune_audio(file_path, output_path, scale_notes=None, pull=pull)
        
        if success:
            return JsonResponse({
                'success': True,
                'autotuned_url': f'/media/uploads/{output_filename}'
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Failed to process autotune.'
            }, status=500)
            
    return JsonResponse({'error': 'Invalid request'}, status=400)

def tuner(request):
    return render(request, 'detector/tuner.html')

def metronome(request):
    return render(request, 'detector/metronome.html')

def guitar(request):
    return render(request, 'detector/guitar.html')

# import os
# import json
# import librosa
# import numpy as np

# from django.shortcuts import render
# from django.conf import settings

# from .dsp_engine import process_audio_file, ANALYSIS_PROFILES
# from .polyphonic_engine import process_audio_polyphonic, POLYPHONIC_PROFILES, get_profile_display_info
# from .polyphonic_engine_v2 import process_audio_polyphonic_v2, V2_PROFILES, get_v2_profile_display_info
# from .polyphonic_engine_v4 import process_audio_polyphonic_v4
# from .audio_renderer import render_reconstructed_audio, calculate_reconstruction_error

# from .visualizer import generate_all_visualizations
# from .ml_chord_classifier import analyze_chords
# from .key_detector import detect_key_from_notes
# from .sheet_generator import generate_piano_abc
# from .karaoke import remove_vocals_center_cancellation, remove_vocals_nmf

# def home(request):
#     return render(request, 'detector/home.html')

# # def upload_audio(request):
# #     if request.method=='POST':
# #         audio_file= request.FILES.get('audio_file')
# #         profile_name = request.POST.get('profile', 'clearn_melody')

# #         if not audio_file:
# #             return render(request, 'detector/upload.html', {
# #                 'upoloaded': False,
# #                 'error' : 'Please select correct audio file',
# #                 'profiles': ANALYSIS_PROFILES,
# #             })

# #         upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
# #         os.makedirs(upload_dir, exist_ok=True)
# #         file_path = os.path.join(upload_dir, audio_file.name)

# #         with open(file_path, 'wb+') as destination:
# #             for chunk in audio_file.chunks():
# #                 destination.write(chunk)

# #         try:
# #             detected_notes = process_audio_file(file_path, profile_name)
# #             if not detected_notes:
# #                 return render(request, 'detector/upload.html', {
# #                     'uploaded': False,
# #                     'error': 'Could not detect any notes. Try a different profile or a clearer audio file.',
# #                     'profiles': ANALYSIS_PROFILES,
# #                 })
# #             reconstructed_filename = f"reconstructed_{audio_file.name.rsplit('.', 1)[0]}.wav"
# #             reconstructed_path = os.path.join(upload_dir, reconstructed_filename)

# #             render_success = render_reconstructed_audio(detected_notes, reconstructed_path,)
# #             total_notes = len([n for n in detected_notes if n['note'] != 'REST'])
# #             total_rests = len([n for n in detected_notes if n['note'] == 'REST'])
# #             unique_notes = len(set(n['note'] for n in detected_notes if n['note'] != 'REST'))

# #             profile_label = ANALYSIS_PROFILES.get(profile_name, {}).get('label', profile_name)

# #             try:
# #                 # chord_analysis = analyze_chords(file_path)
# #                 chord_analysis = analyze_chords(file_path, hop_seconds=3.0, max_duration=60)
# #             except Exception as chord_err:
# #                 print(f"Chord analysis error: {chord_err}")
# #                 chord_analysis = None

# #             try:
# #                 viz_data = generate_all_visualizations(file_path)
# #             except Exception as viz_err:
# #                 viz_data = None

# #             context = {
# #                 'uploaded': True,
# #                 'filename': audio_file.name,
# #                 'filesize': round(audio_file.size / 1024, 2),
# #                 'notes': detected_notes,
# #                 'total_notes': total_notes,
# #                 'total_rests': total_rests,
# #                 'unique_notes': unique_notes,
# #                 'profile_used': profile_label,
# #                 'has_reconstruction': render_success,
# #                 'reconstructed_url': f"/media/uploads/{reconstructed_filename}" if render_success else None,
# #                 'original_url': f"/media/uploads/{audio_file.name}",
# #                 'notes_json': json.dumps(detected_notes),
# #                 'viz_data' : json.dumps(viz_data) if viz_data else None,
# #                 'chord_analysis' : chord_analysis                
# #             }       
# #             return render(request, 'detector/results.html', context)  
# #         except Exception as e:
# #             return render(request, 'detector/upload.html', {
# #                 'uploaded': False,
# #                 'error': f'Processing error: {str(e)}',
# #                 'profiles': ANALYSIS_PROFILES,
# #             })   
            

        
# #         # context lage html e variable hishebe use korar jonno
        
# #     return render(request, 'detector/upload.html', {
# #         'uploaded':False,
# #         'profiles': ANALYSIS_PROFILES,
# #         })    


# def upload_audio(request):
#     # Combine profiles for the UI dropdown
#     combined_profiles = {}
    
#     # Add pYIN profiles
#     for key, profile in ANALYSIS_PROFILES.items():
#         combined_profiles[f"pyin_{key}"] = {
#             "label": f"[pYIN] {profile['label']}",
#             "fmin": profile["fmin"],
#             "fmax": profile["fmax"],
#             "engine": "pyin",
#         }
    
#     # Add Polyphonic CQT v1 profiles
#     for key in POLYPHONIC_PROFILES.keys():
#         info = get_profile_display_info(key)
#         combined_profiles[f"poly_{key}"] = {
#             "label": f"[Polyphonic V1] {info['label']}",
#             "fmin": info["fmin"],
#             "fmax": info["fmax"],
#             "engine": "polyphonic",
#         }

#     # Add V2 Enhanced profiles (Learned Templates + Sparsity + HMM)
#     for key in V2_PROFILES.keys():
#         info = get_v2_profile_display_info(key)
#         combined_profiles[f"v2_{key}"] = {
#             "label": f"[V2 Enhanced] {info['label']}",
#             "fmin": info["fmin"],
#             "fmax": info["fmax"],
#             "engine": "polyphonic_v2",
#         }

#     # Add V4 Hybrid profiles
#     for key in V2_PROFILES.keys():
#         info = get_v2_profile_display_info(key)
#         combined_profiles[f"v4_{key}"] = {
#             "label": f"[V4 Hybrid] {info['label']} (Pitch-Aware Filter)",
#             "fmin": info["fmin"],
#             "fmax": info["fmax"],
#             "engine": "polyphonic_v4",
#         }

#     if request.method == 'POST':
#         audio_file = request.FILES.get('audio_file')
#         profile_name = request.POST.get('profile', 'pyin_clean_melody')
        
#         if not audio_file:
#             return render(request, 'detector/upload.html', {
#                 'uploaded': False,
#                 'error': 'Please select a correct audio file',
#                 'profiles': combined_profiles,
#             })

#         upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
#         os.makedirs(upload_dir, exist_ok=True)
#         file_path = os.path.join(upload_dir, audio_file.name)

#         with open(file_path, 'wb+') as destination:
#             for chunk in audio_file.chunks():
#                 destination.write(chunk)



#         try:
#             # Route to the correct engine based on profile prefix
#             if profile_name.startswith('v4_'):
#                 actual_profile = profile_name.replace('v4_', '')
#                 detected_notes = process_audio_polyphonic_v4(file_path, actual_profile)
#                 profile_label = V2_PROFILES.get(actual_profile, {}).get('label', actual_profile)
#                 engine_used = "Polyphonic V4 (Pitch-Aware Hybrid Filter)"
#             elif profile_name.startswith('v2_'):
#                 actual_profile = profile_name.replace('v2_', '')
#                 detected_notes = process_audio_polyphonic_v2(file_path, actual_profile)
#                 profile_label = V2_PROFILES.get(actual_profile, {}).get('label', actual_profile)
#                 engine_used = "Polyphonic V2 (CQT + Learned Templates + Sparse NMF + HMM)"
#             elif profile_name.startswith('poly_'):
#                 actual_profile = profile_name.replace('poly_', '')
#                 detected_notes = process_audio_polyphonic(file_path, actual_profile)
#                 profile_label = POLYPHONIC_PROFILES.get(actual_profile, {}).get('label', actual_profile)
#                 engine_used = "Polyphonic V1 (CQT DSP)"
#             else:
#                 actual_profile = profile_name.replace('pyin_', '')
#                 detected_notes = process_audio_file(file_path, actual_profile)
#                 profile_label = ANALYSIS_PROFILES.get(actual_profile, {}).get('label', actual_profile)
#                 engine_used = "Monophonic (pYIN)"
                
#             if not detected_notes:
#                 return render(request, 'detector/upload.html', {
#                     'uploaded': False,
#                     'error': 'Could not detect any notes. Try a different profile or a clearer audio file.',
#                     'profiles': combined_profiles,
#                 })

#                         # ============================================
#             # RECONSTRUCTION 1: FULL (all notes)
#             # ============================================
#             reconstructed_filename = f"reconstructed_{audio_file.name.rsplit('.', 1)[0]}.wav"
#             reconstructed_path = os.path.join(upload_dir, reconstructed_filename)

#             render_success = render_reconstructed_audio(detected_notes, reconstructed_path)

#             # ============================================
#             # RECONSTRUCTION 2: CORE MELODY (single note)
#             # ============================================
#             # Pick only the highest note from each chord.
#             # In piano music, the melody is almost always the highest note.
#             # This gives a clean, single-note melody line.
#             melody_filename = f"melody_{audio_file.name.rsplit('.', 1)[0]}.wav"
#             melody_path = os.path.join(upload_dir, melody_filename)

#             melody_notes = []
#             for note_event in detected_notes:
#                 if note_event.get('is_chord', False):
#                     # Pick the highest note (melody is on top) and its associated velocity
#                     midi_list = note_event.get('midi_notes', [])
#                     vel_list = note_event.get('velocities', [0.7] * len(midi_list))
#                     if midi_list:
#                         highest_idx = int(np.argmax(midi_list))
#                         highest_midi = midi_list[highest_idx]
#                         highest_vel = vel_list[highest_idx] if highest_idx < len(vel_list) else 0.7
#                         note_name = librosa.midi_to_note(highest_midi)
#                         freq = librosa.midi_to_hz(highest_midi)
#                         melody_notes.append({
#                             "timestamp": note_event["timestamp"],
#                             "note": note_name,
#                             "frequency": f"{freq:.1f} Hz",
#                             "midi_notes": [highest_midi],
#                             "velocities": [highest_vel],
#                         })
#                 else:
#                     # Single note — keep as-is
#                     melody_notes.append(note_event)

#             melody_success = render_reconstructed_audio(melody_notes, melody_path)
            
#             total_notes = len([n for n in detected_notes if n['note'] != 'REST'])
#             total_rests = len([n for n in detected_notes if n['note'] == 'REST'])
            
#             unique_notes_set = set()
#             for n in detected_notes:
#                 if n['note'] != 'REST':
#                     if ' + ' in n['note']:
#                         for individual_note in n['note'].split(' + '):
#                             unique_notes_set.add(individual_note.strip())
#                     else:
#                         unique_notes_set.add(n['note'])
#             unique_notes = len(unique_notes_set)
            
#             try:
#                 musical_key = detect_key_from_notes(detected_notes)
#             except Exception as e:
#                 print(f"Key detection error: {e}")
#                 musical_key = "Unknown"

#             try:
#                 chord_analysis = analyze_chords(file_path, hop_seconds=3.0, max_duration=60)
#             except Exception as chord_err:
#                 chord_analysis = None

#             try:
#                 viz_data = generate_all_visualizations(file_path)
#             except Exception as viz_err:
#                 viz_data = None

#             reconstruction_metrics = None
#             if render_success:
#                 try:
#                     reconstruction_metrics = calculate_reconstruction_error(file_path, reconstructed_path)
#                 except Exception as err:
#                     print(f"Metrics computation error: {err}")

#             try:
#                 sheet_data = generate_piano_abc(detected_notes, file_path=file_path, musical_key=musical_key)
#             except Exception as sheet_err:
#                 print(f"Sheet music generation error: {sheet_err}")
#                 sheet_data = None

#             context = {
#                 'uploaded': True,
#                 'filename': audio_file.name,
#                 'filesize': round(audio_file.size / 1024, 2),
#                 'notes': detected_notes,
#                 'total_notes': total_notes,
#                 'total_rests': total_rests,
#                 'unique_notes': unique_notes,
#                 'profile_used': profile_label,
#                 'engine_used': engine_used,
#                 'has_reconstruction': render_success,
#                 'reconstructed_url': f"/media/uploads/{reconstructed_filename}" if render_success else None,
#                 'reconstruction_metrics': reconstruction_metrics,
#                 'has_melody': melody_success,
#                 'melody_url': f"/media/uploads/{melody_filename}" if melody_success else None,
#                 'original_url': f"/media/uploads/{audio_file.name}",
#                 'notes_json': json.dumps(detected_notes),
#                 'viz_data' : json.dumps(viz_data) if viz_data else None,
#                 'chord_analysis' : chord_analysis,
#                 'musical_key': musical_key,
#                 'sheet_data': sheet_data,
#                 'sheet_abc': sheet_data['abc_string'] if sheet_data else None
#             }       
#             return render(request, 'detector/results.html', context)  
        
#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             return render(request, 'detector/upload.html', {
#                 'uploaded': False,
#                 'error': f'Processing error: {str(e)}',
#                 'profiles': combined_profiles,
#             })   
            
#     return render(request, 'detector/upload.html', {
#         'uploaded':False,
#         'profiles': combined_profiles,
#     })

# def piano(request):
#     return render(request, 'detector/piano.html')

# def audio_lab(request):
#     return render(request, 'detector/audio_lab.html')

# def tuner(request):
#     return render(request, 'detector/tuner.html')

# def metronome(request):
#     return render(request, 'detector/metronome.html')