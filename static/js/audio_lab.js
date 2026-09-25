/**
 * ChordSense Audio Lab Engine
 * Complete real-time browser audio processing with Scrubbing support
 * NOW WITH: Bulletproof Live Recording + Download + Fixed Autotuner
 */

let player = null;
let isPlaying = false;
let isLooping = false;
let audioLoaded = false;
let effectsInitialized = false;

// Time Tracking Variables
let playOffset = 0;
let startTime = 0;
let progressInterval = null;
let isDraggingTimeline = false;

// Effect Nodes
let pitchShift = null;
let eq = null;
let lowPassFilter = null;
let highPassFilter = null;
let reverb = null;
let delay = null;
let distortion = null;
let tremolo = null;
let compressor = null;
let masterVolume = null;
let analyser = null;

// Autotune Engine Nodes & Variables
let pitchAnalyser = null;
let isAutotuneEnabled = false;
let autotuneSpeed = 0.5;
let autotuneScale = 'chromatic';
let autotuneKey = 'C';
let currentAutotuneCorrection = 0;

// Recording Variables
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = 0;
let recordingTimerInterval = null;
let recordedBlobUrl = null;
let recordingDestination = null;
let isRecording = false;

const NOTE_NAMES_MAP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const SCALE_INTERVALS = {
    chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    major: [0, 2, 4, 5, 7, 9, 11],
    minor: [0, 2, 3, 5, 7, 8, 10]
};

let vizAnimFrame = null;

// ============================================
// 1. INITIALIZE EFFECTS CHAIN
// ============================================
function initEffects() {
    if (effectsInitialized) return;

    try {
        pitchShift = new Tone.PitchShift({ pitch: 0, windowSize: 0.1 });
        eq = new Tone.EQ3({ low: 0, mid: 0, high: 0 });
        lowPassFilter = new Tone.Filter({ frequency: 20000, type: "lowpass" });
        highPassFilter = new Tone.Filter({ frequency: 20, type: "highpass" });
        reverb = new Tone.Freeverb({ roomSize: 0.7, dampening: 3000, wet: 0 });
        delay = new Tone.FeedbackDelay({ delayTime: 0.3, feedback: 0.3, wet: 0 });
        distortion = new Tone.Distortion({ distortion: 0, wet: 0 });
        tremolo = new Tone.Tremolo({ frequency: 4, depth: 0, wet: 0 }).start();
        compressor = new Tone.Compressor({ threshold: -24, ratio: 4 });
        masterVolume = new Tone.Volume(0);
        
        analyser = new Tone.Analyser("waveform", 1024);
        pitchAnalyser = new Tone.Analyser("waveform", 2048);

        pitchShift.chain(
            eq,
            lowPassFilter,
            highPassFilter,
            reverb,
            delay,
            distortion,
            tremolo,
            compressor,
            masterVolume,
            analyser,
            Tone.Destination
        );

        // CREATE MEDIA STREAM DESTINATION FOR RECORDING
        setupRecordingDestination();

        effectsInitialized = true;
        console.log("🎛️ Audio Lab effects chain connected successfully.");
    } catch (err) {
        console.error("Failed to init effects chain:", err);
    }
}

function setupRecordingDestination() {
    try {
        const rawCtx = Tone.context.rawContext || Tone.context._context || Tone.context;
        if (rawCtx && typeof rawCtx.createMediaStreamDestination === 'function') {
            recordingDestination = rawCtx.createMediaStreamDestination();
        }

        if (recordingDestination && masterVolume) {
            // Connect native node via output
            if (masterVolume.output && typeof masterVolume.output.connect === 'function') {
                masterVolume.output.connect(recordingDestination);
            } else if (typeof masterVolume.connect === 'function') {
                masterVolume.connect(recordingDestination);
            }
        }
    } catch (e) {
        console.warn("MediaStream destination setup warning:", e);
    }
}

// ============================================
// 2. FILE UPLOAD & LOADING
// ============================================
async function handleFileUpload(file) {
    if (!file) return;

    const statusEl = document.getElementById('audioStatus');
    const fileInfo = document.getElementById('labFileInfo');
    
    if (statusEl) statusEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading & decoding...';
    if (fileInfo) {
        fileInfo.textContent = `File: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        fileInfo.style.display = 'block';
    }

    if (Tone.context.state !== 'running') {
        await Tone.start();
    }

    initEffects();
    stopAudio();

    if (player) {
        player.disconnect();
        player.dispose();
    }

    const fileUrl = URL.createObjectURL(file);

    try {
        player = new Tone.Player({
            url: fileUrl,
            loop: isLooping,
            autostart: false,
            onload: () => {
                try {
                    audioLoaded = true;
                    player.connect(pitchShift);
                    
                    if (pitchAnalyser) {
                        player.connect(pitchAnalyser);
                    }

                    const duration = player.buffer.duration;
                    
                    const timeline = document.getElementById('timelineSlider');
                    if (timeline) {
                        timeline.max = duration;
                        timeline.value = 0;
                    }
                    
                    playOffset = 0;
                    updateTimeLabel(0, duration);
                    enableAllControls();
                    
                    if (statusEl) {
                        statusEl.innerHTML = '<span style="color: #4ade80;"><i class="fas fa-check-circle"></i> Audio loaded successfully</span>';
                    }
                    
                    startVisualization();
                } catch (loadErr) {
                    console.error("Error setting up audio routing in onload:", loadErr);
                    if (statusEl) statusEl.innerHTML = '<span style="color: #ef4444;">✗ Error initializing audio route</span>';
                }
            },
            onerror: (err) => {
                console.error("Tone.Player load error:", err);
                if (statusEl) statusEl.innerHTML = '<span style="color: #ef4444;">✗ Error reading file</span>';
            }
        });

    } catch (e) {
        console.error("Audio Load exception:", e);
        if (statusEl) statusEl.innerHTML = '<span style="color: #ef4444;">✗ Error loading audio context</span>';
    }
}

// ============================================
// 3. PLAYBACK CONTROLS & TIMELINE SCRUBBING
// ============================================
function enableAllControls() {
    const playBtn = document.getElementById('playBtn');
    const stopBtn = document.getElementById('stopBtn');
    const loopBtn = document.getElementById('loopBtn');
    const timeline = document.getElementById('timelineSlider');
    const recordBtn = document.getElementById('recordBtn');
    
    if (playBtn) { playBtn.disabled = false; playBtn.removeAttribute('disabled'); playBtn.style.opacity = '1'; playBtn.style.cursor = 'pointer'; }
    if (stopBtn) { stopBtn.disabled = false; stopBtn.removeAttribute('disabled'); stopBtn.style.opacity = '1'; stopBtn.style.cursor = 'pointer'; }
    if (loopBtn) { loopBtn.disabled = false; loopBtn.removeAttribute('disabled'); loopBtn.style.opacity = '1'; loopBtn.style.cursor = 'pointer'; }
    if (timeline) { timeline.disabled = false; timeline.removeAttribute('disabled'); timeline.style.opacity = '1'; timeline.style.cursor = 'pointer'; }
    if (recordBtn) { recordBtn.disabled = false; recordBtn.removeAttribute('disabled'); recordBtn.style.opacity = '1'; recordBtn.style.cursor = 'pointer'; }

    document.querySelectorAll('.lab-slider').forEach(slider => {
        slider.disabled = false;
        slider.removeAttribute('disabled');
        slider.style.opacity = '1';
        slider.style.cursor = 'pointer';
    });
}

async function togglePlay() {
    if (!player || !audioLoaded) return;

    if (Tone.context.state !== 'running') {
        await Tone.start();
    }

    if (isPlaying) {
        playOffset = getElapsedTime();
        player.stop();
        isPlaying = false;
        updatePlayButton(false);
        clearInterval(progressInterval);
    } else {
        if (playOffset >= player.buffer.duration) {
            playOffset = 0;
        }
        
        startTime = Tone.now();
        player.start(0, playOffset);
        isPlaying = true;
        updatePlayButton(true);
        startProgressTimer();
    }
}

function stopAudio() {
    if (!player) return;
    player.stop();
    isPlaying = false;
    playOffset = 0;
    updatePlayButton(false);
    clearInterval(progressInterval);
    
    const timeline = document.getElementById('timelineSlider');
    if (timeline) timeline.value = 0;
    
    if (audioLoaded) {
        updateTimeLabel(0, player.buffer.duration);
    }
}

function toggleLoop() {
    isLooping = !isLooping;
    if (player) player.loop = isLooping;

    const btn = document.getElementById('loopBtn');
    if (btn) btn.classList.toggle('active', isLooping);
}

function updatePlayButton(playing) {
    const btn = document.getElementById('playBtn');
    if (btn) {
        btn.innerHTML = playing
            ? '<i class="fas fa-pause"></i> Pause'
            : '<i class="fas fa-play"></i> Play';
        btn.classList.toggle('active', playing);
    }
}

function getElapsedTime() {
    if (!isPlaying) return playOffset;
    
    const elapsed = Tone.now() - startTime;
    const rate = player ? player.playbackRate : 1;
    const currentPosition = playOffset + (elapsed * rate);
    
    if (currentPosition >= player.buffer.duration) {
        if (isLooping) {
            startTime = Tone.now();
            playOffset = 0;
            return 0;
        } else {
            setTimeout(() => { 
                stopAudio(); 
                if (isRecording) stopRecording();
            }, 10);
            return player.buffer.duration;
        }
    }
    return currentPosition;
}

function startProgressTimer() {
    clearInterval(progressInterval);
    progressInterval = setInterval(() => {
        if (!player || !audioLoaded || isDraggingTimeline) return;

        const current = getElapsedTime();
        const timeline = document.getElementById('timelineSlider');
        if (timeline) {
            timeline.value = current;
        }
        updateTimeLabel(current, player.buffer.duration);
    }, 100);
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function updateTimeLabel(current, duration) {
    const timeLabel = document.getElementById('timeLabel');
    if (timeLabel) {
        timeLabel.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
    }
}

function onTimelineSeek(val) {
    if (!player || !audioLoaded) return;
    
    playOffset = parseFloat(val);
    updateTimeLabel(playOffset, player.buffer.duration);

    if (isPlaying) {
        player.stop();
        startTime = Tone.now();
        player.start(0, playOffset);
    }
}

// ============================================
// 4. REAL-TIME EFFECT PARAMETERS
// ============================================
function setPitch(val) {
    if (pitchShift && !isAutotuneEnabled) {
        pitchShift.pitch = parseFloat(val);
    }
    updateLabel('pitchValue', `${val > 0 ? '+' : ''}${val} st`);
}

function setSpeed(val) {
    if (player) {
        if (isPlaying) {
            playOffset = getElapsedTime();
            startTime = Tone.now();
        }
        player.playbackRate = parseFloat(val);
    }
    updateLabel('speedValue', `${val}x`);
}

function setVolume(val) {
    if (!masterVolume) return;
    if (val == 0) {
        masterVolume.mute = true;
    } else {
        masterVolume.mute = false;
        masterVolume.volume.value = (val / 100) * 46 - 40;
    }
    updateLabel('volumeValue', `${Math.round(val)}%`);
}

function setBass(val) {
    if (eq) eq.low.value = parseFloat(val);
    updateLabel('bassValue', `${val > 0 ? '+' : ''}${val} dB`);
}

function setMid(val) {
    if (eq) eq.mid.value = parseFloat(val);
    updateLabel('midValue', `${val > 0 ? '+' : ''}${val} dB`);
}

function setTreble(val) {
    if (eq) eq.high.value = parseFloat(val);
    updateLabel('trebleValue', `${val > 0 ? '+' : ''}${val} dB`);
}

function setLowPass(val) {
    if (!lowPassFilter) return;
    lowPassFilter.frequency.value = parseFloat(val);
    updateLabel('lowPassValue', val >= 20000 ? 'OFF' : `${val} Hz`);
}

function setHighPass(val) {
    if (!highPassFilter) return;
    highPassFilter.frequency.value = parseFloat(val);
    updateLabel('highPassValue', val <= 20 ? 'OFF' : `${val} Hz`);
}

function setReverb(val) {
    if (reverb) reverb.wet.value = parseFloat(val);
    updateLabel('reverbValue', `${Math.round(val * 100)}%`);
}

function setReverbSize(val) {
    if (reverb) reverb.roomSize.value = parseFloat(val);
    updateLabel('reverbDecayValue', `${Math.round(val * 100)}%`);
}

function setDelayTime(val) {
    if (delay) delay.delayTime.value = parseFloat(val);
    updateLabel('delayTimeValue', `${val}s`);
}

function setDelayFeedback(val) {
    if (delay) delay.feedback.value = parseFloat(val);
    updateLabel('delayFeedbackValue', `${Math.round(val * 100)}%`);
}

function setDelayWet(val) {
    if (delay) delay.wet.value = parseFloat(val);
    updateLabel('delayWetValue', `${Math.round(val * 100)}%`);
}

function setDistortion(val) {
    if (!distortion) return;
    distortion.distortion = parseFloat(val);
    distortion.wet.value = val > 0 ? 1 : 0;
    updateLabel('distortionValue', `${Math.round(val * 100)}%`);
}

function setTremoloSpeed(val) {
    if (tremolo) tremolo.frequency.value = parseFloat(val);
    updateLabel('tremoloSpeedValue', `${val} Hz`);
}

function setTremoloDepth(val) {
    if (!tremolo) return;
    tremolo.depth.value = parseFloat(val);
    tremolo.wet.value = val > 0 ? 1 : 0;
    updateLabel('tremoloDepthValue', `${Math.round(val * 100)}%`);
}

function setCompressor(val) {
    if (!compressor) return;
    if (val == 0) {
        compressor.ratio.value = 1;
    } else {
        compressor.ratio.value = 1 + (val / 100) * 19;
    }
    updateLabel('compressorValue', val == 0 ? 'OFF' : `${(1 + (val / 100) * 19).toFixed(1)}:1`);
}

function updateLabel(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// ============================================
// 4B. AUTOTUNER DSP LOGIC
// ============================================
function toggleAutotune(enabled) {
    isAutotuneEnabled = enabled;
    const controls = document.getElementById('autotuneControls');
    
    if (controls) {
        controls.style.opacity = enabled ? '1' : '0.3';
        controls.style.pointerEvents = enabled ? 'auto' : 'none';
    }

    if (!enabled && pitchShift) {
        const pitchSlider = document.getElementById('pitchSlider');
        pitchShift.pitch = pitchSlider ? parseFloat(pitchSlider.value) : 0;
        currentAutotuneCorrection = 0;
        
        const noteDisp = document.getElementById('autotuneDetectedNote');
        const targetDisp = document.getElementById('autotuneTargetNote');
        const meter = document.getElementById('autotuneMeter');
        
        if (noteDisp) noteDisp.textContent = '—';
        if (targetDisp) targetDisp.textContent = '—';
        if (meter) meter.style.left = '50%';
    }
}

function setAutotuneKey(val) { autotuneKey = val; }
function setAutotuneScale(val) { autotuneScale = val; }
function setAutotuneSpeed(val) {
    autotuneSpeed = parseFloat(val) / 100;
    updateLabel('autotuneSpeedValue', `${val}%`);
}

function detectVocalPitch(buffer, sampleRate) {
    const SIZE = buffer.length;
    let rms = 0;
    for (let i = 0; i < SIZE; i++) rms += buffer[i] * buffer[i];
    rms = Math.sqrt(rms / SIZE);
    if (rms < 0.015) return -1;

    const correlations = new Float32Array(SIZE);
    for (let lag = 0; lag < SIZE; lag++) {
        let sum = 0;
        for (let i = 0; i < SIZE - lag; i++) sum += buffer[i] * buffer[i + lag];
        correlations[lag] = sum;
    }

    let d = 0;
    while (d < SIZE && correlations[d] > correlations[d + 1]) d++;

    let maxVal = -1;
    let maxPos = -1;
    const minLag = Math.floor(sampleRate / 800);
    const maxLag = Math.floor(sampleRate / 60);

    for (let lag = Math.max(d, minLag); lag < Math.min(SIZE, maxLag); lag++) {
        if (correlations[lag] > maxVal) {
            maxVal = correlations[lag];
            maxPos = lag;
        }
    }

    if (maxVal < correlations[0] * 0.15) return -1;

    let refinedLag = maxPos;
    if (maxPos > 0 && maxPos < SIZE - 1) {
        const prev = correlations[maxPos - 1];
        const curr = correlations[maxPos];
        const next = correlations[maxPos + 1];
        const denominator = 2 * (2 * curr - prev - next);
        if (denominator !== 0) refinedLag = maxPos + (prev - next) / denominator;
    }

    return sampleRate / refinedLag;
}

function midiToNoteName(midiVal) {
    const r = Math.round(midiVal);
    const name = NOTE_NAMES_MAP[((r % 12) + 12) % 12];
    const oct = Math.floor(r / 12) - 1;
    return name + oct;
}

function snapToScale(midiNote, keyName, scaleType) {
    const keyIndex = NOTE_NAMES_MAP.indexOf(keyName);
    if (keyIndex === -1) return Math.round(midiNote);
    
    const intervals = SCALE_INTERVALS[scaleType] || SCALE_INTERVALS.chromatic;
    let closestMidi = Math.round(midiNote);
    let minDistance = Infinity;
    
    for (let m = Math.max(0, closestMidi - 12); m <= Math.min(127, closestMidi + 12); m++) {
        const pitchClass = (m - keyIndex + 12) % 12;
        if (intervals.includes(pitchClass)) {
            const dist = Math.abs(m - midiNote);
            if (dist < minDistance) {
                minDistance = dist;
                closestMidi = m;
            }
        }
    }
    return closestMidi;
}

function processAutotune() {
    if (!isAutotuneEnabled || !pitchAnalyser || !player || !isPlaying) return;

    const buffer = pitchAnalyser.getValue();
    const freq = detectVocalPitch(buffer, Tone.context.sampleRate);
    
    const noteDisplay = document.getElementById('autotuneDetectedNote');
    const targetDisplay = document.getElementById('autotuneTargetNote');
    const meter = document.getElementById('autotuneMeter');

    if (freq > 60 && freq < 800) {
        const currentMidi = 69 + 12 * Math.log2(freq / 440.0);
        const targetMidi = snapToScale(currentMidi, autotuneKey, autotuneScale);
        const targetError = targetMidi - currentMidi;
        
        const speedSlider = document.getElementById('autotuneSpeedSlider');
        const speedVal = speedSlider ? parseInt(speedSlider.value) : 50;
        const lerpFactor = Math.pow(speedVal / 100, 1.5) * 0.8 + 0.02;
        
        currentAutotuneCorrection += (targetError - currentAutotuneCorrection) * lerpFactor;

        if (pitchShift) pitchShift.pitch = currentAutotuneCorrection;
        if (noteDisplay) noteDisplay.textContent = midiToNoteName(currentMidi);
        if (targetDisplay) targetDisplay.textContent = midiToNoteName(targetMidi);
        if (meter) {
            const offsetPercent = Math.max(5, Math.min(95, 50 + (currentAutotuneCorrection * 22.5)));
            meter.style.left = `${offsetPercent}%`;
        }
    } else {
        currentAutotuneCorrection += (0 - currentAutotuneCorrection) * 0.15;
        if (pitchShift) pitchShift.pitch = currentAutotuneCorrection;
        if (noteDisplay) noteDisplay.textContent = '—';
        if (targetDisplay) targetDisplay.textContent = '—';
        if (meter) meter.style.left = '50%';
    }
}

// ============================================
// 4C. LIVE RECORDING & DOWNLOAD SYSTEM (RELOAD PROOF)
// ============================================

window.toggleRecording = async function() {
    if (!audioLoaded) {
        alert('Please load an audio file first!');
        return;
    }

    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
};

async function startRecording() {
    if (Tone.context.state !== 'running') {
        await Tone.start();
    }

    setupRecordingDestination();

    if (!recordingDestination || !recordingDestination.stream) {
        alert('Recording stream is not available in your browser.');
        return;
    }

    try {
        recordedChunks = [];
        
        const mimeTypes = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/aac',
            'audio/ogg;codecs=opus'
        ];
        
        let selectedMimeType = '';
        for (const mimeType of mimeTypes) {
            if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(mimeType)) {
                selectedMimeType = mimeType;
                break;
            }
        }

        const options = selectedMimeType ? { mimeType: selectedMimeType } : {};
        mediaRecorder = new MediaRecorder(recordingDestination.stream, options);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blobType = selectedMimeType || 'audio/webm';
            const blob = new Blob(recordedChunks, { type: blobType });
            
            if (recordedBlobUrl) {
                URL.revokeObjectURL(recordedBlobUrl);
            }
            
            recordedBlobUrl = URL.createObjectURL(blob);
            
            // UNLOCK DOWNLOAD BUTTON COMPLETELY
            const downloadBtn = document.getElementById('downloadBtn');
            if (downloadBtn) {
                downloadBtn.disabled = false;
                downloadBtn.removeAttribute('disabled');
                downloadBtn.style.opacity = '1';
                downloadBtn.style.cursor = 'pointer';
                downloadBtn.style.pointerEvents = 'auto';
                downloadBtn.classList.remove('btn-secondary');
                downloadBtn.classList.add('btn-primary');
            }
            
            console.log('🎙️ Recording finished! Blob size:', (blob.size / 1024).toFixed(2), 'KB');
        };

        mediaRecorder.start(100);
        isRecording = true;
        recordingStartTime = Date.now();

        // UI updates
        const recordBtn = document.getElementById('recordBtn');
        const recordBtnText = document.getElementById('recordBtnText');
        const statusBar = document.getElementById('recordingStatus');
        
        if (recordBtn) recordBtn.classList.add('recording-active');
        if (recordBtnText) recordBtnText.textContent = 'Stop Recording';
        if (statusBar) statusBar.style.display = 'flex';

        updateRecordingTimer();
        recordingTimerInterval = setInterval(updateRecordingTimer, 1000);

        if (!isPlaying) {
            togglePlay();
        }

        console.log('🎙️ Recording in progress...');

    } catch (err) {
        console.error('Recording error:', err);
        alert('Failed to start recording: ' + err.message);
    }
}

function stopRecording() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') return;

    mediaRecorder.stop();
    isRecording = false;

    // UI updates
    const recordBtn = document.getElementById('recordBtn');
    const recordBtnText = document.getElementById('recordBtnText');
    const statusBar = document.getElementById('recordingStatus');
    
    if (recordBtn) recordBtn.classList.remove('recording-active');
    if (recordBtnText) recordBtnText.textContent = 'Record';
    if (statusBar) statusBar.style.display = 'none';

    clearInterval(recordingTimerInterval);
    console.log('🎙️ Recording stopped, generating download...');
}

function updateRecordingTimer() {
    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const seconds = (elapsed % 60).toString().padStart(2, '0');
    
    const timeEl = document.getElementById('recordingTime');
    if (timeEl) timeEl.textContent = `Recording: ${minutes}:${seconds}`;
}

window.downloadRecording = function() {
    if (!recordedBlobUrl) {
        alert('No recording found! Click "Record", play your audio, and click "Stop Recording" before downloading.');
        return;
    }

    let ext = 'webm';
    if (mediaRecorder && mediaRecorder.mimeType) {
        if (mediaRecorder.mimeType.includes('mp4')) ext = 'mp4';
        else if (mediaRecorder.mimeType.includes('aac')) ext = 'aac';
        else if (mediaRecorder.mimeType.includes('ogg')) ext = 'ogg';
    }

    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    const filename = `chordsense_audio_lab_${timestamp}.${ext}`;

    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = recordedBlobUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    
    setTimeout(() => {
        if (a.parentNode) a.parentNode.removeChild(a);
    }, 500);
    
    console.log('💾 Audio downloaded:', filename);
};

// ============================================
// 5. RESET ALL
// ============================================
window.resetAllEffects = function() {
    const autotuneCheck = document.getElementById('autotuneToggle');
    if (autotuneCheck) {
        autotuneCheck.checked = false;
        toggleAutotune(false);
    }

    setPitch(0);
    setSpeed(1);
    setVolume(80);
    setBass(0);
    setMid(0);
    setTreble(0);
    setLowPass(20000);
    setHighPass(20);
    setReverb(0);
    setReverbSize(0.7);
    setDelayTime(0.3);
    setDelayFeedback(0.3);
    setDelayWet(0);
    setDistortion(0);
    setTremoloSpeed(4);
    setTremoloDepth(0);
    setCompressor(0);

    const defaults = {
        'pitchSlider': 0, 'speedSlider': 1, 'volumeSlider': 80,
        'bassSlider': 0, 'midSlider': 0, 'trebleSlider': 0,
        'lowPassSlider': 20000, 'highPassSlider': 20,
        'reverbSlider': 0, 'reverbDecaySlider': 0.7,
        'delayTimeSlider': 0.3, 'delayFeedbackSlider': 0.3, 'delayWetSlider': 0,
        'distortionSlider': 0,
        'tremoloSpeedSlider': 4, 'tremoloDepthSlider': 0,
        'compressorSlider': 0,
        'autotuneSpeedSlider': 50
    };

    for (const [id, val] of Object.entries(defaults)) {
        const slider = document.getElementById(id);
        if (slider) slider.value = val;
    }
    
    const keySel = document.getElementById('autotuneKey');
    const scaleSel = document.getElementById('autotuneScale');
    if (keySel) keySel.value = 'C';
    if (scaleSel) scaleSel.value = 'chromatic';
    autotuneKey = 'C';
    autotuneScale = 'chromatic';
};

// ============================================
// 6. VISUALIZER
// ============================================
function startVisualization() {
    const canvas = document.getElementById('labWaveform');
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = 120 * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = '120px';
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = 120;

    function draw() {
        const waveform = analyser.getValue();

        processAutotune();

        ctx.fillStyle = '#0f0a1a';
        ctx.fillRect(0, 0, width, height);

        ctx.strokeStyle = 'rgba(167, 139, 250, 0.15)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        ctx.beginPath();
        ctx.strokeStyle = '#a78bfa';
        ctx.lineWidth = 2;

        const sliceWidth = width / waveform.length;
        let x = 0;

        for (let i = 0; i < waveform.length; i++) {
            const v = waveform[i];
            const y = (height / 2) + (v * height / 2);

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);

            x += sliceWidth;
        }

        ctx.stroke();
        vizAnimFrame = requestAnimationFrame(draw);
    }

    if (vizAnimFrame) cancelAnimationFrame(vizAnimFrame);
    draw();
}

// ============================================
// 7. INITIALIZE BINDINGS (INSTANT & DOM-READY)
// ============================================
window.togglePlay = togglePlay;
window.stopAudio = stopAudio;
window.toggleLoop = toggleLoop;

function initAudioLabUI() {
    const fileInput = document.getElementById('labFileInput');
    const dropZone = document.getElementById('labDropZone');
    const timeline = document.getElementById('timelineSlider');

    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) handleFileUpload(this.files[0]);
        });
    }

    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });
    }

    if (timeline) {
        timeline.addEventListener('mousedown', () => { isDraggingTimeline = true; });
        timeline.addEventListener('input', function() {
            updateTimeLabel(this.value, player ? player.buffer.duration : 0);
        });
        timeline.addEventListener('change', function() {
            onTimelineSeek(this.value);
            isDraggingTimeline = false;
        });
        timeline.addEventListener('touchstart', () => { isDraggingTimeline = true; });
        timeline.addEventListener('touchend', function() {
            onTimelineSeek(this.value);
            isDraggingTimeline = false;
        });
    }

    const sliderMap = {
        'pitchSlider': setPitch,
        'speedSlider': setSpeed,
        'volumeSlider': setVolume,
        'bassSlider': setBass,
        'midSlider': setMid,
        'trebleSlider': setTreble,
        'lowPassSlider': setLowPass,
        'highPassSlider': setHighPass,
        'reverbSlider': setReverb,
        'reverbDecaySlider': setReverbSize,
        'delayTimeSlider': setDelayTime,
        'delayFeedbackSlider': setDelayFeedback,
        'delayWetSlider': setDelayWet,
        'distortionSlider': setDistortion,
        'tremoloSpeedSlider': setTremoloSpeed,
        'tremoloDepthSlider': setTremoloDepth,
        'compressorSlider': setCompressor,
    };

    for (const [id, fn] of Object.entries(sliderMap)) {
        const slider = document.getElementById(id);
        if (slider) {
            slider.addEventListener('input', function() {
                fn(this.value);
            });
        }
    }

    const autotuneToggle = document.getElementById('autotuneToggle');
    if (autotuneToggle) {
        autotuneToggle.addEventListener('change', function() {
            toggleAutotune(this.checked);
        });
    }

    const autotuneKeySelect = document.getElementById('autotuneKey');
    if (autotuneKeySelect) {
        autotuneKeySelect.addEventListener('change', function() {
            setAutotuneKey(this.value);
        });
    }

    const autotuneScaleSelect = document.getElementById('autotuneScale');
    if (autotuneScaleSelect) {
        autotuneScaleSelect.addEventListener('change', function() {
            setAutotuneScale(this.value);
        });
    }

    const autotuneSpeedSlider = document.getElementById('autotuneSpeedSlider');
    if (autotuneSpeedSlider) {
        autotuneSpeedSlider.addEventListener('input', function() {
            setAutotuneSpeed(this.value);
        });
    }
}

// Run immediately if already loaded, otherwise attach listener
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAudioLabUI);
} else {
    initAudioLabUI();
}


// /**
//  * ChordSense Audio Lab Engine
//  * Complete real-time browser audio processing with Scrubbing support
//  * NOW WITH: Live Recording + Download + Fixed Autotuner
//  */

// let player = null;
// let isPlaying = false;
// let isLooping = false;
// let audioLoaded = false;
// let effectsInitialized = false;

// // Time Tracking Variables
// let playOffset = 0;
// let startTime = 0;
// let progressInterval = null;
// let isDraggingTimeline = false;

// // Effect Nodes
// let pitchShift = null;
// let eq = null;
// let lowPassFilter = null;
// let highPassFilter = null;
// let reverb = null;
// let delay = null;
// let distortion = null;
// let tremolo = null;
// let compressor = null;
// let masterVolume = null;
// let analyser = null;

// // Autotune Engine Nodes & Variables
// let pitchAnalyser = null;
// let isAutotuneEnabled = false;
// let autotuneSpeed = 0.5;
// let autotuneScale = 'chromatic';
// let autotuneKey = 'C';
// let currentAutotuneCorrection = 0;

// // Recording Variables
// let mediaRecorder = null;
// let recordedChunks = [];
// let recordingStartTime = 0;
// let recordingTimerInterval = null;
// let recordedBlobUrl = null;
// let recordingDestination = null;
// let isRecording = false;

// const NOTE_NAMES_MAP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
// const SCALE_INTERVALS = {
//     chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
//     major: [0, 2, 4, 5, 7, 9, 11],
//     minor: [0, 2, 3, 5, 7, 8, 10]
// };

// let vizAnimFrame = null;

// // ============================================
// // 1. INITIALIZE EFFECTS CHAIN
// // ============================================
// function initEffects() {
//     if (effectsInitialized) return;

//     try {
//         pitchShift = new Tone.PitchShift({ pitch: 0, windowSize: 0.1 });
//         eq = new Tone.EQ3({ low: 0, mid: 0, high: 0 });
//         lowPassFilter = new Tone.Filter({ frequency: 20000, type: "lowpass" });
//         highPassFilter = new Tone.Filter({ frequency: 20, type: "highpass" });
//         reverb = new Tone.Freeverb({ roomSize: 0.7, dampening: 3000, wet: 0 });
//         delay = new Tone.FeedbackDelay({ delayTime: 0.3, feedback: 0.3, wet: 0 });
//         distortion = new Tone.Distortion({ distortion: 0, wet: 0 });
//         tremolo = new Tone.Tremolo({ frequency: 4, depth: 0, wet: 0 }).start();
//         compressor = new Tone.Compressor({ threshold: -24, ratio: 4 });
//         masterVolume = new Tone.Volume(0);
        
//         // Positional syntax works on all Tone.js versions
//         analyser = new Tone.Analyser("waveform", 1024);
//         pitchAnalyser = new Tone.Analyser("waveform", 2048);

//         pitchShift.chain(
//             eq,
//             lowPassFilter,
//             highPassFilter,
//             reverb,
//             delay,
//             distortion,
//             tremolo,
//             compressor,
//             masterVolume,
//             analyser,
//             Tone.Destination
//         );

//         // Create a recording destination — captures the entire processed audio
//         // This connects AFTER all effects, so recording includes EVERYTHING
//         recordingDestination = Tone.context.createMediaStreamDestination();
//         masterVolume.connect(recordingDestination);

//         effectsInitialized = true;
//         console.log("🎛️ Audio Lab effects chain connected successfully.");
//     } catch (err) {
//         console.error("Failed to init effects chain:", err);
//     }
// }

// // ============================================
// // 2. FILE UPLOAD & LOADING
// // ============================================
// async function handleFileUpload(file) {
//     if (!file) return;

//     const statusEl = document.getElementById('audioStatus');
//     const fileInfo = document.getElementById('labFileInfo');
    
//     if (statusEl) statusEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading & decoding...';
//     if (fileInfo) {
//         fileInfo.textContent = `File: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
//         fileInfo.style.display = 'block';
//     }

//     if (Tone.context.state !== 'running') {
//         await Tone.start();
//     }

//     initEffects();
//     stopAudio();

//     if (player) {
//         player.disconnect();
//         player.dispose();
//     }

//     const fileUrl = URL.createObjectURL(file);

//     try {
//         player = new Tone.Player({
//             url: fileUrl,
//             loop: isLooping,
//             autostart: false,
//             onload: () => {
//                 try {
//                     audioLoaded = true;
//                     player.connect(pitchShift);
                    
//                     if (pitchAnalyser) {
//                         player.connect(pitchAnalyser);
//                     }

//                     const duration = player.buffer.duration;
                    
//                     const timeline = document.getElementById('timelineSlider');
//                     if (timeline) {
//                         timeline.max = duration;
//                         timeline.value = 0;
//                     }
                    
//                     playOffset = 0;
//                     updateTimeLabel(0, duration);
//                     enableAllControls();
                    
//                     if (statusEl) {
//                         statusEl.innerHTML = '<span style="color: #4ade80;"><i class="fas fa-check-circle"></i> Audio loaded successfully</span>';
//                     }
                    
//                     startVisualization();
//                 } catch (loadErr) {
//                     console.error("Error setting up audio routing in onload:", loadErr);
//                     if (statusEl) statusEl.innerHTML = '<span style="color: #ef4444;">✗ Error initializing audio route</span>';
//                 }
//             },
//             onerror: (err) => {
//                 console.error("Tone.Player load error:", err);
//                 if (statusEl) statusEl.innerHTML = '<span style="color: #ef4444;">✗ Error reading file</span>';
//             }
//         });

//     } catch (e) {
//         console.error("Audio Load exception:", e);
//         if (statusEl) statusEl.innerHTML = '<span style="color: #ef4444;">✗ Error loading audio context</span>';
//     }
// }

// // ============================================
// // 3. PLAYBACK CONTROLS & TIMELINE SCRUBBING
// // ============================================
// function enableAllControls() {
//     const playBtn = document.getElementById('playBtn');
//     const stopBtn = document.getElementById('stopBtn');
//     const loopBtn = document.getElementById('loopBtn');
//     const timeline = document.getElementById('timelineSlider');
//     const recordBtn = document.getElementById('recordBtn');
    
//     if (playBtn) { playBtn.disabled = false; playBtn.style.opacity = '1'; playBtn.style.cursor = 'pointer'; }
//     if (stopBtn) { stopBtn.disabled = false; stopBtn.style.opacity = '1'; stopBtn.style.cursor = 'pointer'; }
//     if (loopBtn) { loopBtn.disabled = false; loopBtn.style.opacity = '1'; loopBtn.style.cursor = 'pointer'; }
//     if (timeline) { timeline.disabled = false; timeline.style.opacity = '1'; timeline.style.cursor = 'pointer'; }
//     if (recordBtn) { recordBtn.disabled = false; recordBtn.style.opacity = '1'; recordBtn.style.cursor = 'pointer'; }

//     document.querySelectorAll('.lab-slider').forEach(slider => {
//         slider.disabled = false;
//         slider.style.opacity = '1';
//         slider.style.cursor = 'pointer';
//     });
// }

// async function togglePlay() {
//     if (!player || !audioLoaded) return;

//     if (Tone.context.state !== 'running') {
//         await Tone.start();
//     }

//     if (isPlaying) {
//         playOffset = getElapsedTime();
//         player.stop();
//         isPlaying = false;
//         updatePlayButton(false);
//         clearInterval(progressInterval);
//     } else {
//         if (playOffset >= player.buffer.duration) {
//             playOffset = 0;
//         }
        
//         startTime = Tone.now();
//         player.start(0, playOffset);
//         isPlaying = true;
//         updatePlayButton(true);
//         startProgressTimer();
//     }
// }

// function stopAudio() {
//     if (!player) return;
//     player.stop();
//     isPlaying = false;
//     playOffset = 0;
//     updatePlayButton(false);
//     clearInterval(progressInterval);
    
//     const timeline = document.getElementById('timelineSlider');
//     if (timeline) timeline.value = 0;
    
//     if (audioLoaded) {
//         updateTimeLabel(0, player.buffer.duration);
//     }
// }

// function toggleLoop() {
//     isLooping = !isLooping;
//     if (player) player.loop = isLooping;

//     const btn = document.getElementById('loopBtn');
//     if (btn) btn.classList.toggle('active', isLooping);
// }

// function updatePlayButton(playing) {
//     const btn = document.getElementById('playBtn');
//     if (btn) {
//         btn.innerHTML = playing
//             ? '<i class="fas fa-pause"></i> Pause'
//             : '<i class="fas fa-play"></i> Play';
//         btn.classList.toggle('active', playing);
//     }
// }

// function getElapsedTime() {
//     if (!isPlaying) return playOffset;
    
//     const elapsed = Tone.now() - startTime;
//     const rate = player ? player.playbackRate : 1;
//     const currentPosition = playOffset + (elapsed * rate);
    
//     if (currentPosition >= player.buffer.duration) {
//         if (isLooping) {
//             startTime = Tone.now();
//             playOffset = 0;
//             return 0;
//         } else {
//             setTimeout(() => { stopAudio(); }, 10);
//             return player.buffer.duration;
//         }
//     }
//     return currentPosition;
// }

// function startProgressTimer() {
//     clearInterval(progressInterval);
//     progressInterval = setInterval(() => {
//         if (!player || !audioLoaded || isDraggingTimeline) return;

//         const current = getElapsedTime();
//         const timeline = document.getElementById('timelineSlider');
//         if (timeline) {
//             timeline.value = current;
//         }
//         updateTimeLabel(current, player.buffer.duration);
//     }, 100);
// }

// function formatTime(seconds) {
//     const m = Math.floor(seconds / 60);
//     const s = Math.floor(seconds % 60);
//     return `${m}:${s.toString().padStart(2, '0')}`;
// }

// function updateTimeLabel(current, duration) {
//     const timeLabel = document.getElementById('timeLabel');
//     if (timeLabel) {
//         timeLabel.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
//     }
// }

// function onTimelineSeek(val) {
//     if (!player || !audioLoaded) return;
    
//     playOffset = parseFloat(val);
//     updateTimeLabel(playOffset, player.buffer.duration);

//     if (isPlaying) {
//         player.stop();
//         startTime = Tone.now();
//         player.start(0, playOffset);
//     }
// }

// // ============================================
// // 4. REAL-TIME EFFECT PARAMETERS
// // ============================================
// function setPitch(val) {
//     if (pitchShift && !isAutotuneEnabled) {
//         pitchShift.pitch = parseFloat(val);
//     }
//     updateLabel('pitchValue', `${val > 0 ? '+' : ''}${val} st`);
// }

// function setSpeed(val) {
//     if (player) {
//         if (isPlaying) {
//             playOffset = getElapsedTime();
//             startTime = Tone.now();
//         }
//         player.playbackRate = parseFloat(val);
//     }
//     updateLabel('speedValue', `${val}x`);
// }

// function setVolume(val) {
//     if (!masterVolume) return;
//     if (val == 0) {
//         masterVolume.mute = true;
//     } else {
//         masterVolume.mute = false;
//         masterVolume.volume.value = (val / 100) * 46 - 40;
//     }
//     updateLabel('volumeValue', `${Math.round(val)}%`);
// }

// function setBass(val) {
//     if (eq) eq.low.value = parseFloat(val);
//     updateLabel('bassValue', `${val > 0 ? '+' : ''}${val} dB`);
// }

// function setMid(val) {
//     if (eq) eq.mid.value = parseFloat(val);
//     updateLabel('midValue', `${val > 0 ? '+' : ''}${val} dB`);
// }

// function setTreble(val) {
//     if (eq) eq.high.value = parseFloat(val);
//     updateLabel('trebleValue', `${val > 0 ? '+' : ''}${val} dB`);
// }

// function setLowPass(val) {
//     if (!lowPassFilter) return;
//     lowPassFilter.frequency.value = parseFloat(val);
//     updateLabel('lowPassValue', val >= 20000 ? 'OFF' : `${val} Hz`);
// }

// function setHighPass(val) {
//     if (!highPassFilter) return;
//     highPassFilter.frequency.value = parseFloat(val);
//     updateLabel('highPassValue', val <= 20 ? 'OFF' : `${val} Hz`);
// }

// function setReverb(val) {
//     if (reverb) reverb.wet.value = parseFloat(val);
//     updateLabel('reverbValue', `${Math.round(val * 100)}%`);
// }

// function setReverbSize(val) {
//     if (reverb) reverb.roomSize.value = parseFloat(val);
//     updateLabel('reverbDecayValue', `${Math.round(val * 100)}%`);
// }

// function setDelayTime(val) {
//     if (delay) delay.delayTime.value = parseFloat(val);
//     updateLabel('delayTimeValue', `${val}s`);
// }

// function setDelayFeedback(val) {
//     if (delay) delay.feedback.value = parseFloat(val);
//     updateLabel('delayFeedbackValue', `${Math.round(val * 100)}%`);
// }

// function setDelayWet(val) {
//     if (delay) delay.wet.value = parseFloat(val);
//     updateLabel('delayWetValue', `${Math.round(val * 100)}%`);
// }

// function setDistortion(val) {
//     if (!distortion) return;
//     distortion.distortion = parseFloat(val);
//     distortion.wet.value = val > 0 ? 1 : 0;
//     updateLabel('distortionValue', `${Math.round(val * 100)}%`);
// }

// function setTremoloSpeed(val) {
//     if (tremolo) tremolo.frequency.value = parseFloat(val);
//     updateLabel('tremoloSpeedValue', `${val} Hz`);
// }

// function setTremoloDepth(val) {
//     if (!tremolo) return;
//     tremolo.depth.value = parseFloat(val);
//     tremolo.wet.value = val > 0 ? 1 : 0;
//     updateLabel('tremoloDepthValue', `${Math.round(val * 100)}%`);
// }

// function setCompressor(val) {
//     if (!compressor) return;
//     if (val == 0) {
//         compressor.ratio.value = 1;
//     } else {
//         compressor.ratio.value = 1 + (val / 100) * 19;
//     }
//     updateLabel('compressorValue', val == 0 ? 'OFF' : `${(1 + (val / 100) * 19).toFixed(1)}:1`);
// }

// function updateLabel(id, text) {
//     const el = document.getElementById(id);
//     if (el) el.textContent = text;
// }

// // ============================================
// // 4B. AUTOTUNER DSP LOGIC (FIXED)
// // ============================================

// function toggleAutotune(enabled) {
//     isAutotuneEnabled = enabled;
//     const controls = document.getElementById('autotuneControls');
    
//     console.log('🎤 Autotune toggled:', enabled);
    
//     if (controls) {
//         controls.style.opacity = enabled ? '1' : '0.3';
//         controls.style.pointerEvents = enabled ? 'auto' : 'none';
//     }

//     if (!enabled && pitchShift) {
//         const pitchSlider = document.getElementById('pitchSlider');
//         pitchShift.pitch = pitchSlider ? parseFloat(pitchSlider.value) : 0;
//         currentAutotuneCorrection = 0;
        
//         const noteDisp = document.getElementById('autotuneDetectedNote');
//         const targetDisp = document.getElementById('autotuneTargetNote');
//         const meter = document.getElementById('autotuneMeter');
        
//         if (noteDisp) noteDisp.textContent = '—';
//         if (targetDisp) targetDisp.textContent = '—';
//         if (meter) meter.style.left = '50%';
//     }
// }

// function setAutotuneKey(val) {
//     autotuneKey = val;
//     console.log('🎼 Key changed to:', val);
// }

// function setAutotuneScale(val) {
//     autotuneScale = val;
//     console.log('🎼 Scale changed to:', val);
// }

// function setAutotuneSpeed(val) {
//     autotuneSpeed = parseFloat(val) / 100;
//     updateLabel('autotuneSpeedValue', `${val}%`);
// }

// function detectVocalPitch(buffer, sampleRate) {
//     const SIZE = buffer.length;
//     let rms = 0;
//     for (let i = 0; i < SIZE; i++) {
//         rms += buffer[i] * buffer[i];
//     }
//     rms = Math.sqrt(rms / SIZE);

//     if (rms < 0.015) return -1;

//     const correlations = new Float32Array(SIZE);
//     for (let lag = 0; lag < SIZE; lag++) {
//         let sum = 0;
//         for (let i = 0; i < SIZE - lag; i++) {
//             sum += buffer[i] * buffer[i + lag];
//         }
//         correlations[lag] = sum;
//     }

//     let d = 0;
//     while (d < SIZE && correlations[d] > correlations[d + 1]) {
//         d++;
//     }

//     let maxVal = -1;
//     let maxPos = -1;

//     const minLag = Math.floor(sampleRate / 800);
//     const maxLag = Math.floor(sampleRate / 60);

//     for (let lag = Math.max(d, minLag); lag < Math.min(SIZE, maxLag); lag++) {
//         if (correlations[lag] > maxVal) {
//             maxVal = correlations[lag];
//             maxPos = lag;
//         }
//     }

//     if (maxVal < correlations[0] * 0.15) return -1;

//     let refinedLag = maxPos;
//     if (maxPos > 0 && maxPos < SIZE - 1) {
//         const prev = correlations[maxPos - 1];
//         const curr = correlations[maxPos];
//         const next = correlations[maxPos + 1];
//         const denominator = 2 * (2 * curr - prev - next);
//         if (denominator !== 0) {
//             refinedLag = maxPos + (prev - next) / denominator;
//         }
//     }

//     return sampleRate / refinedLag;
// }

// function midiToNoteName(midiVal) {
//     const r = Math.round(midiVal);
//     const name = NOTE_NAMES_MAP[((r % 12) + 12) % 12];
//     const oct = Math.floor(r / 12) - 1;
//     return name + oct;
// }

// function snapToScale(midiNote, keyName, scaleType) {
//     const keyIndex = NOTE_NAMES_MAP.indexOf(keyName);
//     if (keyIndex === -1) return Math.round(midiNote);
    
//     const intervals = SCALE_INTERVALS[scaleType] || SCALE_INTERVALS.chromatic;
//     let closestMidi = Math.round(midiNote);
//     let minDistance = Infinity;
    
//     for (let m = Math.max(0, closestMidi - 12); m <= Math.min(127, closestMidi + 12); m++) {
//         const pitchClass = (m - keyIndex + 12) % 12;
//         if (intervals.includes(pitchClass)) {
//             const dist = Math.abs(m - midiNote);
//             if (dist < minDistance) {
//                 minDistance = dist;
//                 closestMidi = m;
//             }
//         }
//     }
//     return closestMidi;
// }

// function processAutotune() {
//     if (!isAutotuneEnabled || !pitchAnalyser || !player || !isPlaying) return;

//     const buffer = pitchAnalyser.getValue();
//     const freq = detectVocalPitch(buffer, Tone.context.sampleRate);
    
//     const noteDisplay = document.getElementById('autotuneDetectedNote');
//     const targetDisplay = document.getElementById('autotuneTargetNote');
//     const meter = document.getElementById('autotuneMeter');

//     if (freq > 60 && freq < 800) {
//         const currentMidi = 69 + 12 * Math.log2(freq / 440.0);
//         const targetMidi = snapToScale(currentMidi, autotuneKey, autotuneScale);
        
//         const targetError = targetMidi - currentMidi;
        
//         const speedSlider = document.getElementById('autotuneSpeedSlider');
//         const speedVal = speedSlider ? parseInt(speedSlider.value) : 50;
//         const lerpFactor = Math.pow(speedVal / 100, 1.5) * 0.8 + 0.02;
        
//         currentAutotuneCorrection += (targetError - currentAutotuneCorrection) * lerpFactor;

//         if (pitchShift) {
//             pitchShift.pitch = currentAutotuneCorrection;
//         }

//         if (noteDisplay) noteDisplay.textContent = midiToNoteName(currentMidi);
//         if (targetDisplay) targetDisplay.textContent = midiToNoteName(targetMidi);
//         if (meter) {
//             const offsetPercent = Math.max(5, Math.min(95, 50 + (currentAutotuneCorrection * 22.5)));
//             meter.style.left = `${offsetPercent}%`;
//         }
//     } else {
//         currentAutotuneCorrection += (0 - currentAutotuneCorrection) * 0.15;
//         if (pitchShift) pitchShift.pitch = currentAutotuneCorrection;
        
//         if (noteDisplay) noteDisplay.textContent = '—';
//         if (targetDisplay) targetDisplay.textContent = '—';
//         if (meter) meter.style.left = '50%';
//     }
// }

// // ============================================
// // 4C. LIVE RECORDING & DOWNLOAD (NEW)
// // ============================================

// /**
//  * Toggle recording on/off.
//  * 
//  * How it works:
//  * 1. We create a MediaStreamDestination from Tone's Web Audio context
//  * 2. This destination receives the FULL processed audio (all effects applied)
//  * 3. MediaRecorder captures this stream as webm/opus
//  * 4. When stopped, we save it as a downloadable blob
//  */
// async function toggleRecording() {
//     if (!audioLoaded) {
//         alert('Please load an audio file first!');
//         return;
//     }

//     if (isRecording) {
//         stopRecording();
//     } else {
//         await startRecording();
//     }
// }

// async function startRecording() {
//     if (!recordingDestination) {
//         alert('Recording system not initialized. Try reloading the page.');
//         return;
//     }

//     try {
//         recordedChunks = [];
        
//         // Try different MIME types for compatibility
//         const mimeTypes = [
//             'audio/webm;codecs=opus',
//             'audio/webm',
//             'audio/mp4',
//             'audio/ogg;codecs=opus',
//         ];
        
//         let selectedMimeType = '';
//         for (const mimeType of mimeTypes) {
//             if (MediaRecorder.isTypeSupported(mimeType)) {
//                 selectedMimeType = mimeType;
//                 break;
//             }
//         }
        
//         if (!selectedMimeType) {
//             alert('Your browser does not support audio recording.');
//             return;
//         }

//         mediaRecorder = new MediaRecorder(recordingDestination.stream, {
//             mimeType: selectedMimeType,
//             audioBitsPerSecond: 128000,
//         });

//         mediaRecorder.ondataavailable = (event) => {
//             if (event.data.size > 0) {
//                 recordedChunks.push(event.data);
//             }
//         };

//         mediaRecorder.onstop = () => {
//             // Combine all chunks into a single blob
//             const blob = new Blob(recordedChunks, { type: selectedMimeType });
            
//             // Clean up previous blob URL if exists
//             if (recordedBlobUrl) {
//                 URL.revokeObjectURL(recordedBlobUrl);
//             }
            
//             recordedBlobUrl = URL.createObjectURL(blob);
            
//             // Enable download button
//             const downloadBtn = document.getElementById('downloadBtn');
//             if (downloadBtn) {
//                 downloadBtn.disabled = false;
//                 downloadBtn.style.opacity = '1';
//                 downloadBtn.style.cursor = 'pointer';
//                 downloadBtn.classList.add('has-recording');
//             }
            
//             console.log('🎙️ Recording saved:', (blob.size / 1024).toFixed(2), 'KB');
//         };

//         // Start recording (capture data every 100ms)
//         mediaRecorder.start(100);
//         isRecording = true;
//         recordingStartTime = Date.now();

//         // Update UI
//         const recordBtn = document.getElementById('recordBtn');
//         const recordBtnText = document.getElementById('recordBtnText');
//         const statusBar = document.getElementById('recordingStatus');
        
//         if (recordBtn) recordBtn.classList.add('recording-active');
//         if (recordBtnText) recordBtnText.textContent = 'Stop Recording';
//         if (statusBar) statusBar.style.display = 'flex';

//         // Update timer display
//         updateRecordingTimer();
//         recordingTimerInterval = setInterval(updateRecordingTimer, 1000);

//         // Auto-start playback if not already playing
//         if (!isPlaying) {
//             togglePlay();
//         }

//         console.log('🎙️ Recording started with codec:', selectedMimeType);

//     } catch (err) {
//         console.error('Recording error:', err);
//         alert('Failed to start recording: ' + err.message);
//     }
// }

// function stopRecording() {
//     if (!mediaRecorder || mediaRecorder.state === 'inactive') return;

//     mediaRecorder.stop();
//     isRecording = false;

//     // Update UI
//     const recordBtn = document.getElementById('recordBtn');
//     const recordBtnText = document.getElementById('recordBtnText');
//     const statusBar = document.getElementById('recordingStatus');
    
//     if (recordBtn) recordBtn.classList.remove('recording-active');
//     if (recordBtnText) recordBtnText.textContent = 'Record';
//     if (statusBar) statusBar.style.display = 'none';

//     // Stop timer
//     clearInterval(recordingTimerInterval);

//     console.log('🎙️ Recording stopped');
// }

// function updateRecordingTimer() {
//     const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
//     const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
//     const seconds = (elapsed % 60).toString().padStart(2, '0');
    
//     const timeEl = document.getElementById('recordingTime');
//     if (timeEl) timeEl.textContent = `Recording: ${minutes}:${seconds}`;
// }

// function downloadRecording() {
//     if (!recordedBlobUrl) {
//         alert('No recording available. Click Record first!');
//         return;
//     }

//     // Create a temporary download link
//     const a = document.createElement('a');
//     a.href = recordedBlobUrl;
    
//     // Generate filename with timestamp
//     const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
//     a.download = `chordsense_lab_${timestamp}.webm`;
    
//     document.body.appendChild(a);
//     a.click();
//     document.body.removeChild(a);
    
//     console.log('💾 Recording downloaded');
// }

// // ============================================
// // 5. RESET ALL
// // ============================================
// function resetAllEffects() {
//     const autotuneCheck = document.getElementById('autotuneToggle');
//     if (autotuneCheck) {
//         autotuneCheck.checked = false;
//         toggleAutotune(false);
//     }

//     setPitch(0);
//     setSpeed(1);
//     setVolume(80);
//     setBass(0);
//     setMid(0);
//     setTreble(0);
//     setLowPass(20000);
//     setHighPass(20);
//     setReverb(0);
//     setReverbSize(0.7);
//     setDelayTime(0.3);
//     setDelayFeedback(0.3);
//     setDelayWet(0);
//     setDistortion(0);
//     setTremoloSpeed(4);
//     setTremoloDepth(0);
//     setCompressor(0);

//     const defaults = {
//         'pitchSlider': 0, 'speedSlider': 1, 'volumeSlider': 80,
//         'bassSlider': 0, 'midSlider': 0, 'trebleSlider': 0,
//         'lowPassSlider': 20000, 'highPassSlider': 20,
//         'reverbSlider': 0, 'reverbDecaySlider': 0.7,
//         'delayTimeSlider': 0.3, 'delayFeedbackSlider': 0.3, 'delayWetSlider': 0,
//         'distortionSlider': 0,
//         'tremoloSpeedSlider': 4, 'tremoloDepthSlider': 0,
//         'compressorSlider': 0,
//         'autotuneSpeedSlider': 50
//     };

//     for (const [id, val] of Object.entries(defaults)) {
//         const slider = document.getElementById(id);
//         if (slider) slider.value = val;
//     }
    
//     const keySel = document.getElementById('autotuneKey');
//     const scaleSel = document.getElementById('autotuneScale');
//     if (keySel) keySel.value = 'C';
//     if (scaleSel) scaleSel.value = 'chromatic';
//     autotuneKey = 'C';
//     autotuneScale = 'chromatic';
// }

// // ============================================
// // 6. VISUALIZER
// // ============================================
// function startVisualization() {
//     const canvas = document.getElementById('labWaveform');
//     if (!canvas || !analyser) return;

//     const ctx = canvas.getContext('2d');
//     const dpr = window.devicePixelRatio || 1;
//     const rect = canvas.parentElement.getBoundingClientRect();
    
//     canvas.width = rect.width * dpr;
//     canvas.height = 120 * dpr;
//     canvas.style.width = rect.width + 'px';
//     canvas.style.height = '120px';
//     ctx.scale(dpr, dpr);

//     const width = rect.width;
//     const height = 120;

//     function draw() {
//         const waveform = analyser.getValue();

//         processAutotune();

//         ctx.fillStyle = '#0f0a1a';
//         ctx.fillRect(0, 0, width, height);

//         ctx.strokeStyle = 'rgba(167, 139, 250, 0.15)';
//         ctx.lineWidth = 1;
//         ctx.beginPath();
//         ctx.moveTo(0, height / 2);
//         ctx.lineTo(width, height / 2);
//         ctx.stroke();

//         ctx.beginPath();
//         ctx.strokeStyle = '#a78bfa';
//         ctx.lineWidth = 2;

//         const sliceWidth = width / waveform.length;
//         let x = 0;

//         for (let i = 0; i < waveform.length; i++) {
//             const v = waveform[i];
//             const y = (height / 2) + (v * height / 2);

//             if (i === 0) ctx.moveTo(x, y);
//             else ctx.lineTo(x, y);

//             x += sliceWidth;
//         }

//         ctx.stroke();
//         vizAnimFrame = requestAnimationFrame(draw);
//     }

//     if (vizAnimFrame) cancelAnimationFrame(vizAnimFrame);
//     draw();
// }


// // ============================================
// // 7. LISTENERS & TIMELINE INTERACTIONS
// // ============================================

// // Expose core player and recording functions globally so that 
// // inline HTML onclick attributes (e.g. onclick="toggleRecording()") work seamlessly.
// window.togglePlay = togglePlay;
// window.stopAudio = stopAudio;
// window.toggleLoop = toggleLoop;
// window.toggleRecording = toggleRecording;
// window.downloadRecording = downloadRecording;
// window.resetAllEffects = resetAllEffects;

// document.addEventListener('DOMContentLoaded', function() {
//     const fileInput = document.getElementById('labFileInput');
//     const dropZone = document.getElementById('labDropZone');
//     const timeline = document.getElementById('timelineSlider');

//     if (fileInput) {
//         fileInput.addEventListener('change', function() {
//             if (this.files.length > 0) handleFileUpload(this.files[0]);
//         });
//     }

//     if (dropZone) {
//         dropZone.addEventListener('dragover', (e) => {
//             e.preventDefault();
//             dropZone.classList.add('drag-over');
//         });
//         dropZone.addEventListener('dragleave', () => {
//             dropZone.classList.remove('drag-over');
//         });
//         dropZone.addEventListener('drop', (e) => {
//             e.preventDefault();
//             dropZone.classList.remove('drag-over');
//             if (e.dataTransfer.files.length > 0) {
//                 fileInput.files = e.dataTransfer.files;
//                 handleFileUpload(e.dataTransfer.files[0]);
//             }
//         });
//     }

//     if (timeline) {
//         timeline.addEventListener('mousedown', () => {
//             isDraggingTimeline = true;
//         });

//         timeline.addEventListener('input', function() {
//             updateTimeLabel(this.value, player ? player.buffer.duration : 0);
//         });

//         timeline.addEventListener('change', function() {
//             onTimelineSeek(this.value);
//             isDraggingTimeline = false;
//         });

//         timeline.addEventListener('touchstart', () => {
//             isDraggingTimeline = true;
//         });
//         timeline.addEventListener('touchend', function() {
//             onTimelineSeek(this.value);
//             isDraggingTimeline = false;
//         });
//     }

//     const sliderMap = {
//         'pitchSlider': setPitch,
//         'speedSlider': setSpeed,
//         'volumeSlider': setVolume,
//         'bassSlider': setBass,
//         'midSlider': setMid,
//         'trebleSlider': setTreble,
//         'lowPassSlider': setLowPass,
//         'highPassSlider': setHighPass,
//         'reverbSlider': setReverb,
//         'reverbDecaySlider': setReverbSize,
//         'delayTimeSlider': setDelayTime,
//         'delayFeedbackSlider': setDelayFeedback,
//         'delayWetSlider': setDelayWet,
//         'distortionSlider': setDistortion,
//         'tremoloSpeedSlider': setTremoloSpeed,
//         'tremoloDepthSlider': setTremoloDepth,
//         'compressorSlider': setCompressor,
//     };

//     for (const [id, fn] of Object.entries(sliderMap)) {
//         const slider = document.getElementById(id);
//         if (slider) {
//             slider.addEventListener('input', function() {
//                 fn(this.value);
//             });
//         }
//     }

//     // ==================================================
//     // AUTOTUNER EVENT LISTENERS (FIXED — moved from HTML)
//     // ==================================================
//     const autotuneToggle = document.getElementById('autotuneToggle');
//     if (autotuneToggle) {
//         autotuneToggle.addEventListener('change', function() {
//             toggleAutotune(this.checked);
//         });
//     }

//     const autotuneKeySelect = document.getElementById('autotuneKey');
//     if (autotuneKeySelect) {
//         autotuneKeySelect.addEventListener('change', function() {
//             setAutotuneKey(this.value);
//         });
//     }

//     const autotuneScaleSelect = document.getElementById('autotuneScale');
//     if (autotuneScaleSelect) {
//         autotuneScaleSelect.addEventListener('change', function() {
//             setAutotuneScale(this.value);
//         });
//     }

//     const autotuneSpeedSlider = document.getElementById('autotuneSpeedSlider');
//     if (autotuneSpeedSlider) {
//         autotuneSpeedSlider.addEventListener('input', function() {
//             setAutotuneSpeed(this.value);
//         });
//     }

//     // ==================================================
//     // RECORD & DOWNLOAD BUTTON EVENT LISTENERS
//     // ==================================================
//     const recordBtn = document.getElementById('recordBtn');
//     if (recordBtn) {
//         recordBtn.addEventListener('click', function(e) {
//             e.preventDefault();
//             toggleRecording();
//         });
//     }

//     const downloadBtn = document.getElementById('downloadBtn');
//     if (downloadBtn) {
//         // Initialize download button as visually disabled until a recording is completed
//         downloadBtn.disabled = true;
//         downloadBtn.style.opacity = '0.4';
//         downloadBtn.style.cursor = 'not-allowed';

//         downloadBtn.addEventListener('click', function(e) {
//             e.preventDefault();
//             downloadRecording();
//         });
//     }
// });

