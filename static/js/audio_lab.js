/**
 * ChordSense Audio Lab Engine
 * Complete real-time browser audio processing with Scrubbing support
 */

let player = null;
let isPlaying = false;
let isLooping = false;
let audioLoaded = false;
let effectsInitialized = false;

// Time Tracking Variables
let playOffset = 0;       // Position in the song where playback started (seconds)
let startTime = 0;        // Tone.now() value when play was clicked (seconds)
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
        analyser = new Tone.Analyser({ type: "waveform", size: 1024 });

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

        effectsInitialized = true;
        console.log("🎛️ Audio Lab effects chain connected successfully.");
    } catch (err) {
        console.error("Failed to init effects chain:", err);
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
                audioLoaded = true;
                player.connect(pitchShift);

                const duration = player.buffer.duration;
                
                // Set up the timeline scrubber maximum range
                const timeline = document.getElementById('timelineSlider');
                if (timeline) {
                    timeline.max = duration;
                    timeline.value = 0;
                }
                
                playOffset = 0;
                updateTimeLabel(0, duration);
                enableAllControls();
                startVisualization();
            },
            onerror: (err) => {
                console.error("Tone.Player load error:", err);
                if (statusEl) statusEl.innerHTML = '<span style="color: #ef4444;">✗ Error reading file</span>';
            }
        });

    } catch (e) {
        console.error("Audio Load exception:", e);
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
    
    if (playBtn) { playBtn.disabled = false; playBtn.style.opacity = '1'; playBtn.style.cursor = 'pointer'; }
    if (stopBtn) { stopBtn.disabled = false; stopBtn.style.opacity = '1'; stopBtn.style.cursor = 'pointer'; }
    if (loopBtn) { loopBtn.disabled = false; loopBtn.style.opacity = '1'; loopBtn.style.cursor = 'pointer'; }
    if (timeline) { timeline.disabled = false; timeline.style.opacity = '1'; timeline.style.cursor = 'pointer'; }

    document.querySelectorAll('.lab-slider').forEach(slider => {
        slider.disabled = false;
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
        // PAUSE: Store where we paused, stop player, clear timer
        playOffset = getElapsedTime();
        player.stop();
        isPlaying = false;
        updatePlayButton(false);
        clearInterval(progressInterval);
    } else {
        // PLAY: Start player from offset, record current time, run loop
        if (playOffset >= player.buffer.duration) {
            playOffset = 0; // restart if we reached the end
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

// Helper: Calculate precisely where the playhead is right now
function getElapsedTime() {
    if (!isPlaying) return playOffset;
    
    const elapsed = Tone.now() - startTime;
    // Account for playback rate (speed)
    const rate = player ? player.playbackRate : 1;
    const currentPosition = playOffset + (elapsed * rate);
    
    // Stop naturally when end is reached
    if (currentPosition >= player.buffer.duration) {
        if (isLooping) {
            startTime = Tone.now();
            playOffset = 0;
            return 0;
        } else {
            // Force stop
            setTimeout(() => { stopAudio(); }, 10);
            return player.buffer.duration;
        }
    }
    return currentPosition;
}

// Timer Loop: Updates the scrubber handle and elapsed time labels
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

// Format seconds into readable MM:SS layout
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

// Handler: When user interacts with the timeline scrubber
function onTimelineSeek(val) {
    if (!player || !audioLoaded) return;
    
    playOffset = parseFloat(val);
    updateTimeLabel(playOffset, player.buffer.duration);

    if (isPlaying) {
        // To seek while playing, we must stop, update startTime, and restart
        player.stop();
        startTime = Tone.now();
        player.start(0, playOffset);
    }
}

// ============================================
// 4. REAL-TIME EFFECT PARAMETERS
// ============================================
function setPitch(val) {
    if (pitchShift) pitchShift.pitch = parseFloat(val);
    updateLabel('pitchValue', `${val > 0 ? '+' : ''}${val} st`);
}

function setSpeed(val) {
    if (player) {
        // If playing, we must recalculate offset on-the-fly to prevent jumps
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
// 5. RESET ALL
// ============================================
function resetAllEffects() {
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
        'compressorSlider': 0
    };

    for (const [id, val] of Object.entries(defaults)) {
        const slider = document.getElementById(id);
        if (slider) slider.value = val;
    }
}

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
// 7. LISTENERS & TIMELINE INTERACTIONS (NEW)
// ============================================
document.addEventListener('DOMContentLoaded', function() {
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

    // Scrubber drag-interactions
    if (timeline) {
        timeline.addEventListener('mousedown', () => {
            isDraggingTimeline = true;
        });

        timeline.addEventListener('input', function() {
            // Update labels as the handle is actively moving
            updateTimeLabel(this.value, player ? player.buffer.duration : 0);
        });

        timeline.addEventListener('change', function() {
            // Seek to new position on release
            onTimelineSeek(this.value);
            isDraggingTimeline = false;
        });

        // Touch support for mobile devices
        timeline.addEventListener('touchstart', () => {
            isDraggingTimeline = true;
        });
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
});