/**
 * MISO THE CAT
 * 
 * Behaviors:
 * - Sleeps when idle (default state)
 * - Walks around randomly every 20-40 seconds
 * - Says random music tips in speech bubbles
 * - Meows (real cat sound!) and shows a heart when clicked
 * - Wakes up on mouse movement
 * - Falls asleep after 15 seconds of no interaction
 */

// ============================================
// CAT STATE
// ============================================

let catElement = null;
let catContainer = null;
let bubbleElement = null;
let catState = 'sleeping';  // 'sleeping', 'awake', 'walking', 'happy'
let idleTimer = null;
let walkTimer = null;
let blinkTimer = null;
let speechTimer = null;
let lastMouseMove = Date.now();
let currentPosition = 60; // px from left edge

// Meow audio element
let meowAudio = null;

// Random music-themed messages
const CAT_MESSAGES = [
    "Meow! I'm Miso 🎵",
    "Did you know? A4 = 440 Hz is the universal tuning standard!",
    "Try uploading a guitar recording — I love clean audio!",
    "Purr... the FFT is my favorite algorithm~",
    "Autocorrelation finds pitch by matching sound to itself!",
    "Chords are just 3+ notes played together, meow~",
    "The Sing the Note game is my favorite! Try it!",
    "Chroma features have 12 bins, one for each pitch class!",
    "I can hear frequencies from 20 Hz to 20,000 Hz... probably!",
    "Middle C is 261.63 Hz, remember that! 🎹",
    "The Nyquist theorem says: sample rate ≥ 2× max frequency!",
    "Try the Audio Lab — you can make your voice sound like a robot!",
    "*yawn* Signal processing is tiring work...",
    "Fun fact: Cats can hear up to 64,000 Hz!",
    "Have you tried the piano tab? It's a real Yamaha grand!",
    "Nyaa~ Upload a song and let me analyze it!",
    "Machine learning + DSP = magic ✨",
    "The spectrogram shows time AND frequency at once!",
    "Reverb simulates sound bouncing off walls~",
    "Purrr... I love good chord progressions",
];

const CAT_CLICK_MESSAGES = [
    "Meow! 💕",
    "Purrr~",
    "Nyaa~ 🎶",
    "*happy meow*",
    "Meow meow!",
    "I like pets! 💜",
];

// ============================================
// INITIALIZE CAT
// ============================================

function initCat() {
    catContainer = document.getElementById('misoCat');
    if (!catContainer) return;

    catElement = catContainer.querySelector('.miso-cat');
    bubbleElement = catContainer.querySelector('.cat-speech-bubble');

    // Preload the meow sound
    initMeowAudio();

    // Click to meow
    catContainer.addEventListener('click', onCatClick);

    // Start behaviors
    startBlinking();
    scheduleNextWalk();
    scheduleNextSpeech();
    trackMouseActivity();

    // Wake up on scroll
    window.addEventListener('scroll', () => {
        if (catState === 'sleeping') {
            wakeUp();
        }
    }, { passive: true });
}

// ============================================
// MEOW AUDIO (real cat sound from cat.wav)
// ============================================

function initMeowAudio() {
    // Create audio element and preload the file
    // NOTE: Adjust the path below if your meow.wav is in a different location
    meowAudio = new Audio('/static/sounds/meow.wav');
    meowAudio.preload = 'auto';
    meowAudio.volume = 0.6; // 0.0 to 1.0 — tweak as needed
}

function playMeow() {
    if (!meowAudio) return;

    // Reset to start so rapid clicks all play
    meowAudio.currentTime = 0;

    // .play() returns a promise — catch autoplay-blocked errors gracefully
    const playPromise = meowAudio.play();
    if (playPromise !== undefined) {
        playPromise.catch(err => {
            console.log('Meow blocked by browser (user needs to interact first):', err);
        });
    }
}

// ============================================
// CAT STATES
// ============================================

function setState(newState) {
    if (!catElement) return;

    // Remove all state classes
    catElement.classList.remove('sleeping', 'walking', 'happy');

    catState = newState;

    if (newState !== 'awake') {
        catElement.classList.add(newState);
    }

    // Show/hide sleep Z's
    const zs = catContainer.querySelectorAll('.sleep-z');
    zs.forEach(z => {
        z.style.display = newState === 'sleeping' ? 'block' : 'none';
    });
}

function fallAsleep() {
    if (catState === 'sleeping') return;
    setState('sleeping');
    hideSpeechBubble();
}

function wakeUp() {
    if (catState !== 'sleeping') return;
    setState('awake');
    showSpeech("*yawn* Oh hi! 🐱", 3000);
    resetIdleTimer();
}

// ============================================
// BLINKING
// ============================================

function startBlinking() {
    function blink() {
        if (catState === 'sleeping' || catState === 'happy') {
            blinkTimer = setTimeout(blink, 2000 + Math.random() * 3000);
            return;
        }

        const eyes = catContainer.querySelectorAll('.cat-eye');
        eyes.forEach(eye => {
            eye.style.height = '2px';
        });

        setTimeout(() => {
            eyes.forEach(eye => {
                eye.style.height = '';
            });
        }, 150);

        // Schedule next blink
        blinkTimer = setTimeout(blink, 2000 + Math.random() * 4000);
    }
    blink();
}

// ============================================
// WALKING
// ============================================

function scheduleNextWalk() {
    // Walk every 20-40 seconds
    walkTimer = setTimeout(() => {
        if (catState !== 'sleeping') {
            walkRandomly();
        }
        scheduleNextWalk();
    }, 20000 + Math.random() * 20000);
}

function walkRandomly() {
    if (!catContainer) return;

    // Choose new random position
    const maxLeft = window.innerWidth - 150;
    const minLeft = 20;
    const newPosition = minLeft + Math.random() * (maxLeft - minLeft);

    // Face the direction of movement
    if (newPosition < currentPosition) {
        catContainer.classList.add('facing-left');
    } else {
        catContainer.classList.remove('facing-left');
    }

    setState('walking');
    currentPosition = newPosition;
    catContainer.style.left = newPosition + 'px';

    // Stop walking after arrival
    setTimeout(() => {
        if (catState === 'walking') {
            setState('awake');
        }
    }, 3000);
}

// ============================================
// SPEECH BUBBLES
// ============================================

function scheduleNextSpeech() {
    speechTimer = setTimeout(() => {
        if (catState !== 'sleeping' && catState !== 'happy') {
            const msg = CAT_MESSAGES[Math.floor(Math.random() * CAT_MESSAGES.length)];
            showSpeech(msg, 5000);
        }
        scheduleNextSpeech();
    }, 15000 + Math.random() * 15000);
}

function showSpeech(text, duration = 4000) {
    if (!bubbleElement) return;

    bubbleElement.textContent = text;
    bubbleElement.classList.add('visible');

    clearTimeout(bubbleElement.hideTimer);
    bubbleElement.hideTimer = setTimeout(() => {
        hideSpeechBubble();
    }, duration);
}

function hideSpeechBubble() {
    if (bubbleElement) {
        bubbleElement.classList.remove('visible');
    }
}

// ============================================
// CLICK HANDLER
// ============================================

function onCatClick(e) {
    e.stopPropagation();

    // Wake up if sleeping — AND meow so user gets audio feedback
    if (catState === 'sleeping') {
        wakeUp();
        playMeow();
        return;
    }

    // Happy bounce
    setState('happy');
    playMeow();

    // Show heart
    const heart = document.createElement('span');
    heart.className = 'cat-heart';
    heart.textContent = '💜';
    catContainer.appendChild(heart);
    setTimeout(() => heart.remove(), 1200);

    // Show random meow message
    const msg = CAT_CLICK_MESSAGES[Math.floor(Math.random() * CAT_CLICK_MESSAGES.length)];
    showSpeech(msg, 2000);

    // Return to awake after bounce
    setTimeout(() => {
        if (catState === 'happy') {
            setState('awake');
        }
    }, 600);

    resetIdleTimer();
}

// ============================================
// MOUSE TRACKING (auto-sleep when idle)
// ============================================

function trackMouseActivity() {
    document.addEventListener('mousemove', () => {
        lastMouseMove = Date.now();

        if (catState === 'sleeping') {
            wakeUp();
        }

        resetIdleTimer();
    }, { passive: true });

    // Check every 2 seconds if user is idle
    setInterval(() => {
        const idleTime = Date.now() - lastMouseMove;
        if (idleTime > 15000 && catState === 'awake') {
            fallAsleep();
        }
    }, 2000);
}

function resetIdleTimer() {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
        if (catState === 'awake') {
            fallAsleep();
        }
    }, 15000);
}

// ============================================
// INITIALIZE ON PAGE LOAD
// ============================================

document.addEventListener('DOMContentLoaded', initCat);