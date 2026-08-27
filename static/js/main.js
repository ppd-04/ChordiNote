const canvas = document.getElementById('particleCanvas');
const ctx = canvas.getContext('2d');
// getContext('2d') gives us a drawing tool for the canvas

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Array to store all particle objects
const particles = [];
const musicNotes = ['♪', '♫', '♬', '♩', '🎵'];
const colors = ['#a78bfa', '#f472b6', '#67e8f9', '#c4b5fd', '#f9a8d4'];

class Particle {
    constructor() {
        this.reset();
    }

    reset() {
        this.x = Math.random() * canvas.width;
        // Math.random() returns a number between 0 and 1
        // Multiply by canvas.width to get a random X position

        this.y = canvas.height + 20;
        // Start below the screen so they float UP into view

        this.size = Math.random() * 16 + 10;
        this.speedY = Math.random() * 0.8 + 0.3;
        // How fast the note moves upward

        this.speedX = (Math.random() - 0.5) * 0.5;
        // Slight horizontal drift (can be negative = left)

        this.opacity = Math.random() * 0.4 + 0.1;
        this.note = musicNotes[Math.floor(Math.random() * musicNotes.length)];
        // Pick a random music symbol from the array

        this.color = colors[Math.floor(Math.random() * colors.length)];
        this.rotation = Math.random() * Math.PI * 2;
        this.rotationSpeed = (Math.random() - 0.5) * 0.02;
    }

    update() {
        this.y -= this.speedY;
        this.x += this.speedX;
        this.rotation += this.rotationSpeed;

        if (this.y < -30) {
            this.reset();
        }
    }

    draw() {
        ctx.save();
        // save() stores the current canvas state so we can restore it later

        ctx.translate(this.x, this.y);
        ctx.rotate(this.rotation);
        ctx.globalAlpha = this.opacity;
        ctx.font = `${this.size}px serif`;
        ctx.fillStyle = this.color;
        ctx.textAlign = 'center';
        ctx.fillText(this.note, 0, 0);

        ctx.restore();
        // restore() brings back the canvas state before we rotated/translated
    }
}

// Create 25 particles
for (let i = 0; i < 25; i++) {
    const p = new Particle();
    p.y = Math.random() * canvas.height;
    // Spread them across the screen initially
    particles.push(p);
}

function animateParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Clear the entire canvas each frame

    particles.forEach(p => {
        p.update();
        p.draw();
    });

    requestAnimationFrame(animateParticles);
    // requestAnimationFrame tells the browser:
    // "Call this function again before the next screen repaint"
    // This creates a smooth 60fps animation loop
}

animateParticles();


/* ============================================
   2. ANIMATED WAVEFORM ON HERO SECTION
   Draws a smooth, flowing sine wave on the canvas
   ============================================ */

const waveCanvas = document.getElementById('waveformCanvas');
if (waveCanvas) {
    const wCtx = waveCanvas.getContext('2d');
    let waveTime = 0;

    function drawWaveform() {
        const w = waveCanvas.width;
        const h = waveCanvas.height;

        wCtx.clearRect(0, 0, w, h);

        // Draw 3 overlapping waves with different colors
        const waves = [
            { color: 'rgba(167, 139, 250, 0.6)', amp: 40, freq: 0.02, speed: 0.03 },
            { color: 'rgba(244, 114, 182, 0.4)', amp: 30, freq: 0.025, speed: 0.05 },
            { color: 'rgba(103, 232, 249, 0.3)', amp: 20, freq: 0.03, speed: 0.04 },
        ];

        waves.forEach(wave => {
            wCtx.beginPath();
            // beginPath() starts a new drawing path

            for (let x = 0; x < w; x++) {
                const y = h / 2 +
                    Math.sin(x * wave.freq + waveTime * wave.speed) * wave.amp +
                    Math.sin(x * wave.freq * 2.5 + waveTime * wave.speed * 1.5) * (wave.amp * 0.3);
                // This creates a complex wave by combining two sine waves
                // sine wave 1: main wave
                // sine wave 2: smaller harmonic on top

                if (x === 0) {
                    wCtx.moveTo(x, y);
                } else {
                    wCtx.lineTo(x, y);
                }
            }

            wCtx.strokeStyle = wave.color;
            wCtx.lineWidth = 2.5;
            wCtx.stroke();
        });

        waveTime++;
        requestAnimationFrame(drawWaveform);
    }

    drawWaveform();
}


/* ============================================
   3. TYPEWRITER EFFECT
   Types out words one by one in the hero title
   ============================================ */

const typewriterEl = document.getElementById('typewriterText');
if (typewriterEl) {
    const words = ['Chords', 'Notes', 'Melodies', 'Harmonics', 'Pitches'];
    let wordIndex = 0;
    let charIndex = 0;
    let isDeleting = false;

    function typeWriter() {
        const currentWord = words[wordIndex];

        if (isDeleting) {
            typewriterEl.textContent = currentWord.substring(0, charIndex - 1);
            charIndex--;
        } else {
            typewriterEl.textContent = currentWord.substring(0, charIndex + 1);
            charIndex++;
        }

        let speed = isDeleting ? 50 : 100;
        // Deleting is faster than typing

        if (!isDeleting && charIndex === currentWord.length) {
            speed = 2000;
            // Pause at the full word for 2 seconds
            isDeleting = true;
        } else if (isDeleting && charIndex === 0) {
            isDeleting = false;
            wordIndex = (wordIndex + 1) % words.length;
            // Move to next word, loop back to 0 when reaching the end
            speed = 500;
        }

        setTimeout(typeWriter, speed);
    }

    typeWriter();
}


/* ============================================
   4. SCROLL REVEAL ANIMATION
   Elements fade in when they enter the viewport
   ============================================ */

function revealOnScroll() {
    const reveals = document.querySelectorAll('.reveal');

    reveals.forEach(el => {
        const windowHeight = window.innerHeight;
        const elementTop = el.getBoundingClientRect().top;
        // getBoundingClientRect() returns the element's position
        // relative to the viewport (the visible part of the page)

        const revealPoint = 120;
        // How many pixels from the bottom before the element appears

        if (elementTop < windowHeight - revealPoint) {
            el.classList.add('active');
        }
    });
}

window.addEventListener('scroll', revealOnScroll);
revealOnScroll();
// Run once on load to reveal elements already in view


/* ============================================
   5. NAVBAR SCROLL EFFECT
   Makes navbar more opaque when user scrolls down
   ============================================ */

window.addEventListener('scroll', () => {
    const navbar = document.getElementById('navbar');
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});