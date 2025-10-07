/**
 * Dance Dance Live - Client Application
 * Handles webcam, video playback, and real-time pose scoring
 */

class DanceDanceApp {
    constructor() {
        this.referenceVideo = document.getElementById('referenceVideo');
        this.webcamVideo = document.getElementById('webcamVideo');
        this.webcamCanvas = document.getElementById('webcamCanvas');
        this.ctx = this.webcamCanvas.getContext('2d');

        this.startBtn = document.getElementById('startBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.statusText = document.getElementById('statusText');
        this.timerDisplay = document.getElementById('timer');

        this.isPlaying = false;
        this.startTime = null;
        this.webcamStream = null;
        this.captureInterval = null;
        this.scoreHistory = [];
        this.videoInfo = null;

        this.init();
    }

    async init() {
        // Fetch video info
        try {
            const response = await fetch('/api/video-info');
            this.videoInfo = await response.json();
            console.log('Video info loaded:', this.videoInfo);
        } catch (error) {
            console.error('Failed to load video info:', error);
            this.statusText.textContent = 'Error loading video info';
            return;
        }

        // Wait for video to load
        await new Promise((resolve) => {
            if (this.referenceVideo.readyState >= 3) {
                resolve();
            } else {
                this.referenceVideo.addEventListener('loadeddata', resolve, { once: true });
            }
        });

        console.log('Reference video loaded');

        // Setup webcam
        await this.setupWebcam();

        // Event listeners
        this.startBtn.addEventListener('click', () => this.start());
        this.resetBtn.addEventListener('click', () => this.reset());

        // Video events
        this.referenceVideo.addEventListener('ended', () => this.onVideoEnded());

        this.statusText.textContent = 'Ready to dance! Click Start to begin.';
    }

    async setupWebcam() {
        try {
            this.webcamStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            });

            this.webcamVideo.srcObject = this.webcamStream;

            // Wait for video to be ready
            await new Promise(resolve => {
                this.webcamVideo.onloadedmetadata = resolve;
            });

            // Setup canvas dimensions
            this.webcamCanvas.width = this.webcamVideo.videoWidth;
            this.webcamCanvas.height = this.webcamVideo.videoHeight;

            console.log('Webcam setup complete');
        } catch (error) {
            console.error('Webcam access denied:', error);
            this.statusText.textContent = 'Error: Webcam access required';
        }
    }

    async start() {
        if (this.isPlaying) return;

        this.isPlaying = true;
        this.scoreHistory = [];
        this.startTime = Date.now();

        // Play reference video with error handling
        this.referenceVideo.currentTime = 0;

        try {
            await this.referenceVideo.play();
            console.log('Video playing successfully');
        } catch (error) {
            console.error('Error playing video:', error);
            this.statusText.textContent = 'Error: Click the video to start';
            this.isPlaying = false;
            this.startBtn.disabled = false;
            return;
        }

        // Start capturing and scoring
        this.startCapture();

        // Update UI
        this.startBtn.disabled = true;
        this.statusText.textContent = '🎬 Dancing! Match the moves!';

        console.log('Dance started');
    }

    startCapture() {
        let frameCount = 0;

        // Capture and analyze every 30th frame (once per second at 30fps)
        this.captureInterval = setInterval(async () => {
            if (!this.isPlaying) return;

            const elapsed = (Date.now() - this.startTime) / 1000;

            // Update timer
            this.updateTimer(elapsed);

            // Capture frame every 30th iteration (~1 second)
            if (frameCount % 30 === 0) {
                await this.captureAndAnalyze(elapsed);
            }

            frameCount++;
        }, 1000 / 30); // 30 FPS
    }

    async captureAndAnalyze(timestamp) {
        try {
            // Capture webcam frame to canvas
            this.ctx.drawImage(
                this.webcamVideo,
                0, 0,
                this.webcamCanvas.width,
                this.webcamCanvas.height
            );

            // Get frame as base64
            const frameData = this.webcamCanvas.toDataURL('image/jpeg', 0.8);

            // Send to backend for analysis
            const response = await fetch('/api/analyze-pose', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    frame: frameData,
                    timestamp: timestamp
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.updateScore(result.scores);
                this.scoreHistory.push({
                    timestamp: timestamp,
                    frame: result.frame_number,
                    ...result.scores
                });
            }
        } catch (error) {
            console.error('Error analyzing pose:', error);
        }
    }

    updateScore(scores) {
        if (!scores) return;

        document.getElementById('currentScore').textContent = Math.round(scores.overall_score);
        document.getElementById('angleScore').textContent = scores.angle_score;
        document.getElementById('orientationScore').textContent = scores.orientation_score;
        document.getElementById('extensionScore').textContent = scores.extension_score;

        // Add visual feedback based on score
        const scoreValue = document.getElementById('currentScore');
        if (scores.overall_score >= 80) {
            scoreValue.style.color = '#10b981';
        } else if (scores.overall_score >= 60) {
            scoreValue.style.color = '#f59e0b';
        } else {
            scoreValue.style.color = '#ef4444';
        }
    }

    updateTimer(elapsed) {
        const minutes = Math.floor(elapsed / 60);
        const seconds = Math.floor(elapsed % 60);
        this.timerDisplay.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }

    onVideoEnded() {
        this.stop();
        this.showResults();
    }

    stop() {
        this.isPlaying = false;

        if (this.captureInterval) {
            clearInterval(this.captureInterval);
            this.captureInterval = null;
        }

        this.startBtn.disabled = false;
        this.statusText.textContent = '✅ Dance complete!';
        console.log('Dance stopped');
    }

    reset() {
        this.stop();

        this.scoreHistory = [];
        this.referenceVideo.currentTime = 0;
        this.referenceVideo.pause();

        // Reset scores
        document.getElementById('currentScore').textContent = '0';
        document.getElementById('angleScore').textContent = '0';
        document.getElementById('orientationScore').textContent = '0';
        document.getElementById('extensionScore').textContent = '0';
        this.timerDisplay.textContent = '0:00';

        // Hide results
        document.getElementById('resultsContainer').style.display = 'none';

        this.statusText.textContent = 'Ready to dance! Click Start to begin.';
    }

    showResults() {
        const resultsContainer = document.getElementById('resultsContainer');
        resultsContainer.style.display = 'block';

        if (this.scoreHistory.length === 0) {
            this.statusText.textContent = 'No scores recorded';
            return;
        }

        // Calculate statistics
        const scores = this.scoreHistory.map(s => s.overall_score);
        const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
        const bestScore = Math.max(...scores);

        document.getElementById('avgScore').textContent = avgScore.toFixed(1);
        document.getElementById('bestScore').textContent = bestScore.toFixed(1);
        document.getElementById('framesScored').textContent = this.scoreHistory.length;

        // Draw chart
        this.drawScoreChart();
    }

    drawScoreChart() {
        const canvas = document.getElementById('scoreChart');
        const ctx = canvas.getContext('2d');

        // Destroy existing chart if any
        if (this.chart) {
            this.chart.destroy();
        }

        const timestamps = this.scoreHistory.map(s => s.timestamp.toFixed(1) + 's');
        const scores = this.scoreHistory.map(s => s.overall_score);

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timestamps,
                datasets: [{
                    label: 'Score Over Time',
                    data: scores,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    }
                }
            }
        });
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DanceDanceApp();
});
