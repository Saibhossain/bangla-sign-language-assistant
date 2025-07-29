// ===================== ENHANCED FRONTEND JavaScript =====================

class BanglaSignRecognition {
    constructor() {
        this.initializeElements();
        this.initializeVariables();
        this.setupEventListeners();
        this.initializeMediaPipe();
        this.startSessionTimer();
        this.connectWebSocket();
        this.loadStatistics();
    }

    initializeElements() {
        // Video and prediction elements
        this.video = document.getElementById('videoElement');
        this.predictionText = document.getElementById('predictionText');
        this.statusText = document.getElementById('statusText');
        this.statusDot = document.getElementById('statusDot');
        this.mainStatusDot = document.getElementById('mainStatusDot');
        this.overlayStatus = document.getElementById('overlayStatus');
        this.confidenceBar = document.getElementById('confidenceBar');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.historyList = document.getElementById('historyList');
        this.notification = document.getElementById('notification');
        this.currentModeDisplay = document.getElementById('currentMode');

        // Controls
        this.toggleCameraBtn = document.getElementById('toggleCamera');
        this.captureBtn = document.getElementById('captureBtn');
        this.fullscreenBtn = document.getElementById('fullscreenBtn');
        this.handGuideBtn = document.getElementById('handGuideBtn');

        // Settings
        this.mirrorToggle = document.getElementById('mirrorToggle');
        this.autoToggle = document.getElementById('autoToggle');
        this.voiceToggle = document.getElementById('voiceToggle');

        // Mode buttons
        this.modeButtons = document.querySelectorAll('.mode-btn');

        // Statistics elements
        this.totalPredictionsEl = document.getElementById('totalPredictions');
        this.accuracyRateEl = document.getElementById('accuracyRate');
        this.avgConfidenceEl = document.getElementById('avgConfidence');
        this.sessionTimeEl = document.getElementById('sessionTime');

        // Hand guide
        this.handGuide = document.getElementById('handGuide');
        this.guideContent = document.getElementById('guideContent');
    }

    initializeVariables() {
        this.currentMode = 'alphabet';
        this.predictionHistory = [];
        this.totalPredictions = 0;
        this.sessionStartTime = Date.now();
        this.cameraActive = true;
        this.autoCapture = false;
        this.mirrorView = true;
        this.voiceFeedback = true;
        this.isFullscreen = false;
        this.confidence = 0;
        this.websocket = null;
        this.hands = null;
        this.camera = null;
        this.statistics = {
            sessionTime: 0,
            accuracy: 0,
            avgConfidence: 0,
            modeStats: {
                alphabet: { predictions: 0, accuracy: 0 },
                digit: { predictions: 0, accuracy: 0 },
                word: { predictions: 0, accuracy: 0 }
            }
        };

        // API endpoints based on mode
        this.apiEndpoints = {
            alphabet: '/prediction',
            digit: '/prediction/digit',
            word: '/prediction/word'
        };

        this.modeNames = {
            alphabet: 'বর্ণমালা',
            digit: 'সংখ্যা',
            word: 'শব্দ'
        };
    }

    setupEventListeners() {
        // Mode switching
        this.modeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchMode(e.target.closest('.mode-btn').dataset.mode);
            });
        });

        // Camera controls
        this.toggleCameraBtn.addEventListener('click', () => this.toggleCamera());
        this.captureBtn.addEventListener('click', () => this.captureFrame());
        this.fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
        this.handGuideBtn.addEventListener('click', () => this.toggleHandGuide());

        // Settings toggles
        this.mirrorToggle.addEventListener('click', () => this.toggleMirror());
        this.autoToggle.addEventListener('click', () => this.toggleAutoCapture());
        this.voiceToggle.addEventListener('click', () => this.toggleVoiceFeedback());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case ' ': // Spacebar for capture
                    e.preventDefault();
                    this.captureFrame();
                    break;
                case 'f': // F for fullscreen
                case 'F':
                    this.toggleFullscreen();
                    break;
                case 'c': // C for camera toggle
                case 'C':
                    this.toggleCamera();
                    break;
                case '1':
                    this.switchMode('alphabet');
                    break;
                case '2':
                    this.switchMode('digit');
                    break;
                case '3':
                    this.switchMode('word');
                    break;
            }
        });

        // Window resize handler
        window.addEventListener('resize', () => {
            this.handleResize();
        });
    }

    async initializeMediaPipe() {
        try {
            // Initialize MediaPipe Hands
            this.hands = new Hands({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
            });

            this.hands.setOptions({
                maxNumHands: 1,
                modelComplexity: 1,
                minDetectionConfidence: 0.7,
                minTrackingConfidence: 0.7
            });

            this.hands.onResults((results) => this.onResults(results));

            // Initialize camera
            this.camera = new Camera(this.video, {
                onFrame: async () => {
                    if (this.hands && this.cameraActive) {
                        await this.hands.send({image: this.video});
                    }
                },
                width: 640,
                height: 480
            });

            await this.camera.start();
            this.updateStatus('ক্যামেরা সক্রিয়', true);
            this.showNotification('ক্যামেরা সফলভাবে চালু হয়েছে', 'success');

        } catch (error) {
            console.error('MediaPipe initialization error:', error);
            this.updateStatus('ক্যামেরা ত্রুটি', false);
            this.showNotification('ক্যামেরা চালু করতে সমস্যা হয়েছে', 'error');
        }
    }

    async onResults(results) {
        if (!this.cameraActive) return;

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];
            const flatLandmarks = landmarks.map(lm => [lm.x, lm.y, lm.z]).flat();

            // Send prediction request based on current mode
            try {
                const response = await fetch(`http://localhost:8000${this.apiEndpoints[this.currentMode]}`, {
                    method: 'GET',
                    headers: { 'Content-Type': 'application/json' }
                });

                if (response.ok) {
                    const data = await response.json();
                    this.processPrediction(data);
                    this.updateStatus('সংযুক্ত ও কার্যকর', true);
                } else {
                    throw new Error('Server response error');
                }
            } catch (error) {
                console.error('Prediction error:', error);
                this.handlePredictionError();
            }

            // Auto capture if enabled
            if (this.autoCapture) {
                this.captureFrame();
            }
        } else {
            // No hand detected
            this.predictionText.textContent = 'হাত সনাক্ত করা যায়নি';
            this.confidenceBar.style.width = '0%';
            this.confidenceValue.textContent = '--';
        }
    }

    processPrediction(data) {
        const prediction = data.prediction || 'অজানা';
        const confidence = data.confidence || 0;
        const handDetected = data.hand_detected || false;

        if (handDetected && prediction !== 'অজানা') {
            this.displayPrediction(prediction, confidence);
            this.addToHistory(prediction, confidence);
            this.updateModeStatistics(prediction, confidence);

            // Voice feedback if enabled
            if (this.voiceFeedback && confidence > 0.7) {
                this.speakPrediction(prediction);
            }
        } else {
            this.predictionText.textContent = 'স্পষ্'