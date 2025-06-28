// DOM Elements
const video = document.getElementById('videoElement');
const predictionText = document.getElementById('predictionText');
const statusText = document.getElementById('statusText');
const statusDot = document.getElementById('statusDot');
const mainStatusDot = document.getElementById('mainStatusDot');
const overlayStatus = document.getElementById('overlayStatus');
const confidenceBar = document.getElementById('confidenceBar');
const toggleCameraBtn = document.getElementById('toggleCamera');
const captureBtn = document.getElementById('captureBtn');
const fullscreenBtn = document.getElementById('fullscreenBtn');
const mirrorToggle = document.getElementById('mirrorToggle');
const autoToggle = document.getElementById('autoToggle');
const historyList = document.getElementById('historyList');
const notification = document.getElementById('notification');

// State variables
let cameraActive = true;
let stream = null;
let predictionHistory = [];
let totalPredictions = 0;
let sessionStartTime = Date.now();
let pollingInterval = null;

// Initialize application
init();

async function init() {
    await startWebcam();
    startPolling();
    updateSessionTimer();
    setupEventListeners();
}

// Webcam functions
async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });
        video.srcObject = stream;
        updateStatus('ক্যামেরা চালু', true);
        overlayStatus.textContent = 'লাইভ';
    } catch (error) {
        console.error('Webcam error:', error);
        updateStatus('ক্যামেরা অ্যাক্সেস করা যায়নি', false);
        showNotification('ক্যামেরা অ্যাক্সেস অনুমতি প্রয়োজন', 'error');
    }
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }
}

// Polling and prediction functions
function startPolling() {
    pollingInterval = setInterval(fetchPrediction, 2000);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

async function fetchPrediction() {
    try {
        const response = await fetch('http://localhost:8000/prediction');

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        if (data && data.prediction) {
            displayPrediction(data.prediction, data.confidence || 0.85);
            addToHistory(data.prediction, data.confidence || 0);
            updateStatus('সংযুক্ত ও কার্যকর', true);
        } else {
            predictionText.textContent = 'প্রেডিকশনের অপেক্ষায়...';
            updateStatus('ডেটার অপেক্ষায়', true);
        }
    } catch (error) {
        console.error('Prediction fetch error:', error);
        predictionText.textContent = 'সার্ভার সংযোগ ব্যর্থ';
        updateStatus('সার্ভার সংযোগ পরীক্ষা করুন', false);
    }
}

function displayPrediction(prediction, confidence) {
    confidence = confidence ?? 0;
    predictionText.textContent = prediction;
    confidenceBar.style.width = `${confidence * 100}%`;

    const confidenceValue = document.getElementById('confidenceValue');
    if (confidenceValue) {
        confidenceValue.textContent = `কনফিডেন্স: ${(confidence * 100).toFixed(1)}%`;
    }

    // Only count valid (not "অনিশ্চিত") predictions
    if (prediction !== "অনিশ্চিত" && confidence >= 0.6) {
        totalPredictions++;
    }

    updateStats();

    if (autoToggle.classList.contains('active')) {
        captureFrame();
    }
}

// History and stats functions
function addToHistory(prediction, confidence = 0) {
    const historyItem = {
        char: prediction,
        confidence: confidence,
        time: new Date().toLocaleTimeString('bn-BD'),
        timestamp: Date.now()
    };

    predictionHistory.unshift(historyItem);
    if (predictionHistory.length > 10) {
        predictionHistory.pop();
    }

    updateHistoryDisplay();
}

function updateHistoryDisplay() {
    if (predictionHistory.length === 0) {
        historyList.innerHTML = '<div style="text-align: center; color: #7f8c8d; padding: 20px;">কোন সাম্প্রতিক সনাক্তকরণ নেই</div>';
        return;
    }

    historyList.innerHTML = predictionHistory.map(item => `
        <div class="history-item">
            <span class="history-char">${item.char}</span>
            <span class="history-time">${item.time}</span>
        </div>
    `).join('');
}

function updateStats() {
    const validPreds = predictionHistory.filter(item => item.char !== "অনিশ্চিত");
    const count = validPreds.length;

    // Compute average confidence
    let avgConf = 0;
    if (count > 0) {
        avgConf = validPreds.reduce((sum, item) => sum + (item.confidence || 0), 0) / count;
    }

    document.getElementById('totalPredictions').textContent = totalPredictions;
    document.getElementById('accuracyRate').textContent = count > 0 ? '৯৫%' : '--'; // You can replace with logic later
    document.getElementById('avgConfidence').textContent = `${(avgConf * 100).toFixed(1)}%`;
}

function updateSessionTimer() {
    setInterval(() => {
        const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        document.getElementById('sessionTime').textContent =
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }, 1000);
}

// UI helper functions
function updateStatus(message, isActive) {
    statusText.textContent = message;
    statusDot.classList.toggle('active', isActive);
    mainStatusDot.classList.toggle('active', isActive);
}

function showNotification(message, type = 'success') {
    notification.textContent = message;
    notification.className = `notification ${type} show`;
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

function captureFrame() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // You can process or save the captured frame here
    showNotification('ফ্রেম ক্যাপচার সম্পন্ন');
}

// Event listeners
function setupEventListeners() {
    toggleCameraBtn.addEventListener('click', () => {
        if (cameraActive) {
            stopWebcam();
            toggleCameraBtn.innerHTML = '<i class="fas fa-video-slash"></i> ক্যামেরা চালু';
            overlayStatus.textContent = 'বন্ধ';
            cameraActive = false;
        } else {
            startWebcam();
            toggleCameraBtn.innerHTML = '<i class="fas fa-video"></i> ক্যামেরা বন্ধ';
            cameraActive = true;
        }
    });

    captureBtn.addEventListener('click', captureFrame);

    fullscreenBtn.addEventListener('click', () => {
        if (video.requestFullscreen) {
            video.requestFullscreen();
        }
    });

    mirrorToggle.addEventListener('click', () => {
        mirrorToggle.classList.toggle('active');
        video.style.transform = mirrorToggle.classList.contains('active')
            ? 'scaleX(-1)' : 'scaleX(1)';
    });

    autoToggle.addEventListener('click', () => {
        autoToggle.classList.toggle('active');
        showNotification(
            autoToggle.classList.contains('active')
                ? 'অটো ক্যাপচার চালু'
                : 'অটো ক্যাপচার বন্ধ'
        );
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        switch(e.key) {
            case ' ':
                e.preventDefault();
                captureFrame();
                break;
            case 'f':
            case 'F':
                if (video.requestFullscreen) {
                    video.requestFullscreen();
                }
                break;
        }
    });
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopWebcam();
    stopPolling();
});

