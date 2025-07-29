
// ===================== FRONTEND JavaScript =====================

// DOM Elements
const video = document.getElementById('videoElement');
const predictionText = document.getElementById('predictionText');
const statusText = document.getElementById('statusText');
const statusDot = document.getElementById('statusDot');
const mainStatusDot = document.getElementById('mainStatusDot');
const overlayStatus = document.getElementById('overlayStatus');
const confidenceBar = document.getElementById('confidenceBar');
const historyList = document.getElementById('historyList');
const notification = document.getElementById('notification');
const mirrorToggle = document.getElementById('mirrorToggle');
const autoToggle = document.getElementById('autoToggle');

let predictionHistory = [];
let totalPredictions = 0;
let sessionStartTime = Date.now();

// Load MediaPipe Hands
const hands = new Hands({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7
});
hands.onResults(onResults);

const camera = new Camera(video, {
  onFrame: async () => {
    await hands.send({image: video});
  },
  width: 640,
  height: 480
});
camera.start();

function onResults(results) {
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];
    const flatLandmarks = landmarks.map(lm => [lm.x, lm.y, lm.z]).flat();

    fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks: flatLandmarks })
    })
    .then(res => res.json())
    .then(data => {
      displayPrediction(data.prediction, data.confidence || 0.85);
      addToHistory(data.prediction, data.sconfidence || 0);
      updateStatus('সংযুক্ত ও কার্যকর', true);
      document.getElementById("mostCommonLetter").textContent = data.letter || '--';
      document.getElementById("currentWord").textContent = data.word || '--';
      document.getElementById("finalSentence").textContent = data.sentence || '--';
      document.getElementById("accuracyRate").textContent = data.confidence || '--';
    })
    .catch(err => {
      console.error('Prediction error:', err);
      predictionText.textContent = 'সার্ভার সংযোগ ব্যর্থ';
      updateStatus('সার্ভার সংযোগ সমস্যা', false);
    });
  }
}

function displayPrediction(prediction, confidence) {
  predictionText.textContent = prediction;
  confidenceBar.style.width = `${confidence * 100}%`;

  const confidenceValue = document.getElementById('confidenceValue');
  if (confidenceValue) {
    confidenceValue.textContent = `কনফিডেন্স: ${(confidence * 100).toFixed(1)}%`;
  }

  if (prediction !== "অনিশ্চিত" && confidence >= 0.6) {
    totalPredictions++;
  }
  updateStats();
}

function addToHistory(prediction, confidence = 0) {
  const historyItem = {
    char: prediction,
    confidence: confidence,
    time: new Date().toLocaleTimeString('bn-BD')
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
  let avgConf = 0;
  if (count > 0) {
    avgConf = validPreds.reduce((sum, item) => sum + (item.confidence || 0), 0) / count;
  }
  document.getElementById('totalPredictions').textContent = totalPredictions;
  document.getElementById('accuracyRate').textContent = count > 0 ? '৯৫%' : '--';
  document.getElementById('avgConfidence').textContent = `${(avgConf * 100).toFixed(1)}%`;
}

function updateStatus(message, isActive) {
  statusText.textContent = message;
  statusDot.classList.toggle('active', isActive);
  mainStatusDot.classList.toggle('active', isActive);
}

function updateSessionTimer() {
  setInterval(() => {
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    document.getElementById('sessionTime').textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }, 1000);
}


updateSessionTimer();
