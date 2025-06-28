import os, uuid, cv2, json, numpy as np, joblib, mediapipe as mp, time, threading
from gtts import gTTS
from playsound3 import playsound
from collections import deque, Counter
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ---------- Load Bangla Dataset ----------
with open("bangla_signs_dataset.json", "r", encoding="utf-8") as f:
    bangla_dataset = json.load(f)

def get_speech_text(pred):
    entry = bangla_dataset.get(str(pred), {})
    return entry.get("sentence") or entry.get("letter") or entry.get("example_word") or "অজানা অক্ষর"

# ---------- Text-to-Speech ----------
def speak_bangla(text):
    def speak_thread():
        try:
            tts = gTTS(text=text, lang='bn')
            filename = f"temp_{uuid.uuid4().hex}.mp3"
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print(f"Error speaking Bangla: {e}")
    threading.Thread(target=speak_thread, daemon=True).start()

# ---------- Shared Prediction Store ----------
shared_prediction = {"prediction": ""}

# ---------- ML Model and Hand Detector ----------
def load_model(path):
    return joblib.load(path)

def init_hand_detector():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )

def extract_landmark_features(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()

# ---------- Background Worker ----------
def run_recognition():
    model = load_model("model/HandSign_Classifier_Alphabets_RF.pkl")
    hands = init_hand_detector()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    pred_queue = deque(maxlen=10)
    last_pred_time = 0
    delay_between_preds = 10  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                features = extract_landmark_features(hand_landmarks)
                if features.shape[0] == 63:
                    X = features.reshape(1, -1)
                    pred = model.predict(X)[0]
                    pred_queue.append(pred)

                    if current_time - last_pred_time >= delay_between_preds:
                        common_pred, count = Counter(pred_queue).most_common(1)[0]
                        if count >= 8:
                            sentence = get_speech_text(common_pred)
                            speak_bangla(sentence)
                            shared_prediction["prediction"] = bangla_dataset.get(str(common_pred), {}).get("letter", "Unknown")
                        else:
                            speak_bangla("আপনার হাতটি সঠিকভাবে দেখান।")
                            shared_prediction["prediction"] = "Uncertain"

                        last_pred_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- FastAPI App ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static HTML file
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

@app.get("/")
def serve_home():
    return FileResponse("templates/index.html")

@app.get("/prediction")
def get_prediction():
    return JSONResponse(content=shared_prediction)

# ---------- Startup ----------
@app.on_event("startup")
def start_background_task():
    threading.Thread(target=run_recognition, daemon=True).start()

@app.get("/")
def serve_index():
    return FileResponse("templates/index.html")