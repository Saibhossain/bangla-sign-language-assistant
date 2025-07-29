# ===================== BACKEND Python (FastAPI) =====================
import os, uuid, json, joblib, time, threading,  numpy as np
import kagglehub
from collections import deque, Counter
from gtts import gTTS
from playsound3 import playsound
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import warnings
from BanglaDigitAlphabet.alph_to_word_fuzzyMatching_NLP import WordBuilder, valid_words

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

word_builder = WordBuilder()
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ---------- Load Bangla Dataset ----------
with open("../BSL_Learning_Tool/bangla_signs_dataset.json", "r", encoding="utf-8") as f:
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

# ---------- Prediction API ----------
class LandmarkInput(BaseModel):
    landmarks: list[float]


# Download latest version
path = kagglehub.model_download("saibhossain/bangla_handsign_alphabets_classifier_rf1/scikitLearn/default")
print("Path to model files:", path)
filename = "HandSign_Classifier_Alphabets_RF.pkl"
model_path = os.path.join(path, filename)
model = joblib.load(model_path)



shared_prediction = {
    "prediction": "",
    "letter": "",
    "word": "",
    "sentence": "",
    "confidence": 0.0
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/templates", StaticFiles(directory="templates"), name="templates")

@app.get("/")
def serve_home():
    return FileResponse("../BSL_Learning_Tool/templates/index.html")

# ---------- ESP ----------
class DistanceData(BaseModel):
    distance: float

@app.post("/distance_data")
async def receive_distance(data: DistanceData):
    distance = data.distance
    print(f"📥 Distance from ESP32: {distance:.2f} cm")

    response_data = {
        "distance": distance,
        "prediction": shared_prediction.get("prediction", ""),
        "letter": shared_prediction.get("letter", ""),
        "word": shared_prediction.get("word", ""),
        "sentence": shared_prediction.get("sentence", "")
    }

    return response_data

@app.post("/predict")
def predict(input: LandmarkInput):
    current_time = time.time()
    delay_between_preds = 7
    word_gap_timeout = 7

    # Create global state once
    if not hasattr(predict, "last_pred_time"):
        predict.last_pred_time = 0
        predict.last_word_time = 0
        predict.pred_queue = deque(maxlen=8)

    try:
        if len(input.landmarks) != 63:
            return JSONResponse(status_code=400, content={"error": "Invalid landmark shape"})

        X = np.array(input.landmarks).reshape(1, -1)
        pred_proba = model.predict_proba(X)[0]
        pred_idx = np.argmax(pred_proba)
        confidence = float(pred_proba[pred_idx])

        raw_pred = str(model.classes_[pred_idx])
        letter_pred = bangla_dataset.get(raw_pred, {}).get("letter", raw_pred)



        predict.pred_queue.append((letter_pred, confidence))

        most_common_letter = ""
        if current_time - predict.last_pred_time >= delay_between_preds:
            common_preds = Counter([p for p, _ in predict.pred_queue])
            most_common_letter, count = common_preds.most_common(1)[0]

            if count >= 6:
                word_builder.add_letter(most_common_letter)
                print(f"➕ Letter added: {most_common_letter}")
                predict.last_word_time = current_time
            else:
                print("🟡 Low confidence, skipping letter.")

            predict.last_pred_time = current_time

        if current_time - predict.last_word_time >= word_gap_timeout and word_builder.letter_buffer:
            word = word_builder.commit_word()
            matched_word, success = word_builder.try_add_word(word, valid_words)
            if success:
                print(f"✅ Final Word: {matched_word}")
                speak_bangla(matched_word)
            else:
                print(f"❌ Invalid word: {matched_word}")
                speak_bangla("অবৈধ শব্দ")

        current_word = ''.join(word_builder.letter_buffer)
        sentence = word_builder.get_sentence()

        shared_prediction.update({
            "prediction": letter_pred,
            "letter": most_common_letter,
            "word": current_word,
            "sentence": sentence,
            "confidence": round(confidence * 100, 2)
        })

        return shared_prediction

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
