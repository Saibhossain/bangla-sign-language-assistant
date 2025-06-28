import os, uuid, cv2, json, joblib, time, threading, numpy as np, mediapipe as mp
from gtts import gTTS
from playsound3 import playsound
from collections import deque, Counter


# --------------------- Load Bangla Sign Dataset ---------------------
with open("/Users/mdsaibhossain/code/python/MicroProject/bangla_signs_dataset.json", "r", encoding="utf-8") as f:
    bangla_dataset = json.load(f)

def get_speech_text(pred):
    entry = bangla_dataset.get(str(pred), {})
    return entry.get("sentence") or entry.get("letter") or entry.get("example_word") or "অজানা অক্ষর"

# --------------------- Non-blocking TTS Function ---------------------
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

# --------------------- Init Model & Hand Detection ---------------------
def load_model(path):
    model = joblib.load(path)
    print(f"Loaded Model with {len(model.classes_)} labels")
    return model

def init_hand_detector():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )

def extract_landmark_features(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()

# --------------------- Main Loop ---------------------
def main():
    model_path = "/Users/mdsaibhossain/code/python/MicroProject/model/HandSign_Classifier_Alphabets_RF.pkl"
    model = load_model(model_path)
    hands = init_hand_detector()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    last_pred_time = 0
    delay_between_preds = 10  # seconds
    pred_queue = deque(maxlen=10)

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
                    pred_proba = model.predict_proba(X)[0]
                    pred_idx = np.argmax(pred_proba)
                    confidence = pred_proba[pred_idx]
                    pred = model.classes_[pred_idx]
                    print("confidence =",confidence,"\npred_idx=",pred_idx)
                    #pred = model.predict(X)[0]
                    pred_queue.append((pred,confidence))

                    # Check if enough time passed
                    if current_time - last_pred_time >= delay_between_preds:
                        common_preds = Counter([p for p, _ in pred_queue])
                        most_common_pred, count = common_preds.most_common(1)[0]
                        avg_conf = np.mean([conf for p, conf in pred_queue if p == most_common_pred])

                        if count >= 8:
                            sentence = get_speech_text(most_common_pred)
                            speak_bangla(sentence)
                            print(f"✅ Final Prediction: {most_common_pred}, Sentence: {sentence}")
                        else:
                            speak_bangla("আপনার হাতটি সঠিকভাবে দেখান।")
                            print("❌ Prediction uncertain. Asking user to correct hand.")

                        last_pred_time = current_time

                    label = bangla_dataset.get(str(pred), {}).get("letter", "Unknown")
                    cv2.putText(frame, f'Prediction: {pred}', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "Invalid landmark count!", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Bangla Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
