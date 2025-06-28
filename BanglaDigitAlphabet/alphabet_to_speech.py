import os
import uuid
import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
import threading
from gtts import gTTS
from playsound3 import playsound

# --------------------- Bangla Sign Mapping ---------------------
bangla_signs = {
    0: "অ/আ", 1: "আ", 2: "ই/ঈ", 3: "উ/ঊ", 4: "র/ঋ/ড়/ঢ়", 5: "এ", 6: "ঐ",
    7: "ও", 8: "ঔ", 9: "ক", 10: "খ/ক্ষ", 11: "গ", 12: "ঘ", 13: "ঙ",
    14: "চ", 15: "ছ", 16: "জ/ঝ", 17: "ঞ", 18: "ট", 19: "ঠ", 20: "ড",
    21: "ঢ", 22: "ণ/ন", 23: "ত/থ", 24: "দ", 25: "ধ", 26: "ন", 27: "প",
    28: "ফ", 29: "ব"
}

# --------------------- Non-blocking Speaking ---------------------
def speak_bangla(text):
    def speak_thread(text):
        try:
            tts = gTTS(text=text, lang='bn')
            filename = f"temp_{uuid.uuid4().hex}.mp3"
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print(f"Error speaking Bangla: {e}")

    threading.Thread(target=speak_thread, args=(text,), daemon=True).start()

# --------------------- Initialization ---------------------
def load_model(path):
    model = joblib.load(path)
    print(f"Total labels: {len(model.classes_)}")
    print(f"Labels: {model.classes_}")
    return model

def init_hand_detector():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )

def extract_landmark_features(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()

# --------------------- Main ---------------------
def main():
    model_path = "/Users/mdsaibhossain/code/python/MicroProject/model/HandSign_Classifier_Alphabets_RF.pkl"
    model = load_model(model_path)
    hands = init_hand_detector()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    print("Press 'q' to exit.")

    last_pred = None
    last_pred_time = 0
    delay_between_preds = 3  # seconds

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
                    label = bangla_signs.get(pred, "Unknown")

                    # Speak only if enough time has passed since last speech
                    if pred != last_pred and (current_time - last_pred_time) >= delay_between_preds:
                        print(f"Speaking: {label}")
                        speak_bangla(label)
                        last_pred = pred
                        last_pred_time = current_time

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
