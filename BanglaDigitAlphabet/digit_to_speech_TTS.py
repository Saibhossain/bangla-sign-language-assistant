import cv2
import numpy as np
import time
import joblib
import pyttsx3
import mediapipe as mp

# --------------------- Initialization ---------------------
def load_model(path):
    return joblib.load(path)

def init_hand_detector():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )

def init_tts_engine(rate=150):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    return engine

# --------------------- Helper Function ---------------------
def extract_hand_landmarks(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten().reshape(1, -1)

# --------------------- Main ---------------------
def main():
    model_path = "/Users/mdsaibhossain/code/python/MicroProject/model/hand_digit_classifier_sorted1.pkl"
    model = load_model(model_path)
    hands = init_hand_detector()
    engine = init_tts_engine()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    last_pred = None
    last_pred_time = 0
    delay_between_preds = 3  # seconds

    print("Starting... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                features = extract_hand_landmarks(hand_landmarks)
                pred = model.predict(features)[0]

                # Only speak if it's a new digit and 1s has passed
                if pred != last_pred and (current_time - last_pred_time) >= delay_between_preds:
                    last_pred = pred
                    last_pred_time = current_time

                    print(f"Predicted Digit: {pred}")
                    engine.say(str(pred))
                    engine.runAndWait()

                # Show prediction
                cv2.putText(frame, f'Predicted: {pred}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Show the video feed
        cv2.imshow("Hand Digit Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    engine.stop()

if __name__ == "__main__":
    main()
