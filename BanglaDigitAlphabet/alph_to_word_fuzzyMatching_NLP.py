import os, uuid, cv2, json, joblib, time, threading, unicodedata
import numpy as np
import mediapipe as mp
from gtts import gTTS
from playsound3 import playsound
from collections import deque, Counter
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# --------------------- Load Bangla Sign Dataset ---------------------
with open("/Users/mdsaibhossain/code/python/MicroProject/bangla_signs_dataset.json", "r", encoding="utf-8") as f:
    bangla_dataset = json.load(f)

# Load optional custom word list
with open("/Users/mdsaibhossain/code/python/MicroProject/bangla_words", "r", encoding="utf-8") as f:
    valid_words = set(line.strip() for line in f)

# --------------------- Speech ---------------------
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

# --------------------- Model & MediaPipe Init ---------------------
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

# --------------------- Word Assembler ---------------------
def normalize(text):
    return unicodedata.normalize('NFC', text.strip())

class WordBuilder:
    def __init__(self):
        self.letter_buffer = []
        self.words = []
        self.max_len = 20

    def add_letter(self, letter):
        self.letter_buffer.append(letter)
        if len(self.letter_buffer) > self.max_len:
            self.letter_buffer.pop(0)

    def commit_word(self):
        word = ''.join(self.letter_buffer).strip()
        self.letter_buffer.clear()
        return word

    def get_sentence(self):
        return ' '.join(self.words)

    def try_add_word(self, word, valid_words):
        word = normalize(word)
        normalized_words = [normalize(w) for w in valid_words]

        if word in normalized_words:
            self.words.append(word)
            return word, True

        closest = get_close_matches(word, normalized_words, n=1, cutoff=0.6)
        if closest:
            print(f"[DEBUG] Closest match for '{word}': {closest[0]}")
            self.words.append(closest[0])
            return closest[0], True

        print(f"[DEBUG] No match found for: {word}")
        return word, False


# --------------------- Main ---------------------
def main():
    model_path = "/Users/mdsaibhossain/code/python/MicroProject/model/HandSign_Classifier_Alphabets_RF.pkl"
    model = load_model(model_path)
    hands = init_hand_detector()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    last_pred_time = 0
    last_word_time = 0
    word_gap_timeout = 7  # seconds to consider word break
    delay_between_preds = 7
    pred_queue = deque(maxlen=8)
    word_builder = WordBuilder()

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

                    raw_pred = str(model.classes_[pred_idx])
                    letter_pred = bangla_dataset.get(raw_pred, {}).get("letter", raw_pred)

                    pred_queue.append((letter_pred, confidence))

                    if current_time - last_pred_time >= delay_between_preds:
                        common_preds = Counter([p for p, _ in pred_queue])
                        most_common_letter, count = common_preds.most_common(1)[0]

                        if count >= 6:
                            word_builder.add_letter(most_common_letter)
                            print(f"➕ Letter added: {most_common_letter}")
                            last_word_time = current_time
                        else:
                            print("🟡 Low confidence, skipping letter.")

                        last_pred_time = current_time

        # Check for word boundary (pause in signing)
        if current_time - last_word_time >= word_gap_timeout and word_builder.letter_buffer:
            word = word_builder.commit_word()
            matched_word, success = word_builder.try_add_word(word, valid_words)
            if success:
                print(f"✅ Final Word: {matched_word}")
                speak_bangla(matched_word)
            else:
                print(f"❌ Invalid word: {matched_word}")
                speak_bangla("অবৈধ শব্দ")

        # Overlay current letter buffer + sentence
        current_word = ''.join(word_builder.letter_buffer)
        sentence = word_builder.get_sentence()

        cv2.putText(frame, f'Current Word: {current_word}', (10, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Sentence: {sentence}', (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Bangla Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
