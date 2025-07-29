import os, uuid, cv2, json, joblib, time, threading, unicodedata , numpy as np, mediapipe as mp
from gtts import gTTS
from playsound3 import playsound
from collections import deque, Counter
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# --------------------- Load Bangla Sign Dataset ---------------------
with open("/Users/mdsaibhossain/code/python/BSL_Learning_Tool/bangla_signs_dataset.json", "r", encoding="utf-8") as f:
    bangla_dataset = json.load(f)

# Load optional custom word list
with open("/Users/mdsaibhossain/code/python/BSL_Learning_Tool/bangla_words", "r", encoding="utf-8") as f:
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
