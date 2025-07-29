# 🇧🇩 Bangla Sign Language Assistant 🤟

This project is a real-time Bangla Sign Language recognition tool that uses a webcam to detect hand gestures, predict the corresponding Bangla alphabet, and provide audio-visual feedback to help users learn sign language.

## 🚀 Features

- 📷 Real-time hand landmark detection using MediaPipe
- 🔤 Bangla alphabet prediction using trained ML model
- 🎧 Audio guidance using `gTTS` and `pyttsx3`
- 🌐 Web-based frontend (HTML + Jinja2)
- 🧠 Model trained with Scikit-learn + Transformers
- 🗣️ NLP support with BNLP Toolkit

## 🛠 Technologies Used

- `mediapipe`, `opencv-python`, `scikit-learn`, `transformers`
- `fastapi`, `uvicorn`, `jinja2` for web backend
- `joblib`, `gtts`, `pyttsx3`, `playsound3` for voice feedback
- `bnlp_toolkit`, `bnnumerizer`, `pyspellchecker` for Bangla text processing

## 📂 Project Structure

```bash
.
├── bangla_sign_server.py     # FastAPI backend
├── templates/index.html      # Web frontend
├── model/bangla_handsign_alphabets_classifier_rf1   # Pretrained sign classifier
├── Dataset/bangla_signs_dataset.json  # Sign metadata
├── requirements.txt
```

## 🖥️ How to Run

### Step 1: Clone and Install Dependencies
```bash
git clone https://github.com/YOUR_USERNAME/bangla-sign-language-assistant.git
cd bangla-sign-language-assistant
pip install -r requirements.txt
```

### Step 2: Start the Server
```bash
python -m uvicorn bangla_sign_server:app --reload
```

### Step 3: Open in Browser
...

## 🤝 Contributions

...

## 📜 License
...
