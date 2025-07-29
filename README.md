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
├── model/trained_model.pkl   # Pretrained sign classifier
├── Dataset/hand_feedback_data.json  # Sign metadata
├── requirements.txt
