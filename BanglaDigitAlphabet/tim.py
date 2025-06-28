from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
import joblib
import mediapipe as mp
from PIL import Image
import io
import websocket
import time
import json

app = Flask(__name__)

# Load model and setup MediaPipe
model_path = "hand_digit_classifier_sorted1.pkl"
model = joblib.load(model_path)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# ✅ Setup Serial Port
# Setup WebSocket client
def on_open(ws):
    print("WebSocket connection opened.")

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed:", close_status_code, close_msg)

ws = websocket.WebSocketApp(
    "ws://192.168.0.100:81",  # WebSocket server IP and port of ESP32
    on_open=on_open,
    on_error=on_error,
    on_close=on_close
)

ws.run_forever()

import threading
ws_thread = threading.Thread(target=ws.run_forever)
ws_thread.daemon = True
ws_thread.start()
time.sleep(2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign-language')
def goto():
    return render_template('translation.html')

@app.route('/learn-sign-language')
def learn():
    return render_template('learn.html')

@app.route('/learn_0-9')
def learn0_9():
    return render_template('learn0_9.html')

@app.route('/learn-bangla-alphabet')
def learnbanglaAlphabhet():
    return render_template('learn-bangla-alphabet.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form['image']
        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        frame = np.array(image)
        # cv2.imwrite("debug_frame.jpg", frame)
        print(frame)

        # Flip image to match the camera orientation
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        # print(results)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                print(f"[INFO] Features shape: {len(coords)}")
                X = np.array(coords).reshape(1, -1)
                pred = int(model.predict(X)[0])

                # Send prediction via WebSocket
                if ws.sock and ws.sock.connected:
                    ws.send(json.dumps({"digit": pred}))
                else:
                    print("[WebSocket] Not connected.")

                return jsonify({'prediction': pred})
        else:
            return jsonify({'prediction': 'No hand detected'})
    except Exception as e:
        print("[Predict Error]:", e)
        return jsonify({'prediction': 'Error occurred'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)