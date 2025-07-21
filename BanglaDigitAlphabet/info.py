import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_bangla_text(image, text, position, font_path, font_size=32, color=(255, 255, 255)):
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Load a Unicode Bangla font
    font = ImageFont.truetype(font_path, font_size)

    # Draw Bangla text
    draw.text(position, text, font=font, fill=color)

    # Convert back to OpenCV image
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Example usage
cap = cv2.VideoCapture(0)
bangla_word = "বাংলা"
bangla_sentence = "আমি বাংলায় কথা বলি"
font_path = "SolaimanLipi.ttf"  # or use an absolute path to the Bangla font

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = draw_bangla_text(frame, f"Current Word: {bangla_word}", (10, 420), font_path)
    frame = draw_bangla_text(frame, f"Sentence: {bangla_sentence}", (10, 460), font_path)

    cv2.imshow("Bangla Text Display", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
