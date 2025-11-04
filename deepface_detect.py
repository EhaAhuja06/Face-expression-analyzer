import cv2
from deepface import DeepFace
from collections import deque
import statistics

# Emoji mapping
emotion_emojis = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😄',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

# Initialize webcam (use 0 or 1 depending on which one is available)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Error: Could not access the webcam.")
    exit()

emotion_window = deque(maxlen=10)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame.")
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion'].lower()
        emotion_window.append(dominant_emotion)
        stable_emotion = statistics.mode(emotion_window)

        emoji = emotion_emojis.get(stable_emotion, '')
        cv2.putText(frame, f"{stable_emotion.capitalize()} {emoji}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    except Exception as e:
        print("Error:", e)

    cv2.imshow('Emotion Detector 😊', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
