import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np

st.set_page_config(page_title="Emotion Detector", page_icon="😊")

st.title("🎭 Real-time Emotion Detection")
st.markdown("This app uses DeepFace to detect your emotion in real-time using your webcam.")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Failed to access webcam.")
        break

    # Convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        cv2.putText(frame, f'Emotion: {emotion}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    except:
        cv2.putText(frame, 'No face detected', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)

    FRAME_WINDOW.image(frame)

camera.release()
