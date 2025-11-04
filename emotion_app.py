import streamlit as st
import cv2
from deepface import DeepFace
import tempfile
import os
import numpy as np
from collections import deque
import statistics
import time

# ----------------------------#
# 🎨 Streamlit Page Setup
# ----------------------------#
st.set_page_config(page_title="Face Emotion Analyzer 😊", layout="wide")
st.title("😄 Face Emotion Analyzer")
st.markdown("Detect emotions in real-time or from uploaded images 🎭")

# ----------------------------#
# 😃 Emoji mapping
# ----------------------------#
emotion_emojis = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😄',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

# ----------------------------#
# 🎥 Real-time Detection Function
# ----------------------------#
def real_time_detection():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    stframe = st.empty()

    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check camera permissions.")
        return

    st.warning("Click **Stop** below to end real-time detection.")
    stop_button = st.button("🛑 Stop Detection")

    emotion_window = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Unable to read from webcam.")
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion'].lower()
            emotion_window.append(dominant_emotion)
            stable_emotion = statistics.mode(emotion_window)
            emoji = emotion_emojis.get(stable_emotion, '')

            # Add text overlay
            cv2.putText(frame, f"{stable_emotion.capitalize()} {emoji}", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        except Exception as e:
            print("Error:", e)

        # Convert for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

        # Break if user clicks stop
        if stop_button:
            break

        # Add a small delay to control frame rate
        time.sleep(0.1)

    cap.release()
    st.success("✅ Real-time detection stopped.")


# ----------------------------#
# 🖼️ Image Upload Detection Function
# ----------------------------#
def upload_detection(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    try:
        result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion'].capitalize()
        emoji = emotion_emojis.get(emotion.lower(), '')

        st.image(img_path, caption=f"{emotion} {emoji}", use_column_width=True)
        st.success(f"**Emotion:** {emotion} {emoji}")
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
    finally:
        os.remove(img_path)


# ----------------------------#
# ⚙️ App Options
# ----------------------------#
mode = st.radio("Choose Mode:", ["📸 Real-time Detection", "🖼️ Upload Image"])

if mode == "📸 Real-time Detection":
    st.markdown("### 🎥 Real-time Webcam Emotion Detection")
    if st.button("Start Detection"):
        real_time_detection()

elif mode == "🖼️ Upload Image":
    st.markdown("### 🖼️ Upload an Image for Emotion Analysis")
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        upload_detection(uploaded_file)

# ----------------------------#
# 👩‍💻 Footer
# ----------------------------#
st.markdown("---")
st.markdown("👩‍💻 Built with ❤️ by **Eha Ahuja And Prem**")
