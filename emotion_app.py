import os
import streamlit as st
from deepface import DeepFace
import tempfile
import numpy as np
import statistics
from collections import deque

# ----------------------------#
# 🌍 Environment Setup
# ----------------------------#
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

try:
    import cv2
    opencv_available = True
except Exception:
    opencv_available = False

# ----------------------------#
# 🎨 Streamlit Page Setup
# ----------------------------#
st.set_page_config(page_title="Face Emotion Analyzer 😊", layout="wide")
st.title("😄 Face Emotion Analyzer")
st.markdown("Detect emotions in real-time (local only) or from uploaded images 🎭")

# ----------------------------#
# 😃 Emoji mapping
# ----------------------------#
emotion_emojis = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😄', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

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
# 🎥 Real-time Detection (Only if OpenCV works)
# ----------------------------#
def real_time_detection():
    if not opencv_available:
        st.error("⚠️ Real-time detection is unavailable on Streamlit Cloud.")
        return

    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    st.warning("Click **Stop** below to end real-time detection.")
    stop_button = st.button("🛑 Stop Detection")

    emotion_window = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion'].lower()
            emotion_window.append(dominant_emotion)
            stable_emotion = statistics.mode(emotion_window)
            emoji = emotion_emojis.get(stable_emotion, '')
            cv2.putText(frame, f"{stable_emotion.capitalize()} {emoji}", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        except:
            pass

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")
        if stop_button:
            break

    cap.release()
    st.success("✅ Real-time detection stopped.")


# ----------------------------#
# ⚙️ App Options
# ----------------------------#
if opencv_available:
    mode = st.radio("Choose Mode:", ["📸 Real-time Detection", "🖼️ Upload Image"])
else:
    st.info("Webcam not supported on Streamlit Cloud. Upload mode only.")
    mode = "🖼️ Upload Image"

if mode == "📸 Real-time Detection":
    if st.button("Start Detection"):
        real_time_detection()
elif mode == "🖼️ Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        upload_detection(uploaded_file)

# ----------------------------#
# 👩‍💻 Footer
# ----------------------------#
st.markdown("---")
st.markdown("👩‍💻 Built with ❤️ by **Eha Ahuja and Prem**")
