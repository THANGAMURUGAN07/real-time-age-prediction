import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image

# Constants
IMAGE_SIZE = (200, 200)
AGE_LABELS = ["0-2", "3-12", "13-19", "20-30", "31-40", "41-50", "51-60", "61-70", "71-80", "80+"]

# Load pre-trained model
model = load_model('age_model.h5')

# Streamlit UI
st.set_page_config(page_title="Real-Time Age Prediction", layout="centered")
st.title("📷 Real-Time Age Prediction")
start_camera = st.checkbox("Start Webcam")

# Preprocessing function
def preprocess_image(image):
    image = cv2.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image

# Webcam interface
if start_camera:
    video_capture = cv2.VideoCapture(0)
    stframe = st.empty()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while start_camera:
        ret, frame = video_capture.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            preprocessed = preprocess_image(face)
            preprocessed = np.expand_dims(preprocessed, axis=0)

            pred = model.predict(preprocessed)
            age_category = np.argmax(pred)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {AGE_LABELS[age_category]}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    video_capture.release()
