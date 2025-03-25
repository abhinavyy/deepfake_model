import streamlit as st
# type: ignore
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import os

# Load the Keras model
model = load_model('deepfake_detection_model.h5')

# Set page title and icon
st.set_page_config(page_title="Deepfake Detection", page_icon="🕵️")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stFileUploader>div>div>div>div {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 20px;
    }
    .stMarkdown h1 {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and description
st.title("🕵️ Deepfake Detection")
st.markdown("Upload an image or video, and we'll detect if it's real or fake!")

# File uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to preprocess video
def preprocess_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))  # Resize frame
        frame = frame / 255.0  # Normalize pixel values
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    return frames

# Function to predict
def predict(image):
    prediction = model.predict(image)
    return "Fake" if prediction[0][0] > 0.5 else "Real"

# Display results
if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        # Process image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")
        image = preprocess_image(image)
        result = predict(image)
        st.success(f"Prediction: {result}")

    elif uploaded_file.type.startswith('video'):
        # Process video
        st.video(uploaded_file)
        st.write("Processing video...")
        video_path = os.path.join("temp", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        frames = preprocess_video(video_path)
        result = predict(frames)
        st.success(f"Prediction: {result}")  # Single prediction for the entire video
        os.remove(video_path)  # Clean up temporary file