import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('my_model.keras') 

def process_video(video_file):
    # Open the uploaded video file
    vid = cv2.VideoCapture(video_file)
    
    # List to hold frame data
    frames = []
    
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        # Resize frame to fit ResNet input
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    
    vid.release()
    
    # Convert list to numpy array and preprocess it for model prediction
    frames = np.array(frames)
    
    # Preprocess frames for ResNet50
    frames = frames / 255.0  # Normalize pixel values between 0 and 1
    
    # Make prediction
    prediction = model.predict(frames)  # Use your actual prediction logic based on your model
    
    return prediction

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #ff6347;
        }
        .subheader {
            font-size: 20px;
            color: #4CAF50;
            text-align: center;
        }
        .upload-btn {
            text-align: center;
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 15px 32px;
            font-size: 16px;
            cursor: pointer;
        }
        .upload-btn:hover {
            background-color: #45a049;
        }
        .result {
            font-size: 25px;
            font-weight: bold;
            color: #ff4500;
            text-align: center;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f7f7f7;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<p class="title">Deepfake Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload a video to classify it as Real or Fake.</p>', unsafe_allow_html=True)

# Create a container for the UI elements
with st.container():
    st.markdown('<button class="upload-btn">Upload Video</button>', unsafe_allow_html=True)
    
    # File uploader for video
    video_file = st.file_uploader("Choose a video file (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

    if video_file is not None:
        # Display uploaded video in the app
        st.video(video_file)
        
        # Processing and prediction
        st.markdown("<p class='subheader'>Processing video...</p>", unsafe_allow_html=True)
        prediction = process_video(video_file)
        
        # Display prediction result
        if prediction > 0.5:  # Assuming output is a probability, adjust based on your model
            st.markdown("<p class='result'>Prediction: Fake</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='result'>Prediction: Real</p>", unsafe_allow_html=True)
