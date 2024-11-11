# Deepfake Detection using LSTM and ResNet

## Overview
In a world where deepfake videos are becoming increasingly sophisticated, ensuring the authenticity of visual content is crucial. This project tackles the problem of **deepfake detection** by leveraging state-of-the-art deep learning techniques. Using a combination of **ResNet50** (a pre-trained Convolutional Neural Network) and **Long Short-Term Memory (LSTM)** layers, the model is designed to analyze video frames, capturing both **spatial** and **temporal** features to determine whether a video is real or fake.

The project includes a user-friendly **Streamlit app** where users can upload videos and immediately get predictions on their authenticity. The app processes each uploaded video and uses the trained model to classify it as either **real** or **fake**.

### Key Features:
- **Deepfake Detection**: The app uses a trained deep learning model to classify video content as real or fake, providing immediate results.
- **User-Friendly Interface**: Built with **Streamlit**, the app offers an easy-to-use UI where users can simply upload a video file and get predictions.
- **Advanced Model Architecture**:
  - **ResNet50**: A powerful pre-trained CNN used for feature extraction from video frames.
  - **LSTM Layers**: Temporal sequence modeling to detect subtle fake alterations that might span across multiple frames.
  
### Model Architecture:
The core of the deepfake detection system relies on:
1. **ResNet50**: Pre-trained convolutional neural network that extracts spatial features from individual video frames.
2. **LSTM**: Captures the temporal relationships between frames, ensuring that the model understands how frames interact over time, which is key in detecting manipulated content.

### Dataset
The model was trained on the **Deepfake Detection Challenge** dataset provided by **Kaggle**, which includes:
- **323 fake videos**
- **77 real videos**

### Model Performance:
The model achieves an accuracy of approximately **81%** on the validation dataset, effectively distinguishing between real and fake content.

### How It Works:
1. **Video Upload**: Users can upload video files in MP4, MOV, or AVI formats through the app.
2. **Frame Extraction**: The app reads the video frames, resizes them for model compatibility, and processes them for predictions.
3. **Prediction**: The trained model classifies the video as either "Real" or "Fake" based on the processed frames.
4. **Result Display**: After processing, the result is displayed in an easy-to-read format.

### Installation and Usage:

#### 1. Clone the repository:
```bash
git clone https://github.com/abhinavyy/deepfake_model.git
cd deepfake_detection
```

#### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### 3. Download the Dataset:
Make sure you have joined the **Deepfake Detection Challenge** on **Kaggle**. You can download the dataset using the following command:
```bash
kaggle competitions download -c deepfake-detection-challenge
```

#### 4. Run the Streamlit App:
Run the Streamlit app locally with the following command:
```bash
streamlit run app.py
```

Access the app at **http://localhost:8501**, where you can upload videos and view predictions.

### Technologies Used:
- **TensorFlow** and **Keras**: For building, training, and making predictions using deep learning models.
- **ResNet50**: Pre-trained CNN for feature extraction.
- **LSTM**: For temporal sequence analysis to capture fake alterations over multiple frames.
- **Streamlit**: To create the user interface for video uploads and results display.
- **OpenCV**: For video frame extraction and preprocessing.

### Sample Output:
When a video is uploaded, the app will display a message indicating whether the video is **Real** or **Fake** based on the model's prediction. Here’s how the output might look:

- **Prediction: Fake**
- **Prediction: Real**

### UI Enhancements:
The **Streamlit app** includes the following features for an enhanced user experience:
- A **centered title** for the app with bold colors to capture attention.
- A **custom video upload button** with hover effects for better interactivity.
- **Results displayed in bold, colored text**, making it easy to distinguish the output.

### Future Enhancements:
- **Model Optimization**: You can experiment with other models such as 3D CNNs or more advanced LSTM architectures to improve performance.
- **Data Augmentation**: Implement techniques like frame augmentation or balanced sampling to handle class imbalance more effectively.
- **Real-time Video Processing**: Extend the app to support real-time video streams, enabling live deepfake detection in videos.

### Contributing:
Feel free to fork the repository, suggest improvements, or open issues. Contributions to improve the model or the app are welcome!

---
