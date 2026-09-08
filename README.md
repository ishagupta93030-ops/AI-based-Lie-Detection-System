3# AI-Based Multimodal Lie Detection System

## Overview
This is an M.Tech-level complete AI project that utilizes a multimodal approach to detect deceptive behavior. It analyzes Facial Expressions, Voice Stress, and Text Sentiment using Deep Learning and Machine Learning techniques.

## Features
- **Facial Expression Analysis:** Uses OpenCV for real-time video capture and a Convolutional Neural Network (CNN) to detect micro-expressions.
- **Voice Stress Analysis:** Uses `librosa` to extract MFCC features and a Random Forest/SVM model to classify stress levels indicative of deception.
- **Text Sentiment Analysis:** Uses an LSTM network to identify deceptive linguistic patterns.
- **Multimodal Fusion:** Aggregates confidence scores from all models for an accurate meta-prediction.
- **Demo Case Support:** Built-in sample audio and transcript mode to instantly showcase the system.
- **Explainable UI:** Clear model health status, modality scores, and prediction reasoning in the dashboard.
- **Real-Time Streamlit Interface:** User-friendly GUI for predicting files and viewing real-time data.

## Setup Instructions

1. **Install Dependencies:**
   Ensure you have Python 3.9+ installed. Run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Models:**
   A beginner-friendly script has been provided to generate representative sample data and train the three models (so you don't need gigabytes of datasets to see it working).
   ```bash
   python train_models.py
   ```
   *This will save the trained models into the `models/` directory and plot an accuracy graph.*

3. **Run the Streamlit Interface:**
   ```bash
   streamlit run app.py
   ```
   *Open the provided localhost link in your browser to interact with the system.*

4. **If you do not have system-wide FFmpeg installed:**
   - A portable FFmpeg copy is included under `tools/ffmpeg`.
   - Launch the app with the local FFmpeg binaries using:
     ```powershell
     tools\start_streamlit_with_ffmpeg.bat
     ```
   - `app.py` will also auto-detect `tools/ffmpeg` and configure `pydub` automatically.

## Usage Notes
- Text input supports non-English languages such as Hindi. The system will detect and translate the text to English before analysis.
- Video files can be uploaded directly if webcam capture is unavailable, which is especially useful on Windows systems.
- Built-in demo mode lets you load a sample deceptive transcript and audio with one click.
- If a modality is not provided, the system still runs using the remaining available data.

## Technologies Used
- **Deep Learning:** TensorFlow, Keras (CNN, LSTM)
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Audio Processing:** Librosa
- **Computer Vision:** OpenCV
- **GUI:** Streamlit
