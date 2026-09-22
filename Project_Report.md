# AI-Based Lie Detection System: M.Tech Project Report

## Abstract
Deception detection is a critical requirement in various domains, ranging from law enforcement to corporate interviews. Traditional methods, such as polygraph tests, suffer from high subjective interpretation and are known to be unreliable. With the advent of deep learning and machine learning, automated lie detection has become an achievable goal. This project presents a novel **Multimodal AI-Based Lie Detection System** that integrates three independent modalities: Facial Expression Analysis (Video), Voice Pitch & Stress Analysis (Audio), and Linguistic Sentiment & Pattern Analysis (Text). By aggregating features across multiple modalities, the system overcomes the limitations of single-modal detection, achieving higher accuracy and robustness. The system employs Convolutional Neural Networks (CNN) for video, Support Vector Machines/Random Forests for audio features, and Long Short-Term Memory (LSTM) networks for text. 

## 1. Introduction
### 1.1 Problem Statement
Detecting whether a person is telling the truth or lying is a challenging task due to the complex physiological and psychological indicators of deception. Human judgment is notoriously flawed, often achieving accuracy barely above chance. The objective of this project is to automate deception detection using Artificial Intelligence. We aim to identify deceptive behavior by analyzing micro-expressions in the face, stress markers in the voice, and deceptive linguistic patterns in text.

### 1.2 Objectives
- To develop individual ML/DL models capable of analyzing video, audio, and text for deception cues.
- To combine these models into a multimodal architecture for a unified "Truth/Lie" prediction.
- To build a real-time, user-friendly interface for inference.
- To analyze the system's performance using standard classification metrics.

## 2. Methodology & Architecture
### 2.1 Multimodal Approach
A multimodal system utilizes multiple sensors/inputs to derive a conclusion. Our system uses Late Fusion (Decision-level fusion), where three separate models produce their respective confidence scores, which are then weighted and combined to produce the final result.
- **Audio:** Voice stress indicates cognitive load.
- **Video:** Facial micro-expressions reveal hidden emotions.
- **Text:** Liars often use fewer first-person pronouns, more negative emotion words, and fewer exclusive words.

### 2.2 Data Preprocessing
- **Video Preprocessing:** Frames are extracted from the webcam/video using OpenCV. Faces are detected using Haar Cascades or MTCNN. The facial regions are resized to 48x48 grayscale images.
- **Audio Preprocessing:** Audio files are sampled at 22050 Hz. We extract Mel-Frequency Cepstral Coefficients (MFCCs), Chroma, and Mel spectrograms using the `librosa` library.
- **Text Preprocessing:** Transcripts are lowercased, tokenized, and freed of stop-words. Sequences are padded to a fixed length for neural network processing.

### 2.3 Model Development
1. **Facial Expression Model (CNN):**
   - Architecture: Multiple Conv2D layers followed by MaxPooling, Dropout (to prevent overfitting), and Dense layers. 
   - Output: Softmax layer classifying basic emotions (e.g., anxiety, fear, neutral) mapped to deception probabilities.
2. **Voice Stress Model (Random Forest / SVM):**
   - A traditional machine learning approach works exceptionally well on structured 1D audio features (MFCCs). We use an SVM classifier with an RBF kernel or a Random Forest to classify the stress levels in the voice.
3. **Text Analysis Model (LSTM):**
   - Sequences of words are passed into an Embedding layer, followed by an LSTM (Long Short-Term Memory) layer. LSTMs are perfect for capturing the sequential context of deceptive language.

## 3. Recommended Datasets
To train these models at an academic standard, the following publicly available datasets are recommended:
1. **Face/Video:** *Miami University Deception and Truthfulness Database (MU3D)* or *Real-Life Trial Data (RLTD)*.
2. **Audio/Voice:** *RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)* can be repurposed for stress detection, or *Deceptive Speech Database*.
3. **Text:** *Real-Life Deception Detection* text transcripts or *Deceptive Opinion Spam Corpus* (Ott et al.).

*(Note: The provided codebase includes a generic training structure that can be easily mapped to any of these datasets.)*

## 4. Implementation Details
The system is built in Python. 
- **Libraries:** TensorFlow/Keras (CNN, LSTM), Scikit-Learn (RF, SVM), OpenCV (Video processing), Librosa (Audio processing), NLTK (Text).
- **Interface:** Streamlit is used to create a modern web-based prediction dashboard.
- **Real-Time Execution:** The OpenCV `VideoCapture` streams frames directly to the CNN, while text and audio are processed asynchronously.

## 5. Results and Visualization
During the training phase, the datasets are split into 80% training and 20% testing sets. 
Performance is visualized using Matplotlib to plot:
- **Accuracy and Loss curves** for the deep learning models over epochs.
- **Confusion Matrix** to track True Positives (correctly identified lies) and False Positives.
- Metrics calculated include Accuracy, Precision, Recall, and F1-Score. Multimodal fusion typically shows a 10-15% increase in F1-score over single-modality models.

## 6. Applications
- **Law Enforcement & Interrogation:** Providing an objective baseline of stress and deception markers during questioning.
- **Corporate HR:** Analyzing anomalies or extreme stress levels during remote interviews.
- **Financial Fraud Detection:** Analyzing claims and statements in insurance or banking.
- **Security:** Airport screening and border control pre-screening.

## 7. Future Scope
- **Transformer Models:** Upgrading the LSTM to BERT/RoBERTa for superior sentiment and transcript analysis.
- **Early Fusion:** Integrating feature vectors from audio and video immediately at the neural network layer rather than just averaging final confidence scores.
- **Thermal Imaging:** Adding a thermal camera input to track facial blood flow variations in real-time.

## 8. Conclusion
The AI-Based Lie Detection System provides a robust, scientifically grounded framework for deception detection. By aggregating facial, vocal, and linguistic features, it successfully mitigates the weaknesses of legacy systems like polygraphs. The application provides an M.Tech-level implementation of multimodal machine learning, ready for real-time deployment and future academic expansion.
