import numpy as np
import librosa
import joblib
import os

class AudioProcessor:
    def __init__(self, model_path='models/audio_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print("Audio model loaded successfully.")
        else:
            print(f"Warning: Audio model not found at {self.model_path}")

    def extract_features(self, file_path):
        """
        Extract MFCC (Mel-frequency cepstral coefficients) features from an audio file.
        MFCCs are excellent for speech processing and detecting stress.
        """
        try:
            # Load audio file (resample to 22050 Hz)
            y, sr = librosa.load(file_path, sr=22050)
            
            # Extract MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0) # Average across time frames
            
            return mfccs_scaled
        except Exception as e:
            print(f"Error processing audio {file_path}: {e}")
            return None

    def predict(self, file_path):
        """
        Predicts truth/lie probability based on voice stress.
        Returns: Decimal confidence of "Deception" (0.0 to 1.0)
        """
        if self.model is None:
            return 0.5 # Return neutral if no model
            
        features = self.extract_features(file_path)
        if features is None:
            return 0.5
            
        features = features.reshape(1, -1)
        
        # Predict probability
        # Class 1 is 'Lie', Class 0 is 'Truth'
        prob = self.model.predict_proba(features)[0]
        deception_score = prob[1] # Probability of being a lie
        
        return float(deception_score)
