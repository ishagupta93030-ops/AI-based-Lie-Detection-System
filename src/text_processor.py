import os
import joblib
import numpy as np
from keras.models import load_model  # type: ignore
from keras.utils import pad_sequences  # type: ignore
import re

# Optional language detection and translation
try:
    from langdetect import detect
except Exception:
    detect = None

try:
    from googletrans import Translator
    _translator = Translator()
except Exception:
    _translator = None

class TextProcessor:
    def __init__(self, model_path='models/text_model.keras', tokenizer_path='models/tokenizer.pkl'):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
        self.max_len = 100  # Maximum sequence length for the LSTM
        self.last_detected_language = None
        self.last_translated_text = None
        self.load_artifacts()

    def load_artifacts(self):
        if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
            self.model = load_model(self.model_path)
            self.tokenizer = joblib.load(self.tokenizer_path)
            print("Text model & tokenizer loaded successfully.")
        else:
            print(f"Warning: Text model or tokenizer missing.")

    def preprocess_text(self, text):
        """
        Cleans text and converts to padded token sequences.
        """
        # If available, detect non-English text and translate to English
        try:
            lang = None
            if detect is not None:
                try:
                    lang = detect(text)
                except Exception:
                    lang = None
            self.last_detected_language = lang

            if lang is not None and lang != 'en' and _translator is not None:
                try:
                    translated = _translator.translate(text, dest='en').text
                    self.last_translated_text = translated
                    text_to_process = translated
                except Exception:
                    text_to_process = text
            else:
                text_to_process = text
        except Exception:
            text_to_process = text

        # Basic cleaning: lowercase and remove non-alphabetic chars
        cleaned = re.sub(r'[^a-zA-Z\s]', '', text_to_process).lower()

        if self.tokenizer is None:
            return None

        sequences = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        return padded

    def predict(self, text):
        """
        Predicts whether the written/spoken transcript is deceptive.
        Liars often use fewer exclusive words, more negative emotions, 
        and distance themselves with fewer 1st-person pronouns.
        """
        if self.model is None or self.tokenizer is None:
            return 0.5 # Neutral weight
            
        if not text.strip():
            return 0.5
            
        X = self.preprocess_text(text)
        if X is None:
            return 0.5
            
        # The LSTM predicts a binary classification (Truth=0, Lie=1)
        pred = self.model.predict(X, verbose=0)
        
        # Returns the probability of deception
        return float(pred[0][0])
