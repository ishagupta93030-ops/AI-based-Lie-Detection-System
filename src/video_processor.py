import cv2
import numpy as np
import os
from keras.models import load_model  # type: ignore

class VideoProcessor:
    def __init__(self, model_path='models/video_model.keras'):
        self.model_path = model_path
        self.model = None
        
        # We use OpenCV's built-in Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print("Video/Face model loaded successfully.")
        else:
            print(f"Warning: Video model not found at {self.model_path}")

    def capture_and_predict(self, duration_frames=30):
        """
        Captures a short sequence of frames from the webcam, extracts the face,
        and predicts deception probability based on micro-expressions.
        Returns the average deception probability.
        """
        if self.model is None:
            return 0.5 # Default neutral

        cap = cv2.VideoCapture(0)
        predictions = []
        
        print(f"Capturing {duration_frames} frames from webcam...")

        for _ in range(duration_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Extract the Region of Interest (ROI) containing the face
                roi_gray = gray[y:y+h, x:x+w]
                # Resize to the input size required by the CNN (e.g., 48x48)
                roi_resized = cv2.resize(roi_gray, (48, 48))
                roi_normalized = roi_resized / 255.0
                roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))
                
                # Predict deception probability based on facial expression
                # Output represents [prob_truth, prob_lie]
                pred = self.model.predict(roi_reshaped, verbose=0)
                deception_prob = float(pred[0][1])
                predictions.append(deception_prob)
                
                # Draw a rectangle around the face for UI feedback
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
            # For live webcam we show the camera output
            cv2.imshow('Facial Analysis - Lie Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        if len(predictions) == 0:
            print("No faces detected.")
            return 0.5
            
        return np.mean(predictions)

    def predict_from_file(self, file_path, max_frames=100):
        """
        Process a video file and return the average deception probability.
        """
        if self.model is None:
            return 0.5

        cap = cv2.VideoCapture(file_path)
        predictions = []
        frames_read = 0

        while frames_read < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                try:
                    roi_resized = cv2.resize(roi_gray, (48, 48))
                except Exception:
                    continue
                roi_normalized = roi_resized / 255.0
                roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

                pred = self.model.predict(roi_reshaped, verbose=0)
                deception_prob = float(pred[0][1])
                predictions.append(deception_prob)

            frames_read += 1

        cap.release()

        if len(predictions) == 0:
            return 0.5

        return np.mean(predictions)
