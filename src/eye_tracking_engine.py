"""
Eye Tracking Engine for Forensic Analysis.

Detects and analyzes:
- Blink frequency and patterns
- Gaze direction
- Eye contact stability
- Gaze aversion patterns
- Fixation duration
- Pupil dilation
"""

import numpy as np
import cv2
from collections import deque


class EyeTrackingEngine:
    def __init__(self):
        """Initialize eye tracking engine."""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.blink_history = deque(maxlen=100)
        self.gaze_history = deque(maxlen=100)
        
    def analyze_eye_behavior(self, video_path, frames_to_analyze=60):
        """
        Analyze eye behavior across video frames.
        
        Args:
            video_path: Path to video file
            frames_to_analyze: Number of frames to process
            
        Returns:
            dict with eye tracking metrics
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return self._error_result("Cannot open video")

            blink_count = 0
            gaze_left_count = 0
            gaze_right_count = 0
            gaze_center_count = 0
            eye_closure_duration = 0
            avg_pupil_size = []
            gaze_aversion_count = 0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_step = max(1, total_frames // frames_to_analyze)

            for frame_idx in range(0, total_frames, frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    roi_gray = gray[y:y+h, x:x+w]
                    
                    # Detect eyes
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)
                    if len(eyes) >= 2:
                        # Analyze eye openness (simplified)
                        eye_areas = [e[2] * e[3] for e in eyes[:2]]
                        avg_eye_area = np.mean(eye_areas)
                        
                        if avg_eye_area < 50:
                            blink_count += 1
                        else:
                            # Analyze gaze direction
                            eye1_x = eyes[0][0] + eyes[0][2] // 2
                            roi_width = roi_gray.shape[1]
                            
                            if eye1_x < roi_width * 0.35:
                                gaze_left_count += 1
                            elif eye1_x > roi_width * 0.65:
                                gaze_right_count += 1
                            else:
                                gaze_center_count += 1
                            
                            # Check for gaze aversion (looking away)
                            if eye1_x < roi_width * 0.2 or eye1_x > roi_width * 0.8:
                                gaze_aversion_count += 1
                        
                        avg_pupil_size.extend(eye_areas)

            cap.release()

            total_eye_detections = gaze_left_count + gaze_right_count + gaze_center_count + blink_count
            if total_eye_detections == 0:
                total_eye_detections = 1

            # Calculate metrics
            blink_frequency = (blink_count / max(frames_to_analyze, 1)) * 60  # Blinks per minute equivalent
            eye_contact_stability = (gaze_center_count / total_eye_detections) if total_eye_detections > 0 else 0
            gaze_aversion_percentage = (gaze_aversion_count / total_eye_detections * 100) if total_eye_detections > 0 else 0
            avg_pupil_diameter = np.mean(avg_pupil_size) if avg_pupil_size else 0

            # Interpret results
            if blink_frequency > 25:
                blink_assessment = "Elevated (Stress indicator)"
                blink_color = "#dc2626"
            elif blink_frequency > 15:
                blink_assessment = "Normal"
                blink_color = "#16a34a"
            else:
                blink_assessment = "Low (Possible suppression)"
                blink_color = "#f59e0b"

            if eye_contact_stability > 0.6:
                contact_assessment = "Strong (Confident)"
                contact_color = "#16a34a"
            elif eye_contact_stability > 0.4:
                contact_assessment = "Moderate (Uncertain)"
                contact_color = "#f59e0b"
            else:
                contact_assessment = "Weak (Evasive)"
                contact_color = "#dc2626"

            gaze_risk_score = gaze_aversion_percentage / 100

            return {
                "blink_frequency": round(blink_frequency, 2),
                "blink_assessment": blink_assessment,
                "blink_color": blink_color,
                "eye_contact_stability": round(eye_contact_stability, 3),
                "contact_assessment": contact_assessment,
                "contact_color": contact_color,
                "gaze_aversion_percentage": round(gaze_aversion_percentage, 1),
                "gaze_aversion_count": gaze_aversion_count,
                "avg_pupil_size": round(avg_pupil_diameter, 2),
                "gaze_risk_score": round(gaze_risk_score, 3),
                "frames_analyzed": frames_to_analyze,
                "eyes_detected": len(avg_pupil_size),
            }
        except Exception as e:
            return self._error_result(str(e))

    def _error_result(self, error_msg):
        """Return standardized error result."""
        return {
            "blink_frequency": 0,
            "eye_contact_stability": 0.5,
            "gaze_aversion_percentage": 0,
            "gaze_risk_score": 0.5,
            "error": error_msg,
        }

    def get_eye_tracking_report(self, analysis):
        """Generate human-readable eye tracking report."""
        if "error" in analysis:
            return f"Eye Tracking Error: {analysis['error']}"

        report = f"""
EYE TRACKING & BEHAVIORAL ANALYSIS REPORT
==========================================

BLINK ANALYSIS:
- Blink Frequency: {analysis['blink_frequency']} blinks/min equivalent
- Assessment: {analysis.get('blink_assessment', 'N/A')}
- Interpretation: {'Elevated blinking suggests stress or cognitive load.' if analysis['blink_frequency'] > 20 else 'Normal blinking pattern.'}

EYE CONTACT STABILITY:
- Stability Score: {analysis['eye_contact_stability']:.1%}
- Assessment: {analysis.get('contact_assessment', 'N/A')}
- Gaze Aversion: {analysis['gaze_aversion_percentage']:.1f}% of time

PUPIL METRICS:
- Average Pupil Size: {analysis['avg_pupil_size']} pixels
- Gaze Risk Score: {analysis['gaze_risk_score']:.1%}

FORENSIC INTERPRETATION:
{'✓ Eye behavior consistent with confident testimony.' if analysis['eye_contact_stability'] > 0.6 else '⚠ Mixed eye contact patterns detected.' if analysis['eye_contact_stability'] > 0.4 else '✗ Significant gaze aversion and eye contact avoidance observed.'}

Frames Analyzed: {analysis.get('frames_analyzed', 0)}
Eyes Detected: {analysis.get('eyes_detected', 0)}
        """
        return report.strip()

    def get_eye_risk_score(self, analysis):
        """Calculate composite eye-based deception risk score."""
        try:
            # High blink frequency increases risk
            blink_risk = min(analysis.get('blink_frequency', 20) / 40, 1.0)
            
            # Low eye contact increases risk
            contact_risk = 1 - analysis.get('eye_contact_stability', 0.5)
            
            # High gaze aversion increases risk
            aversion_risk = analysis.get('gaze_risk_score', 0.5)
            
            composite_score = (blink_risk * 0.3 + contact_risk * 0.4 + aversion_risk * 0.3)
            return round(min(composite_score, 1.0), 3)
        except Exception:
            return 0.5
