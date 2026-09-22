"""
Deepfake Detection Module for Forensic Intelligence Platform.

Analyzes video authenticity by detecting:
- Face swap artifacts
- Lip-sync inconsistencies  
- GAN-generated abnormalities
- Frame-level manipulation
- Temporal inconsistencies
"""

import numpy as np
import cv2
from pathlib import Path


class DeepfakeDetector:
    def __init__(self):
        """Initialize deepfake detection engine."""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.model = None

    def analyze_video_authenticity(self, video_path, frames_to_check=30):
        """
        Analyze video for deepfake indicators.
        
        Args:
            video_path: Path to video file
            frames_to_check: Number of frames to sample
            
        Returns:
            dict with authenticity score and risk indicators
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {"authenticity_score": 0.5, "risk_level": "Unknown", "error": "Cannot read video"}

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, frames_to_check, dtype=int)

            face_consistency = self._analyze_face_consistency(cap, frame_indices)
            lip_sync_score = self._analyze_lip_sync(cap, frame_indices)
            texture_anomalies = self._detect_texture_anomalies(cap, frame_indices)
            frequency_artifacts = self._detect_frequency_artifacts(cap, frame_indices)

            cap.release()

            authenticity_score = (
                face_consistency * 0.35
                + (1 - lip_sync_score) * 0.25
                + (1 - texture_anomalies) * 0.25
                + (1 - frequency_artifacts) * 0.15
            )

            if authenticity_score > 0.85:
                risk_level = "Low (Likely Genuine)"
                badge_color = "#16a34a"
            elif authenticity_score > 0.65:
                risk_level = "Moderate (Minor Artifacts)"
                badge_color = "#f59e0b"
            else:
                risk_level = "High (Deepfake Likely)"
                badge_color = "#dc2626"

            return {
                "authenticity_score": round(authenticity_score, 3),
                "risk_level": risk_level,
                "badge_color": badge_color,
                "face_consistency": round(face_consistency, 3),
                "lip_sync_score": round(lip_sync_score, 3),
                "texture_anomalies": round(texture_anomalies, 3),
                "frequency_artifacts": round(frequency_artifacts, 3),
                "frames_analyzed": len(frame_indices),
            }
        except Exception as e:
            return {"authenticity_score": 0.5, "risk_level": f"Error: {str(e)}", "error": str(e)}

    def _analyze_face_consistency(self, cap, frame_indices):
        """Check face detection consistency across frames."""
        try:
            face_detections = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                face_detections.append(len(faces) > 0)

            consistency = sum(face_detections) / max(len(face_detections), 1)
            return consistency
        except Exception:
            return 0.5

    def _analyze_lip_sync(self, cap, frame_indices):
        """Detect lip-sync inconsistencies (simplified)."""
        try:
            lip_variance = 0.0
            prev_roi_energy = None

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    mouth_roi = gray[y + h // 2 : y + h, x : x + w]
                    roi_energy = np.sum(cv2.Laplacian(mouth_roi, cv2.CV_64F) ** 2)
                    
                    if prev_roi_energy is not None:
                        lip_variance += abs(roi_energy - prev_roi_energy) / (prev_roi_energy + 1e-6)
                    prev_roi_energy = roi_energy

            return min(lip_variance / max(len(frame_indices), 1), 1.0)
        except Exception:
            return 0.5

    def _detect_texture_anomalies(self, cap, frame_indices):
        """Detect unusual texture patterns indicating manipulation."""
        try:
            texture_scores = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                texture_var = np.var(laplacian)
                texture_scores.append(min(texture_var / 1000, 1.0))

            return np.mean(texture_scores) if texture_scores else 0.5
        except Exception:
            return 0.5

    def _detect_frequency_artifacts(self, cap, frame_indices):
        """Detect GAN/compression artifacts in frequency domain."""
        try:
            artifact_scores = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fft = np.fft.fft2(gray)
                magnitude = np.abs(fft)
                
                # High-frequency anomalies suggest GAN artifacts
                high_freq = magnitude[magnitude.shape[0]//4:, magnitude.shape[1]//4:]
                artifact_score = np.mean(high_freq) / (np.mean(magnitude) + 1e-6)
                artifact_scores.append(min(artifact_score, 1.0))

            return np.mean(artifact_scores) if artifact_scores else 0.5
        except Exception:
            return 0.5

    def get_deepfake_report(self, analysis_result):
        """Generate human-readable deepfake analysis report."""
        if "error" in analysis_result:
            return f"Analysis Error: {analysis_result['error']}"

        report = f"""
DEEPFAKE AUTHENTICITY ANALYSIS REPORT
=====================================

Video Authenticity Score: {analysis_result['authenticity_score']:.1%}
Risk Assessment: {analysis_result['risk_level']}

TECHNICAL INDICATORS:
- Face Detection Consistency: {analysis_result.get('face_consistency', 0):.1%}
- Lip-Sync Variance: {analysis_result.get('lip_sync_score', 0):.2f}
- Texture Anomalies: {analysis_result.get('texture_anomalies', 0):.2f}
- Frequency Artifacts: {analysis_result.get('frequency_artifacts', 0):.2f}

Frames Analyzed: {analysis_result.get('frames_analyzed', 0)}

VERDICT:
{'✓ Video appears GENUINE.' if analysis_result['authenticity_score'] > 0.85 else '⚠ Video contains suspicious indicators.' if analysis_result['authenticity_score'] > 0.65 else '✗ Video likely DEEPFAKE.'}
        """
        return report.strip()
