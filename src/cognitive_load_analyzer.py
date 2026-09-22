"""
Cognitive Load Analysis Engine.

Estimates mental effort and cognitive strain using:
- Speech pauses and hesitations
- Response delay analysis
- Voice instability
- Filler word detection
- Speaking rate anomalies
- Stress indicators
"""

import numpy as np
import librosa
from pathlib import Path


class CognitiveLoadAnalyzer:
    def __init__(self):
        """Initialize cognitive load analyzer."""
        self.filler_words = {
            "en": ["um", "uh", "like", "you know", "basically", "actually", "i mean", "sort of", "kind of", "well", "i think", "i guess"],
            "hi": ["haan", "na", "yaar", "matlab", "acha", "bas", "kya", "bilkul", "theek", "toh"],
        }
        self.typical_speech_rate = 150  # words per minute

    def analyze_cognitive_load(self, audio_path, transcript="", language="en"):
        """
        Analyze cognitive load from audio and transcript.
        
        Args:
            audio_path: Path to audio file
            transcript: Text transcript of speech
            language: Language code ('en' or 'hi')
            
        Returns:
            dict with cognitive load metrics
        """
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=22050)
            
            # Compute multiple metrics
            pause_analysis = self._analyze_pauses(y, sr)
            voice_stability = self._analyze_voice_stability(y, sr)
            hesitation_score = self._detect_hesitations(transcript, language)
            speech_rate = self._estimate_speech_rate(y, sr, transcript)
            stress_indicators = self._detect_stress_indicators(y, sr)
            
            # Compute composite cognitive load score
            cognitive_load = (
                pause_analysis["pause_ratio"] * 0.25 +
                (1 - voice_stability) * 0.25 +
                hesitation_score * 0.25 +
                stress_indicators["stress_level"] * 0.25
            )
            
            # Classify mental stress level
            if cognitive_load > 0.70:
                mental_stress = "High (Critical stress detected)"
                stress_color = "#dc2626"
            elif cognitive_load > 0.45:
                mental_stress = "Moderate (Some stress indicators)"
                stress_color = "#f59e0b"
            else:
                mental_stress = "Low (Relaxed demeanor)"
                stress_color = "#16a34a"
            
            return {
                "cognitive_load_score": round(cognitive_load, 3),
                "mental_stress_level": mental_stress,
                "stress_color": stress_color,
                "pause_analysis": pause_analysis,
                "voice_stability": round(voice_stability, 3),
                "hesitation_score": round(hesitation_score, 3),
                "speech_rate": round(speech_rate, 1),
                "stress_indicators": stress_indicators,
                "duration_seconds": round(len(y) / sr, 2),
            }
        except Exception as e:
            return {
                "error": str(e),
                "cognitive_load_score": 0.5,
                "mental_stress_level": "Error analyzing audio",
            }

    def _analyze_pauses(self, y, sr):
        """Detect and analyze speech pauses."""
        try:
            # Compute short-time energy
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            energy = librosa.power_to_db(S, ref=np.max)
            energy_per_frame = np.mean(energy, axis=0)
            
            # Threshold for silence
            silence_threshold = np.mean(energy_per_frame) - np.std(energy_per_frame)
            silent_frames = energy_per_frame < silence_threshold
            
            # Convert frames to time
            pause_duration = np.sum(silent_frames) * len(y) / (sr * len(energy_per_frame))
            total_duration = len(y) / sr
            pause_ratio = pause_duration / max(total_duration, 0.1)
            
            num_pauses = np.sum(np.diff(silent_frames.astype(int)) == 1)
            avg_pause_length = pause_duration / max(num_pauses, 1)
            
            return {
                "pause_ratio": round(min(pause_ratio, 1.0), 3),
                "total_pause_duration": round(pause_duration, 2),
                "num_pauses": int(num_pauses),
                "avg_pause_length": round(avg_pause_length, 2),
            }
        except Exception:
            return {
                "pause_ratio": 0.2,
                "total_pause_duration": 0,
                "num_pauses": 0,
                "avg_pause_length": 0,
            }

    def _analyze_voice_stability(self, y, sr):
        """Measure voice stability (spectral variance)."""
        try:
            # Extract MFCC (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Calculate frame-to-frame variance
            mfcc_delta = np.diff(mfcc, axis=1)
            stability = 1 - np.mean(np.abs(mfcc_delta)) / 10  # Normalize
            stability = np.clip(stability, 0, 1)
            
            return stability
        except Exception:
            return 0.5

    def _detect_hesitations(self, transcript, language="en"):
        """Detect filler words and hesitations in transcript."""
        try:
            if not transcript:
                return 0.0
            
            transcript_lower = transcript.lower()
            filler_words = self.filler_words.get(language, self.filler_words["en"])
            
            hesitation_count = 0
            for filler in filler_words:
                hesitation_count += transcript_lower.count(filler)
            
            word_count = len(transcript.split())
            hesitation_ratio = hesitation_count / max(word_count, 1)
            
            return min(hesitation_ratio, 1.0)
        except Exception:
            return 0.0

    def _estimate_speech_rate(self, y, sr, transcript):
        """Estimate words per minute speech rate."""
        try:
            total_duration = len(y) / sr
            word_count = len(transcript.split())
            
            wpm = (word_count / max(total_duration / 60, 0.1))
            
            # Deviation from typical rate indicates stress
            rate_deviation = abs(wpm - self.typical_speech_rate) / self.typical_speech_rate
            
            return wpm
        except Exception:
            return self.typical_speech_rate

    def _detect_stress_indicators(self, y, sr):
        """Detect acoustic stress indicators."""
        try:
            # Extract spectral centroid (brightness - higher when stressed)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_mean = np.mean(spectral_centroids)
            centroid_std = np.std(spectral_centroids)
            
            # Extract zero crossing rate (higher with tension)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            
            # Extract RMS energy (amplitude variation)
            rms = librosa.feature.rms(y=y)[0]
            rms_std = np.std(rms)
            
            # Composite stress score
            stress_level = min(
                (centroid_std / 2000 + zcr_mean * 0.5 + rms_std / 0.1) / 3,
                1.0
            )
            
            return {
                "spectral_centroid": round(centroid_mean, 2),
                "spectral_variance": round(centroid_std, 2),
                "zero_crossing_rate": round(zcr_mean, 3),
                "rms_energy_variance": round(rms_std, 3),
                "stress_level": round(stress_level, 3),
            }
        except Exception:
            return {
                "spectral_centroid": 0,
                "spectral_variance": 0,
                "zero_crossing_rate": 0,
                "rms_energy_variance": 0,
                "stress_level": 0.5,
            }

    def get_cognitive_load_report(self, analysis):
        """Generate human-readable cognitive load report."""
        if "error" in analysis:
            return f"Cognitive Load Analysis Error: {analysis['error']}"

        pa = analysis.get("pause_analysis", {})
        si = analysis.get("stress_indicators", {})

        report = f"""
COGNITIVE LOAD & MENTAL STRESS ANALYSIS
========================================

OVERALL COGNITIVE LOAD SCORE: {analysis['cognitive_load_score']:.1%}
Mental Stress Level: {analysis['mental_stress_level']}

PAUSE ANALYSIS:
- Pause Ratio: {pa.get('pause_ratio', 0):.1%} of speech time
- Total Pause Duration: {pa.get('total_pause_duration', 0):.2f}s
- Number of Pauses: {pa.get('num_pauses', 0)}
- Average Pause Length: {pa.get('avg_pause_length', 0):.2f}s

VOICE STABILITY:
- Stability Score: {analysis['voice_stability']:.1%}
- Interpretation: {'Stable voice suggests confidence.' if analysis['voice_stability'] > 0.7 else 'Voice shows signs of instability.' if analysis['voice_stability'] < 0.5 else 'Moderate voice stability.'}

HESITATION INDICATORS:
- Hesitation Score: {analysis['hesitation_score']:.1%}
- Filler words and speech disfluencies detected

SPEECH RATE:
- Estimated Rate: {analysis['speech_rate']:.1f} words/minute
- Typical Rate: 150 wpm
- Deviation: {'Faster than typical (stress)' if analysis['speech_rate'] > 170 else 'Slower than typical (deliberation)' if analysis['speech_rate'] < 130 else 'Within normal range'}

ACOUSTIC STRESS INDICATORS:
- Spectral Centroid: {si.get('spectral_centroid', 0):.0f} Hz
- Zero Crossing Rate: {si.get('zero_crossing_rate', 0):.3f}
- Energy Variance: {si.get('rms_energy_variance', 0):.3f}

FORENSIC ASSESSMENT:
{'✓ Low cognitive load - Testimony appears fluent and rehearsed.' if analysis['cognitive_load_score'] < 0.40 else '⚠ Moderate cognitive load - Signs of mental effort detected.' if analysis['cognitive_load_score'] < 0.70 else '✗ High cognitive load - Significant stress and cognitive strain observed.'}

Duration: {analysis['duration_seconds']:.2f} seconds
        """
        return report.strip()
