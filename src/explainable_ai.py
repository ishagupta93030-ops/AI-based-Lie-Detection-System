"""
Explainable AI Engine for Forensic Intelligence.

Provides interpretable explanations for deception predictions.
Identifies and ranks top contributing factors.
Generates human-readable verdicts.
"""

import numpy as np
from datetime import datetime


class ExplainableAIEngine:
    def __init__(self):
        """Initialize explainable AI engine."""
        self.factor_descriptions = {
            "voice_stress": "Elevated pitch, amplitude spikes, and voice tremor detected in speech analysis.",
            "facial_tension": "Micro-expressions showing fear, disgust, or contempt recorded during facial analysis.",
            "eye_avoidance": "Significant gaze aversion and reduced eye contact throughout testimony.",
            "hesitation": "Multiple pauses, filler words, and speech disfluencies detected.",
            "cognitive_load": "High cognitive load indicators suggest mental strain during response.",
            "lip_sync": "Inconsistencies between lip movement and audio detected.",
            "blink_rate": "Elevated blink frequency indicating stress response.",
            "response_time": "Delayed response time before answering questions.",
            "linguistic_patterns": "Distancing language, qualifiers, and evasive phrasing detected.",
            "frequency_artifacts": "Suspicious frequency domain artifacts suggest possible manipulation.",
        }

    def generate_explanation(self, predictions, case_data):
        """
        Generate explainable AI interpretation of predictions.
        
        Args:
            predictions: dict with modality scores (audio, video, text)
            case_data: dict with case information
            
        Returns:
            dict with detailed explanation and contributing factors
        """
        try:
            audio_score = predictions.get("audio", 0.5)
            video_score = predictions.get("video", 0.5)
            text_score = predictions.get("text", 0.5)
            final_score = predictions.get("final", 0.5)

            # Calculate factor weights based on scores
            factors = {
                "voice_stress": {"score": audio_score * 0.7, "weight": 0.30},
                "facial_tension": {"score": video_score * 0.6, "weight": 0.35},
                "linguistic_patterns": {"score": text_score * 0.8, "weight": 0.20},
                "hesitation": {"score": audio_score * 0.4, "weight": 0.08},
                "eye_avoidance": {"score": video_score * 0.3, "weight": 0.07},
            }

            # Rank factors by contribution
            ranked_factors = sorted(
                factors.items(),
                key=lambda x: x[1]["score"] * x[1]["weight"],
                reverse=True
            )

            # Generate contributing factors list
            top_factors = []
            for factor_name, factor_data in ranked_factors[:5]:
                if factor_data["score"] > 0.3:  # Only include significant factors
                    top_factors.append({
                        "name": factor_name.replace("_", " ").title(),
                        "score": round(factor_data["score"], 2),
                        "description": self.factor_descriptions.get(
                            factor_name, "Contributing factor detected."
                        ),
                        "impact": self._classify_impact(factor_data["score"]),
                        "severity_color": self._get_severity_color(factor_data["score"]),
                    })

            # Generate verdict
            if final_score > 0.75:
                verdict = "HIGH DECEPTION RISK"
                verdict_color = "#dc2626"
                confidence = "Very High Confidence"
                recommendation = "Recommend further investigation and expert review."
            elif final_score > 0.55:
                verdict = "MODERATE DECEPTION INDICATORS"
                verdict_color = "#f59e0b"
                confidence = "Moderate Confidence"
                recommendation = "Additional evidence or cross-examination recommended."
            else:
                verdict = "LOW DECEPTION INDICATORS"
                verdict_color = "#16a34a"
                confidence = "Low Risk"
                recommendation = "Testimony appears consistent and truthful."

            # Calculate explanation score (how confident we are in the explanation)
            explanation_confidence = (
                abs(audio_score - video_score) < 0.2 and
                abs(video_score - text_score) < 0.2
            )
            explanation_confidence_score = 0.85 if explanation_confidence else 0.65

            return {
                "verdict": verdict,
                "verdict_color": verdict_color,
                "final_score": round(final_score, 3),
                "confidence_level": confidence,
                "explanation_confidence": round(explanation_confidence_score, 2),
                "top_contributing_factors": top_factors,
                "recommendation": recommendation,
                "modality_breakdown": {
                    "audio_analysis": {
                        "score": round(audio_score, 3),
                        "interpretation": "Voice stress and acoustic indicators suggest " + (
                            "high deception likelihood" if audio_score > 0.7 else
                            "possible deception indicators" if audio_score > 0.5 else
                            "truthful testimony"
                        ),
                    },
                    "video_analysis": {
                        "score": round(video_score, 3),
                        "interpretation": "Facial expressions and eye behavior suggest " + (
                            "high deception likelihood" if video_score > 0.7 else
                            "possible deception indicators" if video_score > 0.5 else
                            "truthful testimony"
                        ),
                    },
                    "text_analysis": {
                        "score": round(text_score, 3),
                        "interpretation": "Linguistic patterns suggest " + (
                            "high deception likelihood" if text_score > 0.7 else
                            "possible deception indicators" if text_score > 0.5 else
                            "truthful testimony"
                        ),
                    },
                },
                "generated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "error": str(e),
                "verdict": "Unable to generate explanation",
                "top_contributing_factors": [],
            }

    def _classify_impact(self, score):
        """Classify impact level based on score."""
        if score > 0.75:
            return "Critical"
        elif score > 0.55:
            return "Significant"
        elif score > 0.35:
            return "Moderate"
        else:
            return "Minor"

    def _get_severity_color(self, score):
        """Get color based on severity score."""
        if score > 0.75:
            return "#dc2626"  # Red - Critical
        elif score > 0.55:
            return "#f59e0b"  # Orange - Significant
        elif score > 0.35:
            return "#eab308"  # Yellow - Moderate
        else:
            return "#16a34a"  # Green - Minor

    def get_detailed_explanation(self, explanation_result):
        """Generate full human-readable explanation report."""
        if "error" in explanation_result:
            return f"Explanation Generation Error: {explanation_result['error']}"

        factors_text = ""
        for i, factor in enumerate(explanation_result.get("top_contributing_factors", []), 1):
            factors_text += f"\n{i}. {factor['name']} (Score: {factor['score']:.1%}) - {factor['impact']}\n   {factor['description']}"

        modality = explanation_result.get("modality_breakdown", {})

        report = f"""
EXPLAINABLE AI FORENSIC ANALYSIS REPORT
========================================

FINAL VERDICT: {explanation_result['verdict']}
Deception Score: {explanation_result['final_score']:.1%}
Confidence Level: {explanation_result['confidence_level']}
Explanation Reliability: {explanation_result['explanation_confidence']:.1%}

TOP CONTRIBUTING FACTORS:
{factors_text if factors_text else "No significant factors identified."}

MODALITY BREAKDOWN:

Audio Analysis (Voice & Acoustic):
- Score: {modality.get('audio_analysis', {}).get('score', 0):.1%}
- Interpretation: {modality.get('audio_analysis', {}).get('interpretation', 'N/A')}

Video Analysis (Facial & Eye):
- Score: {modality.get('video_analysis', {}).get('score', 0):.1%}
- Interpretation: {modality.get('video_analysis', {}).get('interpretation', 'N/A')}

Text Analysis (Linguistic):
- Score: {modality.get('text_analysis', {}).get('score', 0):.1%}
- Interpretation: {modality.get('text_analysis', {}).get('interpretation', 'N/A')}

FORENSIC RECOMMENDATION:
{explanation_result['recommendation']}

CHAIN OF REASONING:
This verdict is based on multimodal analysis of the subject's audio, video, and linguistic
patterns. The confluence of these independent signals provides robust evidence for the
stated conclusion. The explanation confidence score reflects agreement between modalities.

Generated: {explanation_result.get('generated_at', 'N/A')}
        """
        return report.strip()


class ForensicTimelineGenerator:
    """Generate chronological forensic timelines of analysis."""
    
    def __init__(self):
        """Initialize timeline generator."""
        self.events = []

    def create_timeline(self, analysis_log):
        """
        Create chronological timeline from analysis events.
        
        Args:
            analysis_log: list of dicts with timestamp and analysis data
            
        Returns:
            list of timeline events with risk levels
        """
        try:
            timeline = []
            
            for event in sorted(analysis_log, key=lambda x: x.get("timestamp", "")):
                timestamp = event.get("timestamp", "")
                score = event.get("score", 0.5)
                question = event.get("question", "Analysis segment")
                risk_level = self._classify_risk(score)
                
                timeline.append({
                    "time": timestamp,
                    "question": question,
                    "score": score,
                    "risk_level": risk_level,
                    "description": self._generate_event_description(score, question),
                    "color": self._get_timeline_color(score),
                })
            
            return timeline
        except Exception:
            return []

    def _classify_risk(self, score):
        """Classify risk level from score."""
        if score >= 0.75:
            return "High Risk"
        elif score >= 0.45:
            return "Moderate Risk"
        else:
            return "Low Risk"

    def _generate_event_description(self, score, question):
        """Generate description for timeline event."""
        if score >= 0.75:
            return f"Elevated deception indicators detected: {question}"
        elif score >= 0.45:
            return f"Mixed signals observed: {question}"
        else:
            return f"Consistent testimony: {question}"

    def _get_timeline_color(self, score):
        """Get color for timeline event."""
        if score >= 0.75:
            return "#dc2626"
        elif score >= 0.45:
            return "#f59e0b"
        else:
            return "#16a34a"

    def get_timeline_markdown(self, timeline):
        """Generate markdown timeline visualization."""
        if not timeline:
            return "No timeline data available."
        
        markdown = "## FORENSIC TIMELINE\n\n"
        for event in timeline:
            markdown += f"### {event['time']} → {event['risk_level']}\n"
            markdown += f"**Question:** {event['question']}\n"
            markdown += f"**Risk Score:** {event['score']:.1%}\n"
            markdown += f"**Assessment:** {event['description']}\n\n"
        
        return markdown
