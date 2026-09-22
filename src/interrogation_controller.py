"""
AI Interrogation Room Controller.

Orchestrates interactive interrogation flow with:
- Question sequencing
- Real-time analysis
- Text-to-speech investigator voice
- Live deception probability updates
- Dynamic risk level tracking
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Optional


class InterrogationController:
    def __init__(self):
        """Initialize interrogation room controller."""
        self.interrogation_questions = [
            "State your full name and date of birth.",
            "Where were you on the date and time in question?",
            "Describe the events you witnessed in detail, from beginning to end.",
            "Who else was present at the scene?",
            "Why do you think you were contacted for this investigation?",
            "Has anyone pressured you to provide false information?",
            "Walk us through your actions step by step.",
            "Can you explain any inconsistencies in your statement?",
            "What is the most important detail you want us to understand?",
            "Is there anything else you'd like to add to your testimony?",
        ]
        
        self.session_state = {
            "active": False,
            "question_index": 0,
            "session_start_time": None,
            "responses": [],
            "risk_scores": [],
            "current_risk": 0.5,
        }

    def start_interrogation(self):
        """Start a new interrogation session."""
        self.session_state = {
            "active": True,
            "question_index": 0,
            "session_start_time": datetime.now().isoformat(),
            "responses": [],
            "risk_scores": [],
            "current_risk": 0.5,
        }
        return self.get_current_question()

    def get_current_question(self) -> Dict:
        """Get current interrogation question."""
        if self.session_state["question_index"] >= len(self.interrogation_questions):
            return {"status": "complete", "message": "Interrogation session completed."}

        question = self.interrogation_questions[self.session_state["question_index"]]
        return {
            "status": "active",
            "question": question,
            "question_number": self.session_state["question_index"] + 1,
            "total_questions": len(self.interrogation_questions),
            "elapsed_time": self._get_elapsed_time(),
            "current_risk": self.session_state["current_risk"],
        }

    def record_response(self, response_text: str, deception_score: float, audio_path: Optional[str] = None):
        """
        Record subject's response and update risk assessment.
        
        Args:
            response_text: Transcribed response text
            deception_score: AI-predicted deception probability (0-1)
            audio_path: Path to recorded audio file
        """
        current_q = self.interrogation_questions[self.session_state["question_index"]]
        
        response_record = {
            "question_number": self.session_state["question_index"] + 1,
            "question": current_q,
            "response": response_text,
            "deception_score": deception_score,
            "timestamp": datetime.now().isoformat(),
            "audio_path": audio_path,
        }
        
        self.session_state["responses"].append(response_record)
        self.session_state["risk_scores"].append(deception_score)
        
        # Update running risk average
        self.session_state["current_risk"] = sum(self.session_state["risk_scores"]) / len(self.session_state["risk_scores"])
        
        # Move to next question
        self.session_state["question_index"] += 1

    def get_session_summary(self) -> Dict:
        """Get comprehensive interrogation session summary."""
        if not self.session_state["responses"]:
            return {"error": "No interrogation data available."}

        avg_risk = sum(self.session_state["risk_scores"]) / len(self.session_state["risk_scores"])
        max_risk = max(self.session_state["risk_scores"])
        min_risk = min(self.session_state["risk_scores"])
        high_risk_count = sum(1 for s in self.session_state["risk_scores"] if s > 0.7)

        if avg_risk > 0.75:
            final_assessment = "HIGH DECEPTION RISK - Further investigation recommended"
            assessment_color = "#dc2626"
        elif avg_risk > 0.55:
            final_assessment = "MODERATE DECEPTION INDICATORS - Cross-examination advised"
            assessment_color = "#f59e0b"
        else:
            final_assessment = "LOW DECEPTION INDICATORS - Testimony appears consistent"
            assessment_color = "#16a34a"

        return {
            "total_questions": len(self.session_state["responses"]),
            "session_duration": self._get_elapsed_time(),
            "average_risk_score": round(avg_risk, 3),
            "peak_risk_score": round(max_risk, 3),
            "minimum_risk_score": round(min_risk, 3),
            "high_risk_questions": high_risk_count,
            "final_assessment": final_assessment,
            "assessment_color": assessment_color,
            "responses": self.session_state["responses"],
            "risk_trajectory": self.session_state["risk_scores"],
        }

    def end_interrogation(self) -> Dict:
        """End interrogation session and return final summary."""
        self.session_state["active"] = False
        return self.get_session_summary()

    def _get_elapsed_time(self) -> str:
        """Calculate elapsed interrogation time."""
        if not self.session_state["session_start_time"]:
            return "00:00"
        
        try:
            start = datetime.fromisoformat(self.session_state["session_start_time"])
            elapsed = datetime.now() - start
            minutes = int(elapsed.total_seconds() // 60)
            seconds = int(elapsed.total_seconds() % 60)
            return f"{minutes:02d}:{seconds:02d}"
        except Exception:
            return "00:00"

    def get_interrogation_statistics(self) -> Dict:
        """Get detailed interrogation statistics."""
        if not self.session_state["responses"]:
            return {}

        questions_with_high_risk = [
            r for r in self.session_state["responses"] if r["deception_score"] > 0.7
        ]

        return {
            "total_responses": len(self.session_state["responses"]),
            "average_response_length": sum(len(r["response"].split()) for r in self.session_state["responses"]) / len(self.session_state["responses"]),
            "high_risk_questions": len(questions_with_high_risk),
            "high_risk_questions_list": [r["question"] for r in questions_with_high_risk],
            "risk_progression": self.session_state["risk_scores"],
            "most_concerning_response": max(
                self.session_state["responses"],
                key=lambda x: x["deception_score"]
            ) if self.session_state["responses"] else None,
        }

    def reset_session(self):
        """Reset interrogation session."""
        self.session_state = {
            "active": False,
            "question_index": 0,
            "session_start_time": None,
            "responses": [],
            "risk_scores": [],
            "current_risk": 0.5,
        }
