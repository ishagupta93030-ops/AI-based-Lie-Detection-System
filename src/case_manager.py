import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class CaseManager:
    def __init__(self, db_path: Path, json_path: Optional[Path] = None):
        self.db_path = db_path
        self.json_path = json_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    mode TEXT,
                    input_summary TEXT,
                    audio_score REAL,
                    video_score REAL,
                    text_score REAL,
                    final_score REAL,
                    prediction TEXT,
                    confidence REAL,
                    risk_level TEXT,
                    audio_insight TEXT,
                    video_insight TEXT,
                    text_insight TEXT,
                    recommendations TEXT,
                    language TEXT,
                    metadata TEXT
                )
                """
            )
            conn.commit()

    def add_case(self, case: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cases (
                    case_id, timestamp, mode, input_summary, audio_score,
                    video_score, text_score, final_score, prediction,
                    confidence, risk_level, audio_insight, video_insight,
                    text_insight, recommendations, language, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case.get("case_id"),
                    case.get("timestamp"),
                    case.get("mode"),
                    case.get("input_summary"),
                    case.get("audio_score"),
                    case.get("video_score"),
                    case.get("text_score"),
                    case.get("final_score"),
                    case.get("prediction"),
                    case.get("confidence"),
                    case.get("risk_level"),
                    case.get("audio_insight"),
                    case.get("video_insight"),
                    case.get("text_insight"),
                    case.get("recommendations"),
                    case.get("language", "en"),
                    json.dumps(case.get("metadata", {})),
                ),
            )
            conn.commit()

        if self.json_path:
            self._sync_json()

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            return self._row_to_dict(row) if row else None

    def get_history(self, search: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            if search:
                query = """
                    SELECT * FROM cases
                    WHERE case_id LIKE ? OR mode LIKE ? OR prediction LIKE ? OR risk_level LIKE ? OR input_summary LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                pattern = f"%{search}%"
                rows = conn.execute(query, (pattern, pattern, pattern, pattern, pattern, limit)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM cases ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def list_case_ids(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT case_id FROM cases ORDER BY timestamp DESC").fetchall()
            return [row[0] for row in rows]

    def _sync_json(self) -> None:
        history = self.get_history(limit=1000)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "case_id": row[0],
            "timestamp": row[1],
            "mode": row[2],
            "input_summary": row[3],
            "audio_score": row[4],
            "video_score": row[5],
            "text_score": row[6],
            "final_score": row[7],
            "prediction": row[8],
            "confidence": row[9],
            "risk_level": row[10],
            "audio_insight": row[11],
            "video_insight": row[12],
            "text_insight": row[13],
            "recommendations": row[14],
            "language": row[15],
            "metadata": json.loads(row[16] or "{}"),
        }
