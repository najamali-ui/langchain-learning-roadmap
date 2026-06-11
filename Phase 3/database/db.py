"""
database/db.py — SQLite schema and CRUD operations for AI Research Assistant.

Tables:
  - research_sessions: metadata for each conversation session
  - research_messages: all chat messages with role, content, timestamp
"""

import sqlite3
import uuid
from datetime import datetime
from typing import Optional
from config import config


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row factory set."""
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the database schema if not already present."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS research_sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,   -- 'human' | 'ai' | 'system'
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES research_sessions(id)
            );
        """)


def create_session(title: str = "New Research Session") -> str:
    """Create a new research session and return its ID."""
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO research_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now)
        )
    return session_id


def update_session_title(session_id: str, title: str) -> None:
    """Update the title of a session (e.g. auto-titled from first question)."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE research_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, session_id)
        )


def save_message(session_id: str, role: str, content: str) -> None:
    """Persist a single message to the database."""
    msg_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        # Insert message
        conn.execute(
            "INSERT INTO research_messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (msg_id, session_id, role, content, now)
        )
        # Update session updated_at
        conn.execute(
            "UPDATE research_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id)
        )


def get_session_messages(session_id: str) -> list[dict]:
    """Retrieve all messages for a given session, ordered by time."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM research_messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def list_sessions(limit: int = 50) -> list[dict]:
    """List all sessions ordered by most recently updated."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM research_sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def delete_session(session_id: str) -> None:
    """Delete a session and all its messages."""
    with get_connection() as conn:
        conn.execute("DELETE FROM research_messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM research_sessions WHERE id = ?", (session_id,))


# Auto-initialize on import
init_db()
