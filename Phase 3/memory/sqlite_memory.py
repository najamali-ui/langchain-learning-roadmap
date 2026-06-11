"""
memory/sqlite_memory.py — SQLite-backed persistent chat message history.

Implements BaseChatMessageHistory so it integrates natively with
LangChain's RunnableWithMessageHistory for persistent cross-session memory.
"""

from typing import List, Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, messages_from_dict, messages_to_dict
from database.db import save_message, get_session_messages


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history stored in SQLite.
    Automatically persists every message so history survives app restarts.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:
        """Load all messages from SQLite for this session."""
        rows = get_session_messages(self.session_id)
        result = []
        for row in rows:
            if row["role"] == "human":
                result.append(HumanMessage(content=row["content"]))
            elif row["role"] == "ai":
                result.append(AIMessage(content=row["content"]))
        return result

    def add_message(self, message: BaseMessage) -> None:
        """Persist a single message to the database."""
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        else:
            role = "system"
        save_message(self.session_id, role, message.content)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Persist multiple messages."""
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        """Clear all messages for this session from the database."""
        from database.db import get_connection
        with get_connection() as conn:
            conn.execute(
                "DELETE FROM research_messages WHERE session_id = ?",
                (self.session_id,)
            )
