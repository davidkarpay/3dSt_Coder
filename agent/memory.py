"""Memory management for conversation history and context."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
import json
import os
import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in the conversation."""

    role: str  # 'user', 'assistant', 'tool', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool: Optional[str] = None  # Tool name if role is 'tool'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.tool:
            data["tool"] = self.tool
        if self.metadata:
            data["metadata"] = self.metadata
        return data


class ConversationMemory:
    """Manages conversation history with token budget awareness.

    Features:
    - In-memory storage for current session
    - Optional SQLite persistence
    - Token budget management
    - Context window truncation
    """

    def __init__(
        self,
        max_tokens: int = 2048,
        persist_path: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """Initialize conversation memory.

        Args:
            max_tokens: Maximum tokens to keep in context
            persist_path: Optional path to SQLite database for persistence
            project_id: Optional project identifier for scoping conversations
        """
        self.max_tokens = max_tokens
        self.persist_path = persist_path
        self.project_id = project_id or "default"
        self.messages: List[Message] = []
        self._db_connection = None

    async def initialize(self) -> None:
        """Initialize database connection if persistence is enabled."""
        if self.persist_path:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            self._db_connection = await aiosqlite.connect(self.persist_path)
            await self._create_tables()
            await self._load_recent_messages()

    async def _create_tables(self) -> None:
        """Create database tables for message persistence."""
        if not self._db_connection:
            return

        await self._db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool TEXT,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db_connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_project_timestamp
            ON conversations(project_id, timestamp DESC)
            """
        )
        await self._db_connection.commit()

    async def _load_recent_messages(self, limit: int = 50) -> None:
        """Load recent messages from database."""
        if not self._db_connection:
            return

        cursor = await self._db_connection.execute(
            """
            SELECT role, content, tool, metadata, timestamp
            FROM conversations
            WHERE project_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (self.project_id, limit),
        )

        rows = await cursor.fetchall()
        self.messages = []

        for row in reversed(rows):  # Reverse to get chronological order
            role, content, tool, metadata_str, timestamp_str = row
            metadata = json.loads(metadata_str) if metadata_str else {}
            timestamp = datetime.fromisoformat(timestamp_str)

            self.messages.append(
                Message(
                    role=role,
                    content=content,
                    tool=tool,
                    metadata=metadata,
                    timestamp=timestamp,
                )
            )

        logger.info(f"Loaded {len(self.messages)} messages from database")

    async def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.

        Args:
            content: User's message content
        """
        message = Message(role="user", content=content)
        self.messages.append(message)
        await self._persist_message(message)
        await self._truncate_if_needed()

    async def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.

        Args:
            content: Assistant's response content
        """
        message = Message(role="assistant", content=content)
        self.messages.append(message)
        await self._persist_message(message)
        await self._truncate_if_needed()

    async def add_tool_result(self, tool_name: str, result: str) -> None:
        """Add a tool execution result to the conversation.

        Args:
            tool_name: Name of the tool that was executed
            result: Tool execution result
        """
        message = Message(role="tool", content=result, tool=tool_name)
        self.messages.append(message)
        await self._persist_message(message)

    async def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation.

        Args:
            content: System message content
        """
        message = Message(role="system", content=content)
        self.messages.append(message)
        await self._persist_message(message)

    async def _persist_message(self, message: Message) -> None:
        """Persist a message to the database.

        Args:
            message: Message to persist
        """
        if not self._db_connection:
            return

        await self._db_connection.execute(
            """
            INSERT INTO conversations (project_id, role, content, tool, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.project_id,
                message.role,
                message.content,
                message.tool,
                json.dumps(message.metadata) if message.metadata else None,
                message.timestamp.isoformat(),
            ),
        )
        await self._db_connection.commit()

    async def _truncate_if_needed(self) -> None:
        """Truncate conversation history if it exceeds token budget.

        Uses a simple character-based approximation (1 token ≈ 4 characters).
        """
        # Approximate token count (1 token ≈ 4 characters)
        total_chars = sum(len(msg.content) for msg in self.messages)
        estimated_tokens = total_chars // 4

        if estimated_tokens > self.max_tokens:
            # Keep system messages and recent messages
            system_messages = [msg for msg in self.messages if msg.role == "system"]
            other_messages = [msg for msg in self.messages if msg.role != "system"]

            # Remove oldest non-system messages
            while estimated_tokens > self.max_tokens and len(other_messages) > 1:
                removed = other_messages.pop(0)
                estimated_tokens -= len(removed.content) // 4

            self.messages = system_messages + other_messages
            logger.info(f"Truncated conversation to {len(self.messages)} messages")

    def get_context(self) -> List[Dict[str, Any]]:
        """Get conversation context for prompt building.

        Returns:
            List of message dictionaries
        """
        return [msg.to_dict() for msg in self.messages]

    async def clear(self) -> None:
        """Clear conversation memory."""
        self.messages = []
        logger.info("Conversation memory cleared")

    async def close(self) -> None:
        """Close database connection if open."""
        if self._db_connection:
            await self._db_connection.close()
            self._db_connection = None

    async def search_messages(self, query: str, limit: int = 10) -> List[Message]:
        """Search through conversation history.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching messages
        """
        if not self._db_connection:
            # Search in-memory messages
            results = []
            for msg in self.messages:
                if query.lower() in msg.content.lower():
                    results.append(msg)
                    if len(results) >= limit:
                        break
            return results

        # Search in database
        cursor = await self._db_connection.execute(
            """
            SELECT role, content, tool, metadata, timestamp
            FROM conversations
            WHERE project_id = ? AND content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (self.project_id, f"%{query}%", limit),
        )

        rows = await cursor.fetchall()
        results = []

        for row in rows:
            role, content, tool, metadata_str, timestamp_str = row
            metadata = json.loads(metadata_str) if metadata_str else {}
            timestamp = datetime.fromisoformat(timestamp_str)

            results.append(
                Message(
                    role=role,
                    content=content,
                    tool=tool,
                    metadata=metadata,
                    timestamp=timestamp,
                )
            )

        return results