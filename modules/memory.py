"""Memory management module.

Provides conversation memory with pluggable backends.
No dependencies on other modules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from core.interfaces import Conversation, MemoryBackend, MemoryEntry, Message, Role

logger = logging.getLogger(__name__)


class SQLiteMemoryBackend:
    """SQLite-based memory backend.

    Stores memories in local SQLite database.
    No external dependencies required.

    Example:
        memory = SQLiteMemoryBackend("data/memory.db")
        await memory.store(MemoryEntry(content="User likes Python"))

        results = await memory.recall("What does user like?", limit=5)
    """

    def __init__(self, db_path: str | Path = "data/memory.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._local = threading.local()
        self._init_db()

    @property
    def _connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._db_path))
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    source TEXT DEFAULT 'unknown',
                    importance REAL DEFAULT 1.0,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding BLOB
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_source
                ON memories(source)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_created
                ON memories(created_at)
            """)

            conn.commit()

    async def store(self, entry: MemoryEntry) -> None:
        """Store memory entry.

        Args:
            entry: Memory entry to store
        """
        async with self._lock:
            await asyncio.to_thread(self._store_sync, entry)

    def _store_sync(self, entry: MemoryEntry) -> None:
        """Synchronous store implementation."""
        try:
            self._connection.execute(
                """
                INSERT INTO memories (content, source, importance, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (
                    entry.content,
                    entry.source,
                    entry.importance,
                    json.dumps(entry.metadata),
                ),
            )
            self._connection.commit()
        except sqlite3.Error as e:
            logger.error("Failed to store memory: %s", e)
            raise

    async def recall(
        self,
        query: str,
        limit: int = 5,
        **filters: Any,
    ) -> list[MemoryEntry]:
        """Recall relevant memories.

        Simple keyword-based search. Can be extended with embeddings.

        Args:
            query: Search query
            limit: Maximum results
            **filters: Additional filters (source, etc.)

        Returns:
            List of matching memory entries
        """
        async with self._lock:
            return await asyncio.to_thread(self._recall_sync, query, limit, filters)

    def _recall_sync(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any],
    ) -> list[MemoryEntry]:
        """Synchronous recall implementation."""
        try:
            # Build query with filters
            conditions = ["content LIKE ?"]
            params: list[Any] = [f"%{query}%"]

            if "source" in filters:
                conditions.append("source = ?")
                params.append(filters["source"])

            where_clause = " AND ".join(conditions)

            cursor = self._connection.execute(
                f"""
                SELECT content, source, importance, metadata
                FROM memories
                WHERE {where_clause}
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
                """,
                (*params, limit),
            )

            rows = cursor.fetchall()
            return [
                MemoryEntry(
                    content=row["content"],
                    source=row["source"],
                    importance=row["importance"],
                    metadata=json.loads(row["metadata"]),
                )
                for row in rows
            ]

        except sqlite3.Error as e:
            logger.error("Failed to recall memories: %s", e)
            return []

    async def delete(self, entry_id: str) -> bool:
        """Delete memory entry by ID."""
        async with self._lock:
            return await asyncio.to_thread(self._delete_sync, entry_id)

    def _delete_sync(self, entry_id: str) -> bool:
        """Synchronous delete implementation."""
        try:
            cursor = self._connection.execute(
                "DELETE FROM memories WHERE id = ?",
                (entry_id,),
            )
            self._connection.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error("Failed to delete memory: %s", e)
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        async with self._lock:
            return await asyncio.to_thread(self._stats_sync)

    def _stats_sync(self) -> dict[str, Any]:
        """Synchronous stats implementation."""
        cursor = self._connection.execute(
            "SELECT COUNT(*) as total, source FROM memories GROUP BY source"
        )
        sources = {row["source"]: row["total"] for row in cursor.fetchall()}

        cursor = self._connection.execute(
            "SELECT COUNT(*) as total FROM memories"
        )
        total = cursor.fetchone()["total"]

        return {"total": total, "by_source": sources}

    async def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


class InMemoryBackend:
    """In-memory memory backend for testing.

    No persistence. Data is lost on restart.
    """

    def __init__(self) -> None:
        self._memories: list[MemoryEntry] = []

    async def store(self, entry: MemoryEntry) -> None:
        self._memories.append(entry)

    async def recall(
        self,
        query: str,
        limit: int = 5,
        **filters: Any,
    ) -> list[MemoryEntry]:
        results = []
        query_lower = query.lower()

        for entry in self._memories:
            if query_lower in entry.content.lower():
                if "source" in filters and entry.source != filters["source"]:
                    continue
                results.append(entry)

        # Sort by importance
        results.sort(key=lambda e: e.importance, reverse=True)
        return results[:limit]

    async def delete(self, entry_id: str) -> bool:
        # For in-memory, we don't have IDs, so this is a no-op
        return False

    async def clear(self) -> None:
        """Clear all memories."""
        self._memories.clear()


import threading


class MemoryManager:
    """High-level memory management.

    Provides conversation context building and memory storage.

    Example:
        memory = MemoryManager(SQLiteMemoryBackend("memory.db"))
        context = await memory.build_context("user123", "What's the weather?")
    """

    def __init__(self, backend: MemoryBackend) -> None:
        self._backend = backend

    async def remember(
        self,
        content: str,
        source: str = "conversation",
        importance: float = 1.0,
        **metadata: Any,
    ) -> None:
        """Store a memory.

        Args:
            content: Memory content
            source: Memory source
            importance: Importance score (0-10)
            **metadata: Additional metadata
        """
        entry = MemoryEntry(
            content=content,
            source=source,
            importance=importance,
            metadata=metadata,
        )
        await self._backend.store(entry)

    async def recall(
        self,
        query: str,
        limit: int = 5,
        **filters: Any,
    ) -> list[MemoryEntry]:
        """Recall relevant memories."""
        return await self._backend.recall(query, limit, **filters)

    async def build_context(
        self,
        user_id: str,
        current_message: str,
        max_memories: int = 5,
    ) -> Conversation:
        """Build conversation context with relevant memories.

        Args:
            user_id: User identifier
            current_message: Current user message
            max_memories: Maximum memories to include

        Returns:
            Conversation with system prompt and relevant memories
        """
        conversation = Conversation()

        # Add system prompt
        conversation.add_message(
            Role.SYSTEM,
            "You are a helpful AI assistant. Use relevant memories to personalize responses.",
        )

        # Add relevant memories as context
        memories = await self.recall(current_message, limit=max_memories)
        if memories:
            memory_text = "\n".join(
                f"- {m.content}" for m in memories
            )
            conversation.add_message(
                Role.SYSTEM,
                f"Relevant context:\n{memory_text}",
            )

        # Add user message
        conversation.add_message(Role.USER, current_message)

        return conversation

    async def store_interaction(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Store conversation interaction.

        Args:
            user_id: User identifier
            user_message: User's message
            assistant_response: Assistant's response
        """
        await self.remember(
            content=f"User: {user_message}\nAssistant: {assistant_response}",
            source=f"user:{user_id}",
            importance=1.0,
            user_id=user_id,
        )

    async def close(self) -> None:
        """Cleanup resources."""
        if hasattr(self._backend, "close"):
            await self._backend.close()
