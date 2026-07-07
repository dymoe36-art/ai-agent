"""Abstract interfaces for all system components.

This module defines contracts that all implementations must follow.
No external dependencies except standard library and pydantic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, Field


# ============================================================================
# Message Types
# ============================================================================

class Role(StrEnum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """Single message in conversation."""
    role: Role
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """Conversation context."""
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, role: Role, content: str, **meta: Any) -> None:
        """Add message to conversation."""
        self.messages.append(Message(role=role, content=content, metadata=meta))


# ============================================================================
# LLM Provider Interface
# ============================================================================

class LLMResponse(BaseModel):
    """Structured response from LLM provider."""
    content: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamChunk(BaseModel):
    """Single chunk from streaming response."""
    content: str
    is_finished: bool = False


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for LLM providers (Ollama, OpenAI, etc.)."""

    @property
    def name(self) -> str:
        """Provider name."""
        ...

    @property
    def is_available(self) -> bool:
        """Check if provider is ready to serve requests."""
        ...

    async def generate(
        self,
        conversation: Conversation,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion for conversation."""
        ...

    async def stream(
        self,
        conversation: Conversation,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion chunks."""
        ...


# ============================================================================
# Memory Interface
# ============================================================================

@dataclass
class MemoryEntry:
    """Single memory entry."""
    content: str
    source: str = "unknown"
    importance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemoryBackend(Protocol):
    """Interface for memory storage backends."""

    async def store(self, entry: MemoryEntry) -> None:
        """Store memory entry."""
        ...

    async def recall(
        self,
        query: str,
        limit: int = 5,
        **filters: Any,
    ) -> list[MemoryEntry]:
        """Recall relevant memories."""
        ...

    async def delete(self, entry_id: str) -> bool:
        """Delete memory entry by ID."""
        ...


# ============================================================================
# Tool/Plugin Interface
# ============================================================================

class ToolParameter(BaseModel):
    """Tool parameter definition."""
    name: str
    description: str
    type: str = "string"
    required: bool = True
    default: Any | None = None


class ToolDefinition(BaseModel):
    """Tool metadata for registration."""
    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    category: str = "general"
    requires_confirmation: bool = False


class ToolResult(BaseModel):
    """Tool execution result."""
    success: bool
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    """Interface for executable tools."""

    @property
    def definition(self) -> ToolDefinition:
        """Tool metadata."""
        ...

    async def execute(self, **params: Any) -> ToolResult:
        """Execute tool with given parameters."""
        ...


# ============================================================================
# Event System
# ============================================================================

class EventType(StrEnum):
    """Built-in event types."""
    # Message events
    MESSAGE_RECEIVED = "message.received"
    MESSAGE_SENT = "message.sent"

    # LLM events
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_ERROR = "llm.error"

    # Tool events
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"
    TOOL_ERROR = "tool.error"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"

    # Plugin events
    PLUGIN_LOADED = "plugin.loaded"
    PLUGIN_UNLOADED = "plugin.unloaded"
    PLUGIN_ERROR = "plugin.error"

    # Memory events
    MEMORY_STORED = "memory.stored"
    MEMORY_RECALL = "memory.recall"


@dataclass(frozen=True)
class Event:
    """System event."""
    type: EventType | str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    timestamp: float = field(default_factory=lambda: __import__("time").time())


T = TypeVar("T")

EventHandler = Callable[[Event], Any]


# ============================================================================
# Plugin Interface
# ============================================================================

class PluginMetadata(BaseModel):
    """Plugin metadata."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: list[str] = Field(default_factory=list)
    optional_dependencies: list[str] = Field(default_factory=list)
    entry_point: str = ""
    enabled_by_default: bool = True


@runtime_checkable
class Plugin(Protocol):
    """Interface for system plugins."""

    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata."""
        ...

    async def initialize(self, context: PluginContext) -> None:
        """Initialize plugin with system context."""
        ...

    async def shutdown(self) -> None:
        """Cleanup plugin resources."""
        ...


@runtime_checkable
class PluginContext(Protocol):
    """Context provided to plugins during initialization."""

    @property
    def event_bus(self) -> EventBus:
        """System event bus."""
        ...

    def register_tool(self, tool: Tool) -> None:
        """Register tool in system."""
        ...

    def get_provider(self) -> LLMProvider:
        """Get configured LLM provider."""
        ...


# ============================================================================
# Transport Interface (Telegram, Web, CLI)
# ============================================================================

@runtime_checkable
class Transport(Protocol):
    """Interface for message transports."""

    @property
    def name(self) -> str:
        """Transport name."""
        ...

    @property
    def is_running(self) -> bool:
        """Check if transport is active."""
        ...

    async def start(self) -> None:
        """Start transport."""
        ...

    async def stop(self) -> None:
        """Stop transport gracefully."""
        ...

    async def send_message(self, chat_id: str, content: str, **kwargs: Any) -> None:
        """Send message via transport."""
        ...


# ============================================================================
# Configuration
# ============================================================================

class AgentConfig(BaseModel):
    """Main agent configuration."""
    # LLM settings
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    llm_base_url: str = "http://localhost:11434"
    llm_timeout: float = 120.0

    # Memory settings
    memory_backend: str = "sqlite"
    memory_max_entries: int = 10000

    # Plugin settings
    plugin_dirs: list[str] = Field(default_factory=lambda: ["plugins"])
    disabled_plugins: list[str] = Field(default_factory=list)

    # Logging
    log_level: str = "INFO"
    log_format: str = "structured"

    model_config = {"env_prefix": "AGENT_", "env_file": ".env"}
