"""Test configuration and fixtures."""

from __future__ import annotations

import pytest
import pytest_asyncio

from core.event_bus import EventBus
from core.interfaces import AgentConfig, Conversation, Role
from core.plugin_manager import PluginManager
from core.registry import ToolRegistry


@pytest.fixture
def event_bus() -> EventBus:
    """Fresh event bus for each test."""
    return EventBus()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Fresh tool registry for each test."""
    return ToolRegistry()


@pytest.fixture
def plugin_manager(event_bus: EventBus, tool_registry: ToolRegistry) -> PluginManager:
    """Fresh plugin manager for each test."""
    return PluginManager(event_bus, tool_registry)


@pytest.fixture
def config() -> AgentConfig:
    """Test configuration."""
    return AgentConfig(
        llm_provider="ollama",
        llm_model="test-model",
        memory_backend="memory",
    )


@pytest.fixture
def sample_conversation() -> Conversation:
    """Sample conversation for testing."""
    conv = Conversation()
    conv.add_message(Role.SYSTEM, "You are a test assistant")
    conv.add_message(Role.USER, "Hello")
    conv.add_message(Role.ASSISTANT, "Hi there!")
    return conv
