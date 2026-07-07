"""Example plugin demonstrating the plugin architecture.

This plugin:
1. Has its own metadata
2. Gets isolated context with namespaced event bus
3. Registers a tool
4. Handles events independently

No dependencies on other plugins or modules.
"""

from __future__ import annotations

import logging
from typing import Any

from core.interfaces import (
    PluginContext,
    Tool,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)

# Plugin metadata - used for discovery
METADATA = {
    "name": "example",
    "version": "1.0.0",
    "description": "Example plugin with echo tool",
    "author": "AI Agent Team",
    "entry_point": "ExamplePlugin",
}


class EchoTool:
    """Simple echo tool for demonstration."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Echo back the input message",
            parameters=[
                ToolParameter(
                    name="message",
                    description="Message to echo",
                    required=True,
                ),
            ],
            category="utility",
        )

    async def execute(self, **params: Any) -> ToolResult:
        message = params.get("message", "")
        return ToolResult(
            success=True,
            content=f"Echo: {message}",
        )


class GreetTool:
    """Greeting tool demonstrating plugin capabilities."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="greet",
            description="Generate a greeting message",
            parameters=[
                ToolParameter(
                    name="name",
                    description="Name to greet",
                    required=False,
                    default="User",
                ),
                ToolParameter(
                    name="language",
                    description="Language (en, ru, es)",
                    required=False,
                    default="en",
                ),
            ],
            category="social",
        )

    async def execute(self, **params: Any) -> ToolResult:
        name = params.get("name", "User")
        language = params.get("language", "en")

        greetings = {
            "en": f"Hello, {name}! 👋",
            "ru": f"Привет, {name}! 👋",
            "es": f"¡Hola, {name}! 👋",
        }

        return ToolResult(
            success=True,
            content=greetings.get(language, greetings["en"]),
        )


class ExamplePlugin:
    """Example plugin implementation.

    Demonstrates:
    - Tool registration
    - Event handling
    - Isolated logging
    - Graceful initialization/shutdown
    """

    @property
    def metadata(self):
        from core.interfaces import PluginMetadata
        return PluginMetadata(**METADATA)

    async def initialize(self, context: PluginContext) -> None:
        """Initialize plugin with system context."""
        logger.info("ExamplePlugin initializing...")

        # Register tools
        context.register_tool(EchoTool())
        context.register_tool(GreetTool())

        # Subscribe to events (automatically namespaced)
        context.event_bus.on("startup")(self._on_startup)
        context.event_bus.on("shutdown")(self._on_shutdown)

        logger.info("ExamplePlugin initialized with 2 tools")

    async def shutdown(self) -> None:
        """Cleanup plugin resources."""
        logger.info("ExamplePlugin shutting down...")

    async def _on_startup(self, event: Any) -> None:
        """Handle startup event."""
        logger.debug("ExamplePlugin received startup event")

    async def _on_shutdown(self, event: Any) -> None:
        """Handle shutdown event."""
        logger.debug("ExamplePlugin received shutdown event")
