"""Tool registry for managing agent capabilities.

Tools are self-contained functions that the agent can call.
Each tool is isolated and handles its own errors.
"""

from __future__ import annotations

import logging
from typing import Any

from core.interfaces import Tool, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all agent tools.

    Tools are registered by plugins or core modules.
    Each tool runs in isolation - errors don't affect other tools.

    Example:
        registry = ToolRegistry()
        registry.register(MyTool())

        result = await registry.execute("my_tool", param="value")
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, tool: Tool) -> None:
        """Register tool in registry.

        Args:
            tool: Tool implementation

        Raises:
            ValueError: If tool with same name already registered
        """
        definition = tool.definition
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")

        self._tools[definition.name] = tool
        self._definitions[definition.name] = definition
        logger.debug("Registered tool: %s", definition.name)

    def unregister(self, name: str) -> bool:
        """Unregister tool by name.

        Args:
            name: Tool name

        Returns:
            True if tool was removed
        """
        if name in self._tools:
            del self._tools[name]
            del self._definitions[name]
            logger.debug("Unregistered tool: %s", name)
            return True
        return False

    def get(self, name: str) -> Tool | None:
        """Get tool by name."""
        return self._tools.get(name)

    def get_definition(self, name: str) -> ToolDefinition | None:
        """Get tool definition by name."""
        return self._definitions.get(name)

    def list_tools(self, category: str | None = None) -> list[ToolDefinition]:
        """List all registered tool definitions.

        Args:
            category: Filter by category

        Returns:
            List of tool definitions
        """
        tools = list(self._definitions.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    async def execute(self, name: str, **params: Any) -> ToolResult:
        """Execute tool by name with parameters.

        Args:
            name: Tool name
            **params: Tool parameters

        Returns:
            Tool execution result

        Raises:
            KeyError: If tool not found
        """
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"Tool not found: {name}")

        # Execute with error isolation
        try:
            result = await tool.execute(**params)
            if not isinstance(result, ToolResult):
                result = ToolResult(success=True, content=str(result))
            return result
        except Exception as e:
            logger.exception("Tool %s execution failed", name)
            return ToolResult(
                success=False,
                content=f"Error: {type(e).__name__}: {e}",
                metadata={"error_type": type(e).__name__},
            )

    def get_schema_for_llm(self) -> list[dict[str, Any]]:
        """Get tool schemas formatted for LLM function calling.

        Returns:
            List of tool schemas in OpenAI-compatible format
        """
        schemas = []
        for definition in self._definitions.values():
            schema = {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            p.name: {
                                "type": p.type,
                                "description": p.description,
                            }
                            for p in definition.parameters
                        },
                        "required": [
                            p.name for p in definition.parameters if p.required
                        ],
                    },
                },
            }
            schemas.append(schema)
        return schemas

    def clear(self) -> None:
        """Remove all tools. Used for testing."""
        self._tools.clear()
        self._definitions.clear()
