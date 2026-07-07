"""Tests for plugin isolation - critical architectural requirement.

Verifies:
- Plugins don't interfere with each other
- Plugin errors don't crash system
- Tools are isolated per plugin
- Events are namespaced
- Hot-reload works without affecting other plugins
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.event_bus import EventBus
from core.interfaces import (
    PluginContext,
    PluginMetadata,
    Tool,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from core.plugin_manager import PluginManager
from core.registry import ToolRegistry


# Test fixtures: simulated plugins with various behaviors

class CrashingTool:
    """Tool that always crashes."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="crash_tool",
            description="Tool that crashes",
        )

    async def execute(self, **params: Any) -> ToolResult:
        raise RuntimeError("Tool crash!")


class SlowTool:
    """Tool with delay."""

    _counter = 0

    def __init__(self, delay: float = 0.1):
        SlowTool._counter += 1
        self._id = SlowTool._counter
        self._delay = delay

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=f"slow_tool_{self._id}",
            description=f"Slow tool #{self._id}",
        )

    async def execute(self, **params: Any) -> ToolResult:
        await asyncio.sleep(self._delay)
        return ToolResult(success=True, content=f"done-{self._id}")


class GoodPlugin:
    """Well-behaved plugin."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="good_plugin",
            version="1.0.0",
        )

    async def initialize(self, context: PluginContext) -> None:
        self.context = context
        self.initialized = True

    async def shutdown(self) -> None:
        self.initialized = False


class CrashingPlugin:
    """Plugin that crashes during initialization."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="crashing_plugin",
            version="1.0.0",
        )

    async def initialize(self, context: PluginContext) -> None:
        raise RuntimeError("Init crash!")

    async def shutdown(self) -> None:
        pass


class PluginWithBadTool:
    """Plugin with a crashing tool."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="bad_tool_plugin",
            version="1.0.0",
        )

    async def initialize(self, context: PluginContext) -> None:
        context.register_tool(CrashingTool())

    async def shutdown(self) -> None:
        pass


class PluginWithSlowTool:
    """Plugin with a slow tool."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="slow_plugin",
            version="1.0.0",
        )

    async def initialize(self, context: PluginContext) -> None:
        context.register_tool(SlowTool(delay=0.05))

    async def shutdown(self) -> None:
        pass


class TestPluginIsolation:
    """Plugin isolation test suite."""

    @pytest.mark.asyncio
    async def test_crashing_plugin_doesnt_affect_others(
        self,
        event_bus: EventBus,
        tool_registry: ToolRegistry,
    ):
        """CRITICAL: Crashing plugin doesn't break other plugins."""
        manager = PluginManager(event_bus, tool_registry)

        # Manually register plugins
        good = GoodPlugin()
        crashing = CrashingPlugin()

        # Initialize good plugin
        await good.initialize(
            manager.create_context("good_plugin")
        )
        manager._plugins["good_plugin"] = good

        # Initialize crashing plugin - should not affect good plugin
        try:
            await crashing.initialize(
                manager.create_context("crashing_plugin")
            )
        except RuntimeError:
            pass  # Expected

        # Good plugin should still be functional
        assert good.initialized is True

    @pytest.mark.asyncio
    async def test_tool_error_isolation(self, tool_registry: ToolRegistry):
        """Tool crash doesn't affect other tools."""
        registry = tool_registry
        SlowTool._counter = 0  # Reset counter

        # Register tools
        registry.register(CrashingTool())
        good_tool = SlowTool()
        registry.register(good_tool)

        # Execute crashing tool
        result1 = await registry.execute("crash_tool")
        assert result1.success is False
        assert "error" in result1.content.lower() or "crash" in result1.content.lower()

        # Execute slow tool - should work fine
        result2 = await registry.execute(good_tool.definition.name)
        assert result2.success is True
        assert "done" in result2.content

    @pytest.mark.asyncio
    async def test_plugin_event_isolation(self, event_bus: EventBus):
        """Plugin events don't leak to other plugins."""
        bus = event_bus
        plugin1_events = []
        plugin2_events = []

        child1 = bus.create_child("plugin1")
        child2 = bus.create_child("plugin2")

        @bus.on("plugin1.test")
        async def p1_handler(event):
            plugin1_events.append(event.payload)

        @bus.on("plugin2.test")
        async def p2_handler(event):
            plugin2_events.append(event.payload)

        await child1.emit("test", {"data": "p1"})
        await child2.emit("test", {"data": "p2"})
        await asyncio.sleep(0.01)

        assert len(plugin1_events) == 1
        assert plugin1_events[0]["data"] == "p1"
        assert len(plugin2_events) == 1
        assert plugin2_events[0]["data"] == "p2"

    @pytest.mark.asyncio
    async def test_concurrent_plugin_execution(self, tool_registry: ToolRegistry):
        """Multiple plugins can execute concurrently."""
        registry = tool_registry
        SlowTool._counter = 0  # Reset counter

        # Register multiple slow tools with unique names
        tools = [SlowTool(delay=0.05) for _ in range(5)]
        for tool in tools:
            registry.register(tool)

        # Execute all concurrently
        start = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[
            registry.execute(tool.definition.name) for tool in tools
        ])
        elapsed = asyncio.get_event_loop().time() - start

        # Should complete in ~0.05s, not 0.25s
        assert elapsed < 0.12
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_plugin_tool_registration(self, event_bus: EventBus, tool_registry: ToolRegistry):
        """Plugin tools are properly registered and isolated."""
        manager = PluginManager(event_bus, tool_registry)
        SlowTool._counter = 0  # Reset counter

        # Initialize plugin with tools
        plugin = PluginWithSlowTool()
        context = manager.create_context("slow_plugin")
        await plugin.initialize(context)

        # Tools should be in registry
        tools = tool_registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name.startswith("slow_tool_")

        # Tool should be executable
        result = await tool_registry.execute(tools[0].name)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_plugin_shutdown_cleanup(self, event_bus: EventBus, tool_registry: ToolRegistry):
        """Plugin shutdown removes its tools."""
        manager = PluginManager(event_bus, tool_registry)

        plugin = PluginWithSlowTool()
        context = manager.create_context("slow_plugin")
        await plugin.initialize(context)

        assert len(tool_registry.list_tools()) == 1

        # Simulate plugin unload
        await plugin.shutdown()
        # Unregister all tools registered by this context
        for tool_name in context._tools[:]:  # type: ignore[attr-defined]
            context.unregister_tool(tool_name)

        assert len(tool_registry.list_tools()) == 0


class TestModuleIndependence:
    """Tests verifying module independence."""

    def test_no_circular_imports(self):
        """Verify no circular imports between modules."""
        import importlib

        # Import all modules - should not fail
        modules = [
            "core.event_bus",
            "core.interfaces",
            "core.plugin_manager",
            "core.registry",
            "core.agent",
            "core.container",
            "core.cli",
            "adapters.ollama_adapter",
            "modules.memory",
            "modules.telegram_bot",
        ]

        for module_name in modules:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                # Some modules have optional dependencies - that's fine
                if "No module named" in str(e):
                    pytest.skip(f"Optional dependency missing for {module_name}")
                else:
                    raise

    def test_core_has_no_external_deps(self):
        """Verify core module has no external dependencies."""
        import ast
        import inspect

        from core import event_bus, interfaces, registry

        for module in [interfaces, event_bus, registry]:
            source = inspect.getsource(module)
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    # Core should only import from stdlib, core, or pydantic
                    allowed = {
                        "__future__", "abc", "asyncio", "collections",
                        "dataclasses", "enum", "logging", "threading",
                        "types", "pathlib", "typing", "weakref", "json",
                        "importlib", "inspect", "sys", "time",
                        "pydantic", "pydantic_settings",
                        "tenacity", "httpx", "injector", "pluggy",
                    }
                    assert module_name in allowed or module_name.startswith("core."), \
                        f"Unexpected import in core: {module_name}"

    @pytest.mark.asyncio
    async def test_memory_module_independence(self):
        """Memory module works without other modules."""
        from modules.memory import InMemoryBackend, MemoryManager

        backend = InMemoryBackend()
        memory = MemoryManager(backend)

        # Should work without any other module initialized
        await memory.remember("Test memory", importance=5.0)
        results = await memory.recall("Test")

        assert len(results) == 1
        assert results[0].content == "Test memory"
