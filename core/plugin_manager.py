"""Plugin management system with hot-loading support.

Plugins are self-contained modules that extend agent functionality.
Each plugin runs in isolation and communicates only via EventBus.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from core.event_bus import EventBus, PrefixedEventBus
from core.interfaces import (
    EventType,
    Plugin,
    PluginContext,
    PluginMetadata,
    Tool,
)
from core.registry import ToolRegistry

logger = logging.getLogger(__name__)


class PluginContextImpl:
    """Implementation of PluginContext interface.

    Provides controlled access to system resources for plugins.
    """

    def __init__(
        self,
        plugin_name: str,
        event_bus: EventBus,
        tool_registry: ToolRegistry,
        config: dict[str, Any],
    ) -> None:
        self._name = plugin_name
        self._event_bus = event_bus.create_child(plugin_name)
        self._tool_registry = tool_registry
        self._config = config
        self._tools: list[str] = []
        self._logger = logging.getLogger(f"plugin.{plugin_name}")

    @property
    def event_bus(self) -> PrefixedEventBus:
        """Plugin's namespaced event bus."""
        return self._event_bus

    @property
    def logger(self) -> logging.Logger:
        """Plugin logger."""
        return self._logger

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get plugin-specific configuration value."""
        return self._config.get(key, default)

    def register_tool(self, tool: Tool) -> None:
        """Register tool in system.

        Tool is automatically unregistered when plugin is unloaded.
        """
        self._tool_registry.register(tool)
        self._tools.append(tool.definition.name)
        self._logger.info("Registered tool: %s", tool.definition.name)

    def unregister_tool(self, tool_name: str) -> None:
        """Unregister tool by name."""
        self._tool_registry.unregister(tool_name)
        self._tools = [t for t in self._tools if t != tool_name]


class PluginManager:
    """Manages plugin lifecycle: discovery, loading, and unloading.

    Each plugin is loaded in isolation with its own:
    - Namespaced event bus (no event collisions)
    - Tool namespace (tools prefixed with plugin name)
    - Isolated logger
    - Dedicated configuration

    Example:
        manager = PluginManager(event_bus, tool_registry)
        await manager.discover("plugins/")
        await manager.load_all()

        # Hot-reload single plugin
        await manager.reload("my_plugin")
    """

    def __init__(
        self,
        event_bus: EventBus,
        tool_registry: ToolRegistry,
        plugin_config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._plugin_config = plugin_config or {}
        self._plugins: dict[str, Plugin] = {}
        self._contexts: dict[str, PluginContextImpl] = {}
        self._modules: dict[str, ModuleType] = {}
        self._plugin_dirs: list[Path] = []
        self._discovered: dict[str, Path] = {}

    async def discover(self, *plugin_dirs: str | Path) -> list[PluginMetadata]:
        """Discover available plugins in directories.

        Scans directories for valid plugin modules.
        Does not load plugins yet.

        Args:
            plugin_dirs: Directories to scan for plugins

        Returns:
            List of discovered plugin metadata
        """
        self._plugin_dirs = [Path(d) for d in plugin_dirs]
        discovered: list[PluginMetadata] = []

        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                logger.warning("Plugin directory not found: %s", plugin_dir)
                continue

            for item in plugin_dir.iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    metadata = await self._inspect_plugin(item)
                    if metadata:
                        self._discovered[metadata.name] = item
                        discovered.append(metadata)
                        logger.debug("Discovered plugin: %s", metadata.name)

        await self._event_bus.emit(
            EventType.PLUGIN_LOADED,
            {"plugins": [m.name for m in discovered], "count": len(discovered)},
            source="plugin_manager",
        )

        return discovered

    async def _inspect_plugin(self, plugin_path: Path) -> PluginMetadata | None:
        """Inspect plugin directory without loading it."""
        try:
            # Look for plugin.yaml or metadata in __init__.py
            metadata_file = plugin_path / "plugin.yaml"
            if metadata_file.exists():
                import yaml
                with open(metadata_file) as f:
                    data = yaml.safe_load(f)
                return PluginMetadata(**data)

            # Try to get metadata from module without full import
            init_file = plugin_path / "__init__.py"
            spec = importlib.util.spec_from_file_location(
                plugin_path.name,
                init_file,
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "METADATA"):
                    return PluginMetadata(**module.METADATA)

        except Exception as e:
            logger.warning("Failed to inspect plugin %s: %s", plugin_path.name, e)

        return None

    def create_context(self, name: str) -> PluginContextImpl:
        """Create isolated context for plugin.

        Args:
            name: Plugin name

        Returns:
            Isolated plugin context
        """
        config = self._plugin_config.get(name, {})
        return PluginContextImpl(
            name,
            self._event_bus,
            self._tool_registry,
            config,
        )

    async def load(self, name: str) -> bool:
        """Load single plugin by name.

        Args:
            name: Plugin name

        Returns:
            True if loaded successfully
        """
        if name in self._plugins:
            logger.warning("Plugin already loaded: %s", name)
            return False

        plugin_path = self._discovered.get(name)
        if not plugin_path:
            logger.error("Plugin not found: %s", name)
            return False

        try:
            # Load module in isolation
            module = await self._load_module(name, plugin_path)
            if not module:
                return False

            # Find Plugin implementation
            plugin_instance = self._find_plugin_class(module)
            if not plugin_instance:
                logger.error("No Plugin class found in %s", name)
                return False

            # Create isolated context
            config = self._plugin_config.get(name, {})
            context = PluginContextImpl(
                name,
                self._event_bus,
                self._tool_registry,
                config,
            )

            # Initialize plugin
            await plugin_instance.initialize(context)

            self._plugins[name] = plugin_instance
            self._contexts[name] = context
            self._modules[name] = module

            logger.info("Loaded plugin: %s", name)
            await self._event_bus.emit(
                EventType.PLUGIN_LOADED,
                {"plugin": name, "version": plugin_instance.metadata.version},
                source="plugin_manager",
            )
            return True

        except Exception as e:
            logger.exception("Failed to load plugin %s: %s", name, e)
            await self._event_bus.emit(
                EventType.PLUGIN_ERROR,
                {"plugin": name, "error": str(e)},
                source="plugin_manager",
            )
            return False

    async def load_all(self) -> dict[str, bool]:
        """Load all discovered plugins.

        Returns:
            Dict of plugin name -> success status
        """
        results = {}
        # Load concurrently but with error isolation
        tasks = [self._safe_load(name) for name in self._discovered]
        loaded = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(self._discovered, loaded):
            if isinstance(result, Exception):
                results[name] = False
                logger.error("Plugin %s failed: %s", name, result)
            else:
                results[name] = result

        return results

    async def _safe_load(self, name: str) -> bool:
        """Load plugin with exception isolation."""
        try:
            return await self.load(name)
        except Exception as e:
            logger.exception("Plugin %s load error: %s", name, e)
            return False

    async def unload(self, name: str) -> bool:
        """Unload plugin and cleanup resources.

        Args:
            name: Plugin name

        Returns:
            True if unloaded successfully
        """
        plugin = self._plugins.get(name)
        if not plugin:
            return False

        try:
            # Shutdown plugin
            await plugin.shutdown()

            # Unregister all tools
            context = self._contexts.get(name)
            if context:
                for tool_name in context._tools[:]:  # Copy to avoid mutation
                    context.unregister_tool(tool_name)

            # Remove from sys.modules to allow reload
            module = self._modules.pop(name, None)
            if module and module.__name__ in sys.modules:
                del sys.modules[module.__name__]

            self._plugins.pop(name, None)
            self._contexts.pop(name, None)

            logger.info("Unloaded plugin: %s", name)
            await self._event_bus.emit(
                EventType.PLUGIN_UNLOADED,
                {"plugin": name},
                source="plugin_manager",
            )
            return True

        except Exception as e:
            logger.exception("Error unloading plugin %s: %s", name, e)
            return False

    async def reload(self, name: str) -> bool:
        """Hot-reload single plugin.

        Unloads and re-loads plugin without affecting others.

        Args:
            name: Plugin name

        Returns:
            True if reloaded successfully
        """
        if name in self._plugins:
            await self.unload(name)

        # Force rediscovery
        plugin_path = self._discovered.get(name)
        if plugin_path:
            # Clear import cache for this module
            for mod_name in list(sys.modules.keys()):
                if mod_name.startswith(name):
                    del sys.modules[mod_name]

        return await self.load(name)

    def get(self, name: str) -> Plugin | None:
        """Get loaded plugin by name."""
        return self._plugins.get(name)

    def list_loaded(self) -> list[str]:
        """List all loaded plugin names."""
        return list(self._plugins.keys())

    def list_discovered(self) -> list[str]:
        """List all discovered plugin names."""
        return list(self._discovered.keys())

    async def shutdown(self) -> None:
        """Unload all plugins and cleanup."""
        names = list(self._plugins.keys())
        await asyncio.gather(*[self.unload(n) for n in names], return_exceptions=True)
        self._plugins.clear()
        self._contexts.clear()
        self._modules.clear()

    async def _load_module(self, name: str, path: Path) -> ModuleType | None:
        """Load Python module from path in isolation."""
        try:
            spec = importlib.util.spec_from_file_location(
                name,
                path / "__init__.py",
                submodule_search_locations=[str(path)],
            )
            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        except Exception as e:
            logger.error("Failed to load module %s: %s", name, e)
            # Cleanup on failure
            if name in sys.modules:
                del sys.modules[name]
            return None

    def _find_plugin_class(self, module: ModuleType) -> Any:
        """Find Plugin implementation in module."""
        for _, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                hasattr(obj, "initialize") and
                hasattr(obj, "shutdown") and
                hasattr(obj, "metadata")):
                try:
                    instance = obj()
                    if hasattr(instance, "metadata"):
                        return instance
                except Exception:
                    continue
        return None
