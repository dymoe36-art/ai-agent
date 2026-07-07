"""Dependency Injection container.

Uses injector library for clean dependency management.
All components are wired together here without hard dependencies.
"""

from __future__ import annotations

import logging
from typing import Any

from injector import Injector, Module, provider, singleton

from core.event_bus import EventBus
from core.interfaces import AgentConfig
from core.plugin_manager import PluginManager
from core.registry import ToolRegistry

logger = logging.getLogger(__name__)


class CoreModule(Module):
    """DI module for core components."""

    @singleton
    @provider
    def provide_event_bus(self) -> EventBus:
        return EventBus()

    @singleton
    @provider
    def provide_tool_registry(self) -> ToolRegistry:
        return ToolRegistry()

    @singleton
    @provider
    def provide_config(self) -> AgentConfig:
        return AgentConfig()

    @singleton
    @provider
    def provide_plugin_manager(
        self,
        event_bus: EventBus,
        tool_registry: ToolRegistry,
    ) -> PluginManager:
        return PluginManager(event_bus, tool_registry)


def create_container(config_overrides: dict[str, Any] | None = None) -> Injector:
    """Create DI container with optional config overrides.

    Args:
        config_overrides: Override default configuration values

    Returns:
        Configured Injector instance
    """
    modules = [CoreModule()]
    container = Injector(modules)

    if config_overrides:
        config = container.get(AgentConfig)
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    logger.debug("DI container created")
    return container


class ServiceLocator:
    """Service locator for accessing dependencies without injection.

    Used in contexts where DI is not practical (e.g., legacy code,
    dynamically loaded plugins).
    """

    _container: Injector | None = None

    @classmethod
    def initialize(cls, container: Injector) -> None:
        """Initialize service locator with container."""
        cls._container = container

    @classmethod
    def get(cls, interface: type[T]) -> T:
        """Get service by interface.

        Args:
            interface: Service interface type

        Returns:
            Service implementation

        Raises:
            RuntimeError: If locator not initialized
        """
        if cls._container is None:
            raise RuntimeError("ServiceLocator not initialized")
        return cls._container.get(interface)

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if locator is ready."""
        return cls._container is not None


from typing import TypeVar
T = TypeVar("T")
