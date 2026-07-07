"""AgentCore - central coordinator.

Does not contain business logic. Only orchestrates components via EventBus.
All features come from plugins and modules.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.event_bus import EventBus
from core.interfaces import (
    AgentConfig,
    Conversation,
    EventType,
    LLMProvider,
    Message,
    Role,
)
from core.plugin_manager import PluginManager
from core.registry import ToolRegistry
from modules.memory import MemoryManager

logger = logging.getLogger(__name__)


class AgentCore:
    """Central agent coordinator.

    Responsibilities:
    - Initialize and connect all components
    - Route messages between transport and LLM
    - Handle tool calls from LLM
    - Coordinate plugin lifecycle

    Does NOT contain:
    - Telegram-specific code
    - LLM implementation details
    - Tool implementations
    - Business logic

    Example:
        agent = AgentCore(config)
        await agent.start()
        # Runs until shutdown
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus | None = None,
        tool_registry: ToolRegistry | None = None,
        plugin_manager: PluginManager | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus or EventBus()
        self._tool_registry = tool_registry or ToolRegistry()
        self._plugin_manager = plugin_manager or PluginManager(
            self._event_bus,
            self._tool_registry,
        )
        self._llm_provider: LLMProvider | None = None
        self._memory: MemoryManager | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Setup event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        """Subscribe to system events."""
        self._event_bus.on(EventType.MESSAGE_RECEIVED)(self._on_message_received)
        self._event_bus.on(EventType.TOOL_CALLED)(self._on_tool_called)
        self._event_bus.on(EventType.SYSTEM_ERROR)(self._on_system_error)

    async def start(self) -> None:
        """Start agent and all components.

        1. Initialize LLM provider
        2. Initialize memory
        3. Discover and load plugins
        4. Wait for shutdown signal
        """
        if self._running:
            return

        self._running = True
        logger.info("AgentCore starting...")

        await self._event_bus.emit(
            EventType.SYSTEM_STARTUP,
            {"version": "0.1.0"},
            source="agent_core",
        )

        # Initialize LLM provider
        await self._init_llm()

        # Initialize memory if configured
        await self._init_memory()

        # Discover and load plugins
        await self._plugin_manager.discover(*self._config.plugin_dirs)
        await self._plugin_manager.load_all()

        logger.info(
            "AgentCore ready. Plugins: %s, Tools: %s",
            self._plugin_manager.list_loaded(),
            [t.name for t in self._tool_registry.list_tools()],
        )

        # Wait for shutdown
        await self._shutdown_event.wait()

    async def shutdown(self) -> None:
        """Shutdown agent gracefully."""
        if not self._running:
            return

        logger.info("AgentCore shutting down...")
        self._running = False

        # Unload all plugins
        await self._plugin_manager.shutdown()

        # Cleanup LLM provider
        if self._llm_provider and hasattr(self._llm_provider, "close"):
            await self._llm_provider.close()

        # Cleanup memory
        if self._memory:
            await self._memory.close()

        # Shutdown event bus
        await self._event_bus.shutdown()

        self._shutdown_event.set()
        logger.info("AgentCore shutdown complete")

    async def process_message(
        self,
        user_id: str,
        message: str,
        transport: str = "unknown",
    ) -> str:
        """Process user message and return response.

        This is the main entry point for message processing.

        Args:
            user_id: User identifier
            message: User message
            transport: Transport name

        Returns:
            Assistant response text
        """
        if not self._llm_provider:
            return "❌ LLM provider not available"

        try:
            # Build context with memory
            if self._memory:
                conversation = await self._memory.build_context(user_id, message)
            else:
                conversation = Conversation()
                conversation.add_message(Role.USER, message)

            # Check if tool call is needed
            tools = self._tool_registry.get_schema_for_llm()
            if tools:
                # Generate with tool support
                response = await self._generate_with_tools(conversation, tools)
            else:
                # Simple generation
                response = await self._llm_provider.generate(conversation)

            # Store interaction in memory
            if self._memory:
                await self._memory.store_interaction(
                    user_id, message, response.content,
                )

            # Emit events
            await self._event_bus.emit(
                EventType.LLM_RESPONSE,
                {"user_id": user_id, "response": response.content},
                source="agent_core",
            )

            return response.content

        except Exception as e:
            logger.exception("Message processing failed")
            await self._event_bus.emit(
                EventType.LLM_ERROR,
                {"error": str(e), "user_id": user_id},
                source="agent_core",
            )
            return f"❌ Ошибка обработки: {type(e).__name__}"

    async def _generate_with_tools(
        self,
        conversation: Conversation,
        tools: list[dict[str, Any]],
    ) -> Any:
        """Generate response with potential tool calls.

        Current implementation does a simple generation.
        Full implementation would parse tool calls from response.
        """
        # For now, simple generation without tool calling
        # Tool calling can be added via plugin
        return await self._llm_provider.generate(conversation)

    async def _init_llm(self) -> None:
        """Initialize LLM provider based on configuration."""
        provider_name = self._config.llm_provider

        if provider_name == "ollama":
            from adapters.ollama_adapter import OllamaProvider
            self._llm_provider = OllamaProvider(
                model=self._config.llm_model,
                base_url=self._config.llm_base_url,
                timeout=self._config.llm_timeout,
            )

            if not self._llm_provider.is_available:
                logger.warning(
                    "Ollama not available at %s. "
                    "Make sure Ollama is running.",
                    self._config.llm_base_url,
                )
            else:
                logger.info(
                    "Ollama provider ready: %s",
                    self._llm_provider.name,
                )
        else:
            logger.error("Unknown LLM provider: %s", provider_name)

    async def _init_memory(self) -> None:
        """Initialize memory backend."""
        backend_name = self._config.memory_backend

        if backend_name == "sqlite":
            from modules.memory import MemoryManager, SQLiteMemoryBackend
            backend = SQLiteMemoryBackend("data/memory.db")
            self._memory = MemoryManager(backend)
            logger.info("SQLite memory initialized")
        elif backend_name == "memory":
            from modules.memory import InMemoryBackend, MemoryManager
            backend = InMemoryBackend()
            self._memory = MemoryManager(backend)
            logger.info("In-memory backend initialized")
        else:
            logger.warning("Unknown memory backend: %s", backend_name)

    async def _on_message_received(self, event: Any) -> None:
        """Handle incoming message event."""
        payload = event.payload
        chat_id = payload.get("chat_id", "unknown")
        text = payload.get("text", "")
        transport = payload.get("transport", "unknown")

        logger.debug("Message from %s via %s: %s", chat_id, transport, text)

    async def _on_tool_called(self, event: Any) -> None:
        """Handle tool call event."""
        payload = event.payload
        tool_name = payload.get("tool")
        params = payload.get("params", {})

        try:
            result = await self._tool_registry.execute(tool_name, **params)
            await self._event_bus.emit(
                EventType.TOOL_COMPLETED,
                {
                    "tool": tool_name,
                    "result": result.content,
                    "success": result.success,
                },
                source="agent_core",
            )
        except Exception as e:
            logger.exception("Tool execution failed: %s", tool_name)
            await self._event_bus.emit(
                EventType.TOOL_ERROR,
                {"tool": tool_name, "error": str(e)},
                source="agent_core",
            )

    async def _on_system_error(self, event: Any) -> None:
        """Handle system error with isolation."""
        payload = event.payload
        error = payload.get("error", "unknown")
        module = payload.get("module", "unknown")

        logger.error("System error in %s: %s", module, error)
        # Don't propagate - error is contained in the module

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def plugin_manager(self) -> PluginManager:
        return self._plugin_manager

    @property
    def llm_provider(self) -> LLMProvider | None:
        return self._llm_provider
