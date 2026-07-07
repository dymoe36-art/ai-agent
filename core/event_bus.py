"""Event bus implementation - central communication system.

All modules communicate through events, never directly.
This ensures complete decoupling between components.
"""

from __future__ import annotations

import asyncio
import weakref
from collections import defaultdict
from typing import Any, Awaitable, Callable

from core.interfaces import Event, EventHandler, EventType


class EventBus:
    """Asynchronous event bus with wildcard support.

    All inter-module communication happens through the event bus.
    Modules don't know about each other - they only publish and subscribe to events.

    Example:
        bus = EventBus()

        # Subscribe to specific event
        @bus.on(EventType.MESSAGE_RECEIVED)
        async def handler(event):
            print(f"Got message: {event.payload}")

        # Subscribe to pattern (all llm.* events)
        @bus.on("llm.*")
        async def llm_monitor(event):
            print(f"LLM event: {event.type}")

        # Publish event
        await bus.emit(EventType.MESSAGE_RECEIVED, {"text": "Hello"})
    """

    def __init__(self) -> None:
        self._handlers: dict[str, set[EventHandler]] = defaultdict(set)
        self._wildcards: set[Callable[[Event], Awaitable[None]]] = set()
        self._history: list[Event] = []
        self._max_history = 1000
        self._lock = asyncio.Lock()
        self._running = True

    def on(
        self,
        event_type: EventType | str,
    ) -> Callable[[EventHandler], EventHandler]:
        """Decorator to subscribe to events.

        Args:
            event_type: Event type or pattern (e.g., "llm.*")
        """
        def decorator(handler: EventHandler) -> EventHandler:
            self.subscribe(event_type, handler)
            return handler
        return decorator

    def subscribe(
        self,
        event_type: EventType | str,
        handler: EventHandler,
    ) -> None:
        """Subscribe handler to event type or pattern.

        Args:
            event_type: Specific event or pattern with * wildcard
            handler: Callback function
        """
        type_str = event_type.value if isinstance(event_type, EventType) else event_type

        if "*" in type_str:
            self._wildcards.add(handler)
        else:
            self._handlers[type_str].add(handler)

    def unsubscribe(
        self,
        event_type: EventType | str,
        handler: EventHandler,
    ) -> None:
        """Unsubscribe handler from event."""
        type_str = event_type.value if isinstance(event_type, EventType) else event_type
        self._handlers[type_str].discard(handler)
        self._wildcards.discard(handler)

    async def emit(self, event_type: EventType | str, payload: dict[str, Any] | None = None, *, source: str = "unknown") -> None:
        """Emit event to all subscribers.

        Args:
            event_type: Type of event
            payload: Event data
            source: Module that emitted the event
        """
        if not self._running:
            return

        event = Event(
            type=event_type.value if isinstance(event_type, EventType) else event_type,
            payload=payload or {},
            source=source,
        )

        # Store in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Collect matching handlers
        handlers: set[EventHandler] = set()
        handlers.update(self._handlers.get(event.type, set()))

        # Check wildcard subscribers
        type_prefix = event.type.split(".")[0] if "." in event.type else event.type
        for wildcard_handler in self._wildcards:
            handlers.add(wildcard_handler)

        # Execute handlers concurrently with error isolation
        if handlers:
            await asyncio.gather(
                *[self._safe_execute(h, event) for h in handlers],
                return_exceptions=True,
            )

    async def _safe_execute(self, handler: EventHandler, event: Event) -> None:
        """Execute handler with error isolation.

        Errors in one handler don't affect others.
        """
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            # Log but don't propagate - isolation is critical
            import logging
            logging.getLogger("event_bus").exception(
                "Handler failed for event %s from %s",
                event.type,
                event.source,
            )

    def get_history(
        self,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Get event history, optionally filtered.

        Args:
            event_type: Filter by event type
            limit: Maximum number of events
        Returns:
            List of matching events
        """
        events = self._history
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    async def shutdown(self) -> None:
        """Shutdown event bus gracefully."""
        self._running = False
        async with self._lock:
            self._handlers.clear()
            self._wildcards.clear()

    def create_child(self, prefix: str) -> "PrefixedEventBus":
        """Create child event bus with automatic prefix.

        Useful for plugins to namespace their events.
        """
        return PrefixedEventBus(self, prefix)


class PrefixedEventBus:
    """Wrapper that automatically prefixes all events.

    Plugins use this to avoid event name collisions.
    """

    def __init__(self, parent: EventBus, prefix: str) -> None:
        self._parent = parent
        self._prefix = prefix

    def on(self, event_type: str) -> Callable[[EventHandler], EventHandler]:
        """Subscribe to prefixed event."""
        return self._parent.on(f"{self._prefix}.{event_type}")

    async def emit(self, event_type: str, payload: dict[str, Any] | None = None, *, source: str = "") -> None:
        """Emit prefixed event."""
        await self._parent.emit(
            f"{self._prefix}.{event_type}",
            payload,
            source=source or self._prefix,
        )

    async def shutdown(self) -> None:
        """No-op - shutdown is managed by parent."""
        pass
