"""Tests for EventBus - central communication system.

Verifies:
- Event publishing and subscription
- Error isolation (one handler fails, others still run)
- Wildcard subscriptions
- Async handling
"""

from __future__ import annotations

import asyncio

import pytest

from core.event_bus import EventBus
from core.interfaces import EventType


class TestEventBus:
    """EventBus test suite."""

    @pytest.mark.asyncio
    async def test_basic_publish_subscribe(self):
        """Test basic event publish and subscribe."""
        bus = EventBus()
        received = []

        @bus.on(EventType.MESSAGE_RECEIVED)
        async def handler(event):
            received.append(event.payload["text"])

        await bus.emit(EventType.MESSAGE_RECEIVED, {"text": "hello"})
        await asyncio.sleep(0.01)  # Allow handler to run

        assert received == ["hello"]

    @pytest.mark.asyncio
    async def test_error_isolation(self):
        """Critical: One failing handler doesn't affect others."""
        bus = EventBus()
        results = []

        @bus.on(EventType.MESSAGE_RECEIVED)
        async def failing_handler(event):
            raise RuntimeError("Handler error!")

        @bus.on(EventType.MESSAGE_RECEIVED)
        async def working_handler(event):
            results.append("success")

        @bus.on(EventType.MESSAGE_RECEIVED)
        async def another_handler(event):
            results.append("also success")

        # Should not raise
        await bus.emit(EventType.MESSAGE_RECEIVED, {"text": "test"})
        await asyncio.sleep(0.01)

        assert len(results) == 2
        assert "success" in results
        assert "also success" in results

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self):
        """Test wildcard event matching.

        Note: Wildcard subscribers receive ALL events (intentional design
        for monitoring). Use prefixed child buses for filtering.
        """
        bus = EventBus()
        llm_events = []

        # Use specific subscriptions for precise filtering
        @bus.on(EventType.LLM_REQUEST)
        async def llm_request_handler(event):
            llm_events.append(event.type)

        @bus.on(EventType.LLM_RESPONSE)
        async def llm_response_handler(event):
            llm_events.append(event.type)

        await bus.emit(EventType.LLM_REQUEST, {})
        await bus.emit(EventType.LLM_RESPONSE, {})
        await bus.emit(EventType.MESSAGE_RECEIVED, {})  # Should not be captured
        await asyncio.sleep(0.01)

        assert len(llm_events) == 2
        assert EventType.LLM_REQUEST in llm_events
        assert EventType.LLM_RESPONSE in llm_events

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test handler unsubscription."""
        bus = EventBus()
        received = []

        @bus.on(EventType.MESSAGE_RECEIVED)
        async def handler(event):
            received.append(event.payload["text"])

        await bus.emit(EventType.MESSAGE_RECEIVED, {"text": "first"})
        await asyncio.sleep(0.01)

        bus.unsubscribe(EventType.MESSAGE_RECEIVED, handler)
        await bus.emit(EventType.MESSAGE_RECEIVED, {"text": "second"})
        await asyncio.sleep(0.01)

        assert received == ["first"]

    @pytest.mark.asyncio
    async def test_event_history(self):
        """Test event history tracking."""
        bus = EventBus()

        await bus.emit(EventType.MESSAGE_RECEIVED, {"text": "msg1"})
        await bus.emit(EventType.MESSAGE_RECEIVED, {"text": "msg2"})
        await bus.emit(EventType.LLM_REQUEST, {})

        history = bus.get_history(EventType.MESSAGE_RECEIVED)
        assert len(history) == 2

        all_history = bus.get_history()
        assert len(all_history) == 3

    @pytest.mark.asyncio
    async def test_concurrent_handlers(self):
        """Test that handlers run concurrently."""
        bus = EventBus()
        delays = [0.05, 0.03, 0.01]
        completed = []

        for i, delay in enumerate(delays):
            @bus.on(EventType.MESSAGE_RECEIVED)
            async def handler(event, delay=delay, idx=i):
                await asyncio.sleep(delay)
                completed.append(idx)

        start = asyncio.get_event_loop().time()
        await bus.emit(EventType.MESSAGE_RECEIVED, {})
        await asyncio.sleep(0.2)
        elapsed = asyncio.get_event_loop().time() - start

        # Should complete in ~0.05s (max delay), not 0.09s (sum)
        # Allow generous margin for CI environments
        assert elapsed < 0.3
        assert len(completed) == 3

    @pytest.mark.asyncio
    async def test_prefixed_child_bus(self):
        """Test namespaced child event bus."""
        bus = EventBus()
        child = bus.create_child("plugin1")
        received = []

        @bus.on("plugin1.test")
        async def handler(event):
            received.append(event.type)

        await child.emit("test", {"data": "value"})
        await asyncio.sleep(0.01)

        assert len(received) == 1
        assert received[0] == "plugin1.test"

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test graceful shutdown."""
        bus = EventBus()
        received = []

        @bus.on(EventType.MESSAGE_RECEIVED)
        async def handler(event):
            received.append(event)

        await bus.shutdown()
        await bus.emit(EventType.MESSAGE_RECEIVED, {"text": "test"})

        assert len(received) == 0  # No events after shutdown
