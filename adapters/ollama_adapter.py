"""Ollama LLM provider adapter.

Isolates all Ollama-specific code. Other modules use LLMProvider interface only.
No Ollama imports leak outside this module.

Handles:
- Model availability checking
- Chat completions
- Streaming responses
- Error recovery with tenacity
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.interfaces import (
    Conversation,
    LLMProvider,
    LLMResponse,
    Message,
    Role,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Ollama-specific error."""
    pass


class OllamaProvider:
    """Ollama LLM provider implementation.

    Completely self-contained. No dependencies on other adapters or modules.
    Communicates only through the LLMProvider interface contract.

    Example:
        provider = OllamaProvider(model="llama3.2", base_url="http://localhost:11434")

        if provider.is_available:
            response = await provider.generate(conversation)
            print(response.content)
    """

    DEFAULT_TIMEOUT = 120.0
    MAX_RETRIES = 3

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = DEFAULT_TIMEOUT,
        **options: Any,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._options = options
        self._client: httpx.AsyncClient | None = None
        self._available: bool | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    @property
    def is_available(self) -> bool:
        """Check if Ollama server is reachable and model is loaded.

        Result is cached to avoid repeated health checks.
        """
        if self._available is not None:
            return self._available

        # Async check can't be done in property, so we do sync version
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self._base_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    import json
                    data = json.loads(resp.read())
                    models = [m["name"] for m in data.get("models", [])]
                    self._available = any(
                        self._model in m or m in self._model
                        for m in models
                    )
                    if not self._available:
                        logger.warning(
                            "Model %s not found in Ollama. Available: %s",
                            self._model,
                            models,
                        )
                    return bool(self._available)
        except Exception as e:
            logger.warning("Ollama not available: %s", e)
            self._available = False
            return False

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    @retry(
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate(
        self,
        conversation: Conversation,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion for conversation.

        Args:
            conversation: Conversation context
            **kwargs: Additional generation parameters
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter
                - max_tokens: Maximum tokens to generate

        Returns:
            LLM response with content and metadata

        Raises:
            OllamaError: If generation fails after retries
        """
        client = await self._get_client()
        messages = self._convert_messages(conversation)

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": self._build_options(kwargs),
        }

        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=self._model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
                metadata={
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                },
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error("Ollama API error: %s", error_msg)
            raise OllamaError(error_msg) from e
        except Exception as e:
            logger.exception("Ollama generation failed")
            raise OllamaError(f"Generation failed: {e}") from e

    async def stream(
        self,
        conversation: Conversation,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion chunks.

        Args:
            conversation: Conversation context
            **kwargs: Additional generation parameters

        Yields:
            Stream chunks as they arrive
        """
        client = await self._get_client()
        messages = self._convert_messages(conversation)

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": self._build_options(kwargs),
        }

        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        import json
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        is_done = data.get("done", False)

                        yield StreamChunk(
                            content=content,
                            is_finished=is_done,
                        )

                        if is_done:
                            break

                    except json.JSONDecodeError:
                        logger.warning("Failed to parse stream chunk: %s", line)
                        continue

        except Exception as e:
            logger.exception("Ollama streaming failed")
            yield StreamChunk(
                content=f"\n[Error: {type(e).__name__}]",
                is_finished=True,
            )

    def _convert_messages(self, conversation: Conversation) -> list[dict[str, str]]:
        """Convert internal Conversation format to Ollama format."""
        messages = []
        for msg in conversation.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        return messages

    def _build_options(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Build Ollama options from defaults and overrides."""
        options = {
            "temperature": 0.7,
            "top_p": 0.9,
        }
        options.update(self._options)

        # Map common parameter names
        if "temperature" in overrides:
            options["temperature"] = overrides["temperature"]
        if "top_p" in overrides:
            options["top_p"] = overrides["top_p"]
        if "max_tokens" in overrides:
            options["num_predict"] = overrides["max_tokens"]

        return options

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OllamaProvider":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


class OllamaProviderFactory:
    """Factory for creating Ollama providers.

    Allows switching models without changing consumer code.
    """

    @staticmethod
    def create(
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> OllamaProvider:
        """Create configured Ollama provider.

        Args:
            model: Model name
            base_url: Ollama server URL
            **kwargs: Additional options

        Returns:
            Configured OllamaProvider
        """
        return OllamaProvider(model=model, base_url=base_url, **kwargs)
