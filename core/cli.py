"""Command-line interface for the agent.

Entry point: ai-agent
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

from core.agent import AgentCore
from core.interfaces import AgentConfig


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    try:
        import structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if "json" in level else structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    except ImportError:
        # Fallback to standard logging
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )


async def run_agent(config: AgentConfig) -> None:
    """Run agent with graceful shutdown."""
    agent = AgentCore(config)

    def signal_handler(sig: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger = logging.getLogger(__name__)
        logger.info("Received signal %s, shutting down...", sig)
        asyncio.create_task(agent.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await agent.start()
    except Exception as e:
        logging.getLogger(__name__).exception("Agent failed")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AI Agent")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2",
        help="Ollama model name",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="sqlite",
        choices=["sqlite", "memory", "none"],
        help="Memory backend",
    )
    parser.add_argument(
        "--plugin-dir",
        type=str,
        action="append",
        default=["plugins"],
        help="Plugin directories",
    )
    parser.add_argument(
        "--telegram-token",
        type=str,
        default="",
        help="Telegram bot token (enables Telegram transport)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    # Build configuration
    config = AgentConfig(
        llm_model=args.model,
        llm_base_url=args.ollama_url,
        log_level=args.log_level,
        memory_backend=args.memory,
        plugin_dirs=args.plugin_dir,
    )

    # Override from config file if provided
    if args.config and Path(args.config).exists():
        import json
        with open(args.config) as f:
            file_config = json.load(f)
        for key, value in file_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Run
    asyncio.run(run_agent(config))


if __name__ == "__main__":
    main()
