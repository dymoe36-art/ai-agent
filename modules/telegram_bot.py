"""Telegram bot module - completely isolated from other modules.

This module only depends on:
- python-telegram-bot (optional dependency group: telegram)
- core interfaces (EventBus, Conversation, etc.)
- adapters (via DI container, not direct import)

No other module depends on this one. Removing it doesn't break anything.
"""

from __future__ import annotations

import logging
from typing import Any

from core.event_bus import EventBus
from core.interfaces import (
    Conversation,
    EventType,
    Message,
    Role,
    Transport,
)

logger = logging.getLogger(__name__)

# Graceful import - module works even if python-telegram-bot not installed
try:
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Telegram module disabled.")


class TelegramTransport:
    """Telegram bot transport implementation.

    Completely self-contained. Uses DI to get LLM provider and EventBus.

    Example:
        transport = TelegramTransport(token="...", event_bus=bus)
        await transport.start()  # Runs until stopped
    """

    def __init__(
        self,
        token: str,
        event_bus: EventBus,
        llm_provider_getter: Any | None = None,
        allowed_users: list[int] | None = None,
    ) -> None:
        if not TELEGRAM_AVAILABLE:
            raise RuntimeError(
                "python-telegram-bot not installed. "
                "Install with: pip install ai-agent[telegram]"
            )

        self._token = token
        self._event_bus = event_bus
        self._get_llm = llm_provider_getter
        self._allowed_users = set(allowed_users) if allowed_users else None
        self._application: Application | None = None
        self._running = False

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start Telegram bot."""
        if self._running:
            return

        self._application = (
            Application.builder()
            .token(self._token)
            .build()
        )

        # Register handlers
        self._application.add_handler(
            CommandHandler("start", self._cmd_start)
        )
        self._application.add_handler(
            CommandHandler("help", self._cmd_help)
        )
        self._application.add_handler(
            CommandHandler("clear", self._cmd_clear)
        )
        self._application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        # Error handler with isolation
        self._application.add_error_handler(self._handle_error)

        self._running = True
        logger.info("Telegram bot starting...")

        await self._event_bus.emit(
            EventType.SYSTEM_STARTUP,
            {"transport": "telegram"},
            source="telegram",
        )

        await self._application.initialize()
        await self._application.start()
        await self._application.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        """Stop Telegram bot gracefully."""
        if not self._running or not self._application:
            return

        logger.info("Telegram bot stopping...")
        self._running = False

        if self._application.updater.running:
            await self._application.updater.stop()
        await self._application.stop()
        await self._application.shutdown()

        await self._event_bus.emit(
            EventType.SYSTEM_SHUTDOWN,
            {"transport": "telegram"},
            source="telegram",
        )

    async def send_message(self, chat_id: str, content: str, **kwargs: Any) -> None:
        """Send message to chat."""
        if self._application:
            await self._application.bot.send_message(
                chat_id=int(chat_id),
                text=content,
                **kwargs,
            )

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not self._check_access(update):
            return

        await update.message.reply_text(
            "🤖 Привет! Я AI-агент с автономными возможностями.\n"
            "Отправь мне сообщение или используй /help для списка команд."
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        if not self._check_access(update):
            return

        help_text = (
            "📋 Доступные команды:\n"
            "/start - Начать работу\n"
            "/help - Показать помощь\n"
            "/clear - Очистить историю\n"
            "\nПросто отправь сообщение, и я отвечу!"
        )
        await update.message.reply_text(help_text)

    async def _cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /clear command."""
        if not self._check_access(update):
            return

        chat_id = str(update.effective_chat.id)
        if chat_id in self._conversations:
            del self._conversations[chat_id]

        await update.message.reply_text("🗑 История диалога очищена.")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text message."""
        if not self._check_access(update):
            return

        user_message = update.message.text
        chat_id = str(update.effective_chat.id)
        user = update.effective_user.username or update.effective_user.first_name

        # Emit event - core agent handles the logic
        await self._event_bus.emit(
            EventType.MESSAGE_RECEIVED,
            {
                "chat_id": chat_id,
                "user": user,
                "text": user_message,
                "transport": "telegram",
            },
            source="telegram",
        )

        # If LLM provider available, generate response
        if self._get_llm:
            try:
                llm = self._get_llm()
                conversation = self._get_or_create_conversation(chat_id)
                conversation.add_message(Role.USER, user_message)

                # Show typing indicator
                await context.bot.send_chat_action(
                    chat_id=update.effective_chat.id,
                    action="typing",
                )

                response = await llm.generate(conversation)
                conversation.add_message(Role.ASSISTANT, response.content)

                await update.message.reply_text(response.content)

                await self._event_bus.emit(
                    EventType.MESSAGE_SENT,
                    {
                        "chat_id": chat_id,
                        "text": response.content,
                        "transport": "telegram",
                    },
                    source="telegram",
                )

            except Exception as e:
                logger.exception("Error processing message")
                await update.message.reply_text(
                    f"❌ Ошибка: {type(e).__name__}\n"
                    "Попробуйте позже или обратитесь к администратору."
                )
                await self._event_bus.emit(
                    EventType.SYSTEM_ERROR,
                    {"error": str(e), "module": "telegram"},
                    source="telegram",
                )

    def _check_access(self, update: Update) -> bool:
        """Check if user is allowed to use the bot."""
        if self._allowed_users is None:
            return True
        return update.effective_user.id in self._allowed_users

    def _get_or_create_conversation(self, chat_id: str) -> Conversation:
        """Get or create conversation for chat."""
        if not hasattr(self, "_conversations"):
            self._conversations: dict[str, Conversation] = {}

        if chat_id not in self._conversations:
            self._conversations[chat_id] = Conversation()
        return self._conversations[chat_id]

    async def _handle_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in handlers with isolation."""
        logger.exception("Telegram handler error")

        if update and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "⚠️ Произошла ошибка. Попробуйте ещё раз."
                )
            except Exception:
                pass

        await self._event_bus.emit(
            EventType.SYSTEM_ERROR,
            {
                "error": str(context.error),
                "module": "telegram",
                "update_id": update.update_id if update else None,
            },
            source="telegram",
        )
