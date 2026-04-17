"""Telegram bridge.

Two roles:
1. Push notifications — Jarvis can send warnings / status to a chat.
2. Remote chat — the user can text the bot and it is answered by the same agent.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from jarvis.config import Settings

log = logging.getLogger(__name__)

Responder = Callable[[str], Awaitable[str]]


class TelegramBridge:
    def __init__(self, settings: Settings, responder: Responder) -> None:
        self.settings = settings
        self.responder = responder
        self._app: Application | None = None

    # ----- lifecycle -----

    def enabled(self) -> bool:
        return bool(self.settings.telegram_token)

    async def start(self) -> None:
        if not self.enabled():
            log.info("telegram disabled (no token)")
            return
        self._app = Application.builder().token(self.settings.telegram_token).build()
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("whoami", self._cmd_whoami))
        self._app.add_handler(CommandHandler("ask", self._cmd_ask))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        log.info("telegram bridge started")

    async def stop(self) -> None:
        if not self._app:
            return
        if self._app.updater and self._app.updater.running:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        self._app = None

    # ----- push API -----

    async def notify(self, text: str) -> None:
        """Send a warning/status message to the configured notify chat."""
        if not self._app:
            log.warning("notify dropped (telegram not started): %s", text[:60])
            return
        chat = self.settings.notify_chat_id
        if not chat:
            log.debug("no notify chat configured; dropping: %s", text[:60])
            return
        try:
            await self._app.bot.send_message(chat_id=chat, text=text)
        except Exception as e:  # pragma: no cover - network side-effect
            log.warning("telegram notify failed: %s", e)

    # ----- handlers -----

    def _authorised(self, chat_id: int) -> bool:
        allowed = self.settings.allowed_chat_ids
        return not allowed or chat_id in allowed

    async def _cmd_start(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Jarvis online. Send me a question about your codebase "
            "or use /ask <question>. Use /whoami to see your chat id."
        )

    async def _cmd_help(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Commands: /ask <question>, /whoami. "
            "Plain messages are forwarded to Jarvis."
        )

    async def _cmd_whoami(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(f"chat_id: {update.effective_chat.id}")

    async def _cmd_ask(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        query = " ".join(ctx.args) if ctx.args else ""
        if not query:
            await update.message.reply_text("Usage: /ask <your question>")
            return
        await self._handle(update, query)

    async def _on_text(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await self._handle(update, update.message.text or "")

    async def _handle(self, update: Update, query: str) -> None:
        chat_id = update.effective_chat.id
        if not self._authorised(chat_id):
            await update.message.reply_text("Not authorised. Ask the admin to add your chat id.")
            log.warning("rejected telegram chat %s", chat_id)
            return
        await update.message.chat.send_action(ChatAction.TYPING)
        try:
            answer = await asyncio.wait_for(self.responder(query), timeout=180)
        except asyncio.TimeoutError:
            await update.message.reply_text("Timed out. Try again with a smaller question.")
            return
        except Exception as e:
            log.exception("responder failed")
            await update.message.reply_text(f"Error: {e}")
            return
        # Telegram has a 4096 char limit per message; chunk politely.
        for i in range(0, len(answer), 3500):
            await update.message.reply_text(answer[i : i + 3500])
