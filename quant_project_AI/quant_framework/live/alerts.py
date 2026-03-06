"""Multi-channel alerts: Telegram, Discord, Email, Console."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AlertManager:
    """Multi-channel alerts: Telegram / Discord / Email / Console."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        config = config or {}
        self._telegram_token = config.get("telegram_bot_token")
        self._telegram_chat = config.get("telegram_chat_id")
        self._discord_webhook = config.get("discord_webhook_url")
        self._log_to_console = config.get("console", True)

    async def send(self, level: str, title: str, body: str) -> List[Any]:
        """Send alert to all configured channels."""
        tasks: List[asyncio.Task[Any]] = []
        if self._log_to_console:
            tasks.append(asyncio.create_task(self._send_console(level, title, body)))
        if self._telegram_token and self._telegram_chat:
            tasks.append(asyncio.create_task(self._send_telegram(level, title, body)))
        if self._discord_webhook:
            tasks.append(asyncio.create_task(self._send_discord(level, title, body)))
        if not tasks:
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("Alert channel %d failed: %s", i, r)
        return list(results)

    async def _send_console(self, level: str, title: str, body: str) -> None:
        msg = f"[{level}] {title}: {body}"
        if level.upper() == "CRITICAL":
            logger.critical("%s", msg)
        elif level.upper() == "ERROR":
            logger.error("%s", msg)
        elif level.upper() == "WARNING":
            logger.warning("%s", msg)
        else:
            logger.info("%s", msg)

    async def _send_telegram(self, level: str, title: str, body: str) -> None:
        text = f"*{level}*\n*{title}*\n\n{body}"
        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        payload = {"chat_id": self._telegram_chat, "text": text, "parse_mode": "Markdown"}
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=payload) as resp:
                resp.raise_for_status()

    async def _send_discord(self, level: str, title: str, body: str) -> None:
        color = 0xE74C3C if level.upper() in ("CRITICAL", "ERROR") else 0xF39C12
        payload = {
            "embeds": [{
                "title": f"[{level}] {title}",
                "description": body,
                "color": color,
            }]
        }
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.post(self._discord_webhook, json=payload) as resp:
                resp.raise_for_status()
