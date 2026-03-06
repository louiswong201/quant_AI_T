"""Lightweight async HTTP health and metrics server."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import aiohttp.web


class HealthCheckServer:
    """Async HTTP server exposing /health and /metrics via aiohttp.web."""

    def __init__(
        self,
        *,
        health_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        metrics_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self._health_provider = health_provider or (lambda: {})
        self._metrics_provider = metrics_provider or (lambda: {})
        self._start_time: float = 0.0
        self._runner: Optional[aiohttp.web.AppRunner] = None
        self._site: Optional[aiohttp.web.TCPSite] = None

    def _health_response(self) -> Dict[str, Any]:
        defaults = {
            "feed_healthy": False,
            "broker_connected": False,
            "uptime_seconds": 0.0,
            "open_positions": 0,
            "margin_ratio": 1.0,
            "kill_switch_active": False,
        }
        data = dict(self._health_provider())
        for k, v in defaults.items():
            if k not in data:
                data[k] = v
        data["uptime_seconds"] = time.monotonic() - self._start_time
        return data

    def _metrics_response(self) -> Dict[str, Any]:
        defaults = {
            "total_trades": 0,
            "daily_pnl": 0.0,
            "max_drawdown": 0.0,
            "latency_ms": {"mean": 0.0, "p50": 0.0, "p99": 0.0},
        }
        data = dict(self._metrics_provider())
        for k, v in defaults.items():
            if k not in data:
                data[k] = v
        return data

    async def _health_handler(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.json_response(self._health_response())

    async def _metrics_handler(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.json_response(self._metrics_response())

    async def start(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        self._start_time = time.monotonic()
        app = aiohttp.web.Application()
        app.router.add_get("/health", self._health_handler)
        app.router.add_get("/metrics", self._metrics_handler)
        self._runner = aiohttp.web.AppRunner(app)
        await self._runner.setup()
        self._site = aiohttp.web.TCPSite(self._runner, host, port)
        await self._site.start()

    async def stop(self) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
