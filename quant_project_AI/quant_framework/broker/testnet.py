"""Testnet configuration for Binance and IBKR."""

from __future__ import annotations

from typing import Any, Dict


class TestnetConfig:
    """Testnet URLs for Binance and IBKR."""

    BINANCE_FUTURES: Dict[str, str] = {
        "rest": "https://testnet.binancefuture.com",
        "ws": "wss://stream.binancefuture.com",
        "ws_user": "wss://stream.binancefuture.com",
    }
    IBKR_PAPER: Dict[str, Any] = {"port": 7497}
