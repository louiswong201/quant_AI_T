"""API key management. Priority: env vars > .env file > config file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import yaml
except ImportError:
    yaml = None


class CredentialManager:
    """API key management. Priority: env vars > .env file > config file."""

    _ENV_KEYS: dict[str, Tuple[str, str]] = {
        "binance": ("BINANCE_API_KEY", "BINANCE_SECRET"),
        "ibkr": ("IBKR_HOST", "IBKR_PORT"),
    }

    def __init__(self, config_path: str | Path = "config/credentials.yaml") -> None:
        self._config_path = Path(config_path)
        if load_dotenv:
            load_dotenv()

    def load(self, exchange: str) -> Tuple[str, str]:
        keys = self._ENV_KEYS.get(exchange)
        if not keys:
            raise ValueError(f"Unknown exchange: {exchange}")
        key1, key2 = keys
        v1 = os.environ.get(key1)
        v2 = os.environ.get(key2)
        if v1 and v2:
            return v1.strip(), v2.strip()
        if self._config_path.exists() and yaml:
            with open(self._config_path) as f:
                cfg = yaml.safe_load(f) or {}
            exc_cfg = cfg.get(exchange, {})
            cfg_keys = {"binance": ("api_key", "secret"), "ibkr": ("host", "port")}
            ck1, ck2 = cfg_keys.get(exchange, ("api_key", "secret"))
            v1 = v1 or exc_cfg.get(ck1, "")
            v2 = v2 if v2 is not None else exc_cfg.get(ck2, "")
        return (v1 or "").strip(), str(v2 or "").strip()

    def sign_request(self, params: dict, secret: str) -> str:
        """Binance HMAC-SHA256 signature."""
        import hmac
        import hashlib
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
