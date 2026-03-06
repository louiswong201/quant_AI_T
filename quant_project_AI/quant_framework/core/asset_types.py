from __future__ import annotations

from enum import Enum


class AssetClass(Enum):
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_PERP = "crypto_perp"
    CRYPTO_INVERSE = "crypto_inverse"
    US_EQUITY = "us_equity"
    US_EQUITY_MARGIN = "us_equity_margin"
