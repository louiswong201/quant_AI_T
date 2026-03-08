"""Binance USDT-M futures broker."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

import aiohttp
import websockets

from ..core.asset_types import AssetClass
from ..core.symbol_spec import SymbolSpec
from .base import Broker
from .credentials import CredentialManager
from .rate_limiter import RateLimiter
from .testnet import TestnetConfig

logger = logging.getLogger(__name__)


class BinanceFuturesBroker(Broker):
    BASE_URL = "https://fapi.binance.com"
    TESTNET_URL = "https://testnet.binancefuture.com"
    WS_URL = "wss://fstream.binance.com"
    WS_TESTNET_URL = "wss://stream.binancefuture.com"

    def __init__(
        self,
        credentials: CredentialManager,
        testnet: bool = False,
        margin_type: str = "ISOLATED",
    ) -> None:
        self._cred = credentials
        cfg = TestnetConfig.BINANCE_FUTURES
        self._base = cfg["rest"] if testnet else self.BASE_URL
        self._ws_base = cfg["ws"] if testnet else self.WS_URL
        self._margin_type = margin_type
        self._rate_limiter = RateLimiter()
        self._session: Optional[aiohttp.ClientSession] = None
        self._positions: Dict[str, float] = {}
        self._balances: Dict[str, float] = {}
        self._symbol_specs: Dict[str, SymbolSpec] = {}
        self._api_key = ""
        self._secret = ""
        self._running = True
        self._total_equity = 0.0
        self._total_maint_margin = 0.0
        self._available_balance = 0.0
        self._order_symbols: Dict[str, str] = {}
        self._user_data_stream_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        api_key, secret = self._cred.load("binance")
        if not api_key or not secret:
            raise ValueError("Binance credentials not found")
        self._api_key = api_key
        self._secret = secret
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=20, keepalive_timeout=30),
            timeout=aiohttp.ClientTimeout(total=10, connect=5),
        )
        await self._load_exchange_info()
        await self.sync_positions()
        await self.sync_balance()
        self._user_data_stream_task = asyncio.create_task(self._user_data_stream())

    async def close(self) -> None:
        self._running = False
        if getattr(self, "_user_data_stream_task", None) is not None:
            self._user_data_stream_task.cancel()
            try:
                await self._user_data_stream_task
            except asyncio.CancelledError:
                pass
            self._user_data_stream_task = None
        if self._session:
            await self._session.close()
            self._session = None

    async def _load_exchange_info(self) -> None:
        data = await self._request("GET", "/fapi/v1/exchangeInfo")
        for s in data.get("symbols", []):
            if not s.get("status") == "TRADING" or s.get("contractType") != "PERPETUAL":
                continue
            filters = {f["filterType"]: f for f in s.get("filters", [])}
            lot = filters.get("LOT_SIZE", {})
            price_f = filters.get("PRICE_FILTER", {})
            notional = filters.get("MIN_NOTIONAL", filters.get("NOTIONAL", {}))
            tick = float(price_f.get("tickSize", "0.01"))
            step = float(lot.get("stepSize", "0.001"))
            min_qty = float(lot.get("minQty", "0.001"))
            max_qty = float(lot.get("maxQty", "1000000"))
            min_not = float(notional.get("notional", notional.get("minNotional", "5")))
            self._symbol_specs[s["symbol"]] = SymbolSpec(
                symbol=s["symbol"],
                asset_class=AssetClass.CRYPTO_PERP,
                base_asset=s["baseAsset"],
                quote_asset=s["quoteAsset"],
                tick_size=tick,
                step_size=step,
                min_notional=min_not,
                min_qty=min_qty,
                max_qty=max_qty,
                max_leverage=125.0,
            )

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        return await self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    async def set_margin_type(self, symbol: str, margin_type: Optional[str] = None) -> None:
        mt = margin_type or self._margin_type
        try:
            await self._request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": mt})
        except Exception:
            pass

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = True,
    ) -> Any:
        if self._session is None:
            raise RuntimeError("BinanceFuturesBroker not initialized. Call await initialize() first.")
        params = dict(params) if params else {}
        if signed:
            await self._rate_limiter.acquire(weight=1)
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._cred.sign_request(params, self._secret)
        url = f"{self._base}{path}"
        kwargs: Dict[str, Any] = {"headers": {"X-MBX-APIKEY": self._api_key}}
        if method in ("GET", "PUT"):
            kwargs["params"] = params
        else:
            kwargs["data"] = params
        async with self._session.request(method, url, **kwargs) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"Binance API error: {data}")
            return data

    async def submit_order_async(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(signal.get("symbol", ""))
        spec = self._symbol_specs.get(symbol)
        if not spec:
            return {"status": "rejected", "message": f"unknown symbol {symbol}"}
        price = float(signal.get("price") or signal.get("fill_price") or 0.0)
        raw_qty = float(signal.get("shares") or 0)
        qty = spec.round_quantity(abs(raw_qty))
        if qty <= 0:
            return {"status": "rejected", "message": "invalid quantity"}
        if price <= 0 and signal.get("order_type") == "limit":
            return {"status": "rejected", "message": "limit order requires price"}
        if price > 0:
            err = spec.validate_order(qty, price)
            if err:
                return {"status": "rejected", "message": err}
        side = "BUY" if str(signal.get("action", "")).lower() == "buy" else "SELL"
        order_type = str(signal.get("order_type", "market")).upper()
        reduce_only = signal.get("reduce_only") or signal.get("action") in ("close_long", "close_short")
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT" if order_type == "limit" else "MARKET",
            "quantity": qty,
            "newClientOrderId": signal.get("client_order_id") or f"qf_{uuid.uuid4().hex[:16]}",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if order_type == "limit" and price > 0:
            params["price"] = spec.round_price(price)
        if order_type == "limit":
            params["timeInForce"] = signal.get("time_in_force", "GTC")
        try:
            result = await self._request("POST", "/fapi/v1/order", params)
            order_id = str(result.get("orderId", ""))
            self._order_symbols[order_id] = symbol
            status = str(result.get("status", "")).lower()
            return {
                "status": "filled" if status == "filled" else "submitted",
                "order_id": order_id,
                "client_order_id": result.get("clientOrderId"),
                "fill_price": float(result.get("avgPrice") or 0),
                "filled_shares": float(result.get("executedQty") or 0),
                "commission": float(result.get("commission") or 0),
            }
        except Exception as e:
            logger.exception("submit_order_async failed")
            return {"status": "error", "message": str(e)}

    def submit_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.submit_order_async(signal))

    async def cancel_order_async(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        sym = symbol or self._order_symbols.get(order_id)
        if not sym:
            raise ValueError(f"Unknown symbol for cancel: {symbol or order_id}")
        try:
            await self._request("DELETE", "/fapi/v1/order", {"symbol": sym, "orderId": order_id})
            return {"status": "cancelled", "order_id": order_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_order_status_async(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        sym = symbol or self._order_symbols.get(order_id)
        if not sym and self._symbol_specs:
            sym = next(iter(self._symbol_specs))
        if not sym:
            return {"status": "error", "message": "no symbol for order lookup"}
        try:
            result = await self._request("GET", "/fapi/v1/order", {"symbol": sym, "orderId": order_id})
            status = str(result.get("status", "")).lower()
            return {
                "status": status,
                "order_id": str(result.get("orderId")),
                "fill_price": float(result.get("avgPrice") or 0),
                "filled_shares": float(result.get("executedQty") or 0),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_open_orders_async(self, symbol: str = "") -> list:
        params: Dict[str, str] = {}
        if symbol:
            params["symbol"] = symbol
        data = await self._request("GET", "/fapi/v1/openOrders", params)
        orders = list(data) if isinstance(data, list) else []
        for o in orders:
            oid = str(o.get("orderId", ""))
            sym = o.get("symbol", "")
            if oid and sym:
                self._order_symbols[oid] = sym
        return orders

    async def sync_positions(self) -> Dict[str, float]:
        data = await self._request("GET", "/fapi/v2/positionRisk")
        self._positions = {}
        self._total_maint_margin = 0.0
        for p in data:
            qty = float(p.get("positionAmt", 0))
            if abs(qty) > 1e-10:
                sym = p["symbol"]
                self._positions[sym] = qty
                maint = float(p.get("maintMargin", 0))
                self._total_maint_margin += maint
        return dict(self._positions)

    async def sync_balance(self) -> Dict[str, Any]:
        data = await self._request("GET", "/fapi/v2/balance")
        self._balances = {}
        self._total_equity = 0.0
        self._available_balance = 0.0
        for b in data:
            bal = float(b.get("balance", 0))
            avail = float(b.get("availableBalance", bal))
            if bal > 0:
                asset = b["asset"]
                self._balances[asset] = bal
                if asset == "USDT":
                    self._total_equity += bal
                    self._available_balance = avail
            cross_wallet = float(b.get("crossWalletBalance", 0))
            unrealized = float(b.get("unrealizedProfit", 0))
            self._total_equity += cross_wallet + unrealized
        return {"cash": self._balances.get("USDT", 0.0), "balances": dict(self._balances)}

    def _handle_account_update(self, data: Dict[str, Any]) -> None:
        for b in data.get("B", []):
            asset = b.get("a", "")
            balance = float(b.get("wb", 0))
            if asset:
                self._balances[asset] = balance
        for p in data.get("P", []):
            sym = p.get("s", "")
            amt = float(p.get("pa", 0))
            if abs(amt) > 1e-10:
                self._positions[sym] = amt
            else:
                self._positions.pop(sym, None)

    async def _handle_order_update(self, order_data: Dict[str, Any]) -> None:
        pass

    async def _get_listen_key(self) -> str:
        data = await self._request("POST", "/fapi/v1/listenKey", {}, signed=True)
        return data["listenKey"]

    async def _keepalive_listen_key(self, listen_key: str) -> None:
        while self._running:
            await asyncio.sleep(25 * 60)
            if not self._running:
                break
            try:
                await self._request("PUT", "/fapi/v1/listenKey", {"listenKey": listen_key})
            except Exception as e:
                logger.warning("Listen key keepalive failed: %s", e)

    async def _user_data_stream(self) -> None:
        backoff = 1.0
        while self._running:
            try:
                listen_key = await self._get_listen_key()
                keepalive = asyncio.create_task(self._keepalive_listen_key(listen_key))
                url = f"{self._ws_base}/ws/{listen_key}"
                async with websockets.connect(url, ping_interval=20) as ws:
                    backoff = 1.0
                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        event = data.get("e")
                        if event == "ORDER_TRADE_UPDATE":
                            await self._handle_order_update(data.get("o", {}))
                        elif event == "ACCOUNT_UPDATE":
                            self._handle_account_update(data.get("a", {}))
            except Exception as e:
                logger.warning("User data stream error: %s, reconnect in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def get_positions(self) -> Dict[str, float]:
        return dict(self._positions)

    def get_cash(self) -> float:
        return self._balances.get("USDT", 0.0)

    def get_available_margin(self) -> float:
        return self._available_balance if self._available_balance > 0 else self.get_cash()

    def get_margin_ratio(self) -> float:
        if self._total_maint_margin <= 0:
            return 1.0
        return self._total_equity / self._total_maint_margin if self._total_equity > 0 else 0.0
