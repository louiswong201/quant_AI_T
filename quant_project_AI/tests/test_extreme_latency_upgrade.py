from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from quant_framework.features import OfflineMaterializer, OnlineFeatureEngine
from quant_framework.live.events import InMemoryEventBus
from quant_framework.live.trade_journal import TradeJournal
from quant_framework.research.database import ResearchDB


def _sample_frame(n: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    close = np.linspace(100.0, 120.0, n)
    return pd.DataFrame(
        {
            "date": idx,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1000.0, 3000.0, n),
        }
    )


def test_online_feature_engine_matches_expected_core_features():
    df = _sample_frame(40)
    arrays = {
        "open": df["open"].to_numpy(dtype=np.float64),
        "high": df["high"].to_numpy(dtype=np.float64),
        "low": df["low"].to_numpy(dtype=np.float64),
        "close": df["close"].to_numpy(dtype=np.float64),
        "volume": df["volume"].to_numpy(dtype=np.float64),
    }
    engine = OnlineFeatureEngine(feature_set_version="core_v1")
    snap = engine.update("TEST", "1d", arrays)

    assert snap.feature_set_version == "core_v1"
    assert snap.values["close"] == float(df["close"].iloc[-1])
    assert abs(snap.values["ma_10"] - float(df["close"].tail(10).mean())) < 1e-10
    assert abs(snap.values["ma_20"] - float(df["close"].tail(20).mean())) < 1e-10
    assert 0.0 <= snap.values["rsi_14"] <= 100.0
    assert "atr_14" in snap.values


def test_offline_materializer_loads_canonical_arrays(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = _sample_frame(50)
    df.to_csv(data_dir / "TEST.csv", index=False)

    mat = OfflineMaterializer(data_dir)
    arrays = mat.load_ohlcv_arrays("TEST", interval="1d", min_bars=10)

    assert set(arrays) >= {"c", "o", "h", "l", "v"}
    assert len(arrays["c"]) == 50
    assert float(arrays["c"][-1]) == float(df["close"].iloc[-1])


def test_in_memory_event_bus_dispatches():
    received = []

    async def run_test() -> None:
        bus = InMemoryEventBus()

        def handler(event):
            received.append((event.event_type, event.payload["x"]))

        bus.subscribe("test.event", handler)
        await bus.start()
        bus.publish_nowait("test.event", {"x": 42})
        await asyncio.sleep(0.05)
        await bus.stop()

    asyncio.run(run_test())
    assert received == [("test.event", 42)]


def test_trade_journal_background_flush_and_metrics(tmp_path: Path):
    db_path = tmp_path / "journal.db"
    journal = TradeJournal(str(db_path))
    try:
        journal.record_signal("TEST", "buy", "MA", {"price": 100.0})
        journal.record_trade("TEST", "buy", 1.0, 100.0, commission=0.1, pnl=0.0, strategy="MA")
        journal.record_equity(1000.0, 900.0, {"TEST": 1.0})
        journal._flush()
        metrics = journal.get_write_metrics()
        assert metrics["pending_rows"] == 0
        assert not journal.get_trades(limit=10).empty
    finally:
        journal.close()


def test_research_db_persists_feature_set_version(tmp_path: Path):
    db = ResearchDB(str(tmp_path / "research.db"))
    try:
        db.record_param_update(
            "TEST",
            "MA",
            [5, 20],
            sharpe=1.2,
            wf_score=1.0,
            feature_set_version="core_v1",
        )
        row = db.get_latest_params("TEST", "MA")
        assert row is not None
        assert row["feature_set_version"] == "core_v1"

        db.save_config_version(
            {"recommendations": []},
            summary="test",
            feature_set_version="core_v1",
        )
        cfg = db.get_latest_config()
        assert cfg is not None
        assert cfg["feature_set_version"] == "core_v1"
    finally:
        db.close()
