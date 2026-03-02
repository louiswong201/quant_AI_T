"""
Run manifest：记录一次回测的关键信息，便于复现与审计。
"""

from __future__ import annotations

import hashlib
import inspect
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import BacktestConfig


def _safe_public_attrs(obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in vars(obj).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
    return out


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _strategy_source_digest(strategy: Any) -> Dict[str, Any]:
    src_file = inspect.getsourcefile(strategy.__class__)
    if not src_file:
        return {"strategy_source_file": None, "strategy_source_sha256": None}
    p = Path(src_file)
    if not p.exists():
        return {"strategy_source_file": str(p), "strategy_source_sha256": None}
    return {"strategy_source_file": str(p), "strategy_source_sha256": _file_sha256(p)}


def build_run_manifest(
    strategy: Any,
    symbols: List[str],
    start_date: str,
    end_date: str,
    data_by_symbol: Dict[str, pd.DataFrame],
    config: BacktestConfig,
) -> Dict[str, Any]:
    """
    构建一次回测的 manifest。用于回测结果可追溯与复现实验。
    """
    bars_by_symbol = {s: int(len(df)) for s, df in data_by_symbol.items()}
    manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "strategy_class": strategy.__class__.__name__,
        "strategy_params": _safe_public_attrs(strategy),
        "symbols": list(symbols),
        "start_date": start_date,
        "end_date": end_date,
        "bars_by_symbol": bars_by_symbol,
        "cost_config": {
            "commission_pct_buy": config.commission_pct_buy,
            "commission_pct_sell": config.commission_pct_sell,
            "commission_fixed_buy": config.commission_fixed_buy,
            "commission_fixed_sell": config.commission_fixed_sell,
            "slippage_bps_buy": config.slippage_bps_buy,
            "slippage_bps_sell": config.slippage_bps_sell,
            "slippage_fixed_buy": config.slippage_fixed_buy,
            "slippage_fixed_sell": config.slippage_fixed_sell,
            "max_participation_rate": config.max_participation_rate,
            "impact_bps_buy_coeff": config.impact_bps_buy_coeff,
            "impact_bps_sell_coeff": config.impact_bps_sell_coeff,
            "impact_exponent": config.impact_exponent,
            "adaptive_impact": config.adaptive_impact,
            "impact_vol_window": config.impact_vol_window,
            "impact_vol_ref": config.impact_vol_ref,
            "auto_export_execution_report": config.auto_export_execution_report,
            "execution_report_path": config.execution_report_path,
        },
    }
    manifest.update(_strategy_source_digest(strategy))
    return manifest

