"""Quick comparison: Rust eval_kernel vs Numba eval_kernel for all 18 strategies."""
import numpy as np
import sys

sys.path.insert(0, ".")

rng = np.random.RandomState(12345)
ret = rng.randn(500) * 0.01
c = 100.0 * np.cumprod(1.0 + ret)
spread = np.abs(rng.randn(500) * 0.003) * c
h = c + spread
l = c - spread
o = c + rng.randn(500) * 0.001 * c
for a in (c, o, h, l):
    a[:] = np.ascontiguousarray(a, dtype=np.float64)

from quant_framework.backtest.kernels import (
    KERNEL_NAMES, DEFAULT_PARAM_GRIDS, eval_kernel,
    config_to_kernel_costs, _USE_RUST,
)
from quant_framework.backtest.config import BacktestConfig

print(f"Rust available: {_USE_RUST}")

cfg = BacktestConfig.crypto()
costs = config_to_kernel_costs(cfg)
sb, ss, cm = costs["sb"], costs["ss"], costs["cm"]
lev, dc = costs["lev"], costs["dc"]
sl, pfrac, sl_slip = costs["sl"], costs["pfrac"], costs["sl_slip"]

import quant_core
rust_results = {}
for name in KERNEL_NAMES:
    params = DEFAULT_PARAM_GRIDS[name][0]
    r, d, t = quant_core.eval_kernel(
        name, [float(v) for v in params], c, o, h, l,
        sb, ss, cm, lev, dc, 0.0, sl, pfrac, sl_slip,
    )
    rust_results[name] = (r, d, t)

import sys as _sys
km = _sys.modules["quant_framework.backtest.kernels"]
old_flag = km._USE_RUST
km._USE_RUST = False
numba_results = {}
for name in KERNEL_NAMES:
    params = DEFAULT_PARAM_GRIDS[name][0]
    r, d, t = eval_kernel(name, params, c, o, h, l, sb, ss, cm, lev, dc, 0.0, sl, pfrac, sl_slip)
    numba_results[name] = (r, d, t)
km._USE_RUST = old_flag

all_pass = True
for name in KERNEL_NAMES:
    rr, rd, rt = rust_results[name]
    nr, nd, nt = numba_results[name]
    ret_ok = abs(rr - nr) < 0.5
    dd_ok = abs(rd - nd) < 0.5
    tr_ok = rt == nt
    status = "PASS" if (ret_ok and dd_ok and tr_ok) else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"{status} {name:15s}  rust=({rr:+.4f}, {rd:.4f}, {rt})  numba=({nr:+.4f}, {nd:.4f}, {nt})  diff_ret={rr-nr:+.6f}")

print()
if all_pass:
    print("ALL 18 STRATEGIES PASS")
else:
    print("SOME STRATEGIES FAILED")

sys.exit(0 if all_pass else 1)
