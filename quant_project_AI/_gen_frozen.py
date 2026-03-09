import numpy as np
from quant_framework.backtest.kernels import (
    eval_kernel, DEFAULT_PARAM_GRIDS, KERNEL_NAMES, config_to_kernel_costs,
)
from quant_framework.backtest.config import BacktestConfig

rng = np.random.RandomState(12345)
ret = rng.randn(500) * 0.01
close = 100.0 * np.cumprod(1.0 + ret)
spread = np.abs(rng.randn(500) * 0.003) * close
high = close + spread
low = close - spread
open_ = close + rng.randn(500) * 0.001 * close
for arr in (close, open_, high, low):
    arr[:] = np.ascontiguousarray(arr, dtype=np.float64)

cfg = BacktestConfig.crypto()
co = config_to_kernel_costs(cfg)

print("FROZEN_REFS = {")
for name in KERNEL_NAMES:
    params = DEFAULT_PARAM_GRIDS[name][0]
    r, d, t = eval_kernel(
        name, params, close, open_, high, low,
        co["sb"], co["ss"], co["cm"], co["lev"], co["dc"],
        co["sl"], co["pfrac"], co["sl_slip"],
    )
    print(f'    "{name}": ({r:.6f}, {d:.6f}, {t}),')
print("}")
