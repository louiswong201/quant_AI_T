use rayon::prelude::*;
use super::helpers::*;
use super::strategies;

/// Result of a parameter grid scan for a single strategy.
#[derive(Clone, Debug)]
pub struct ScanResult {
    pub best_idx: usize,
    pub best_score: f64,
    pub best_ret: f64,
    pub best_dd: f64,
    pub best_nt: i64,
    pub count: usize,
}

/// Generic scan: run a closure on each parameter set in parallel, return best.
fn scan_grid<F>(grid_len: usize, eval_fn: F) -> ScanResult
where
    F: Fn(usize) -> (f64, f64, i64) + Sync,
{
    if grid_len == 0 {
        return ScanResult {
            best_idx: 0, best_score: -1e18, best_ret: 0.0,
            best_dd: 0.0, best_nt: 0, count: 0,
        };
    }
    let results: Vec<(f64, f64, f64, i64)> = (0..grid_len)
        .into_par_iter()
        .map(|k| {
            let (r, d, nt) = eval_fn(k);
            (score(r, d, nt), r, d, nt)
        })
        .collect();

    let mut bi = 0;
    for k in 1..results.len() {
        if results[k].0 > results[bi].0 {
            bi = k;
        }
    }
    ScanResult {
        best_idx: bi,
        best_score: results[bi].0,
        best_ret: results[bi].1,
        best_dd: results[bi].2,
        best_nt: results[bi].3,
        count: grid_len,
    }
}

/// MA strategy scan.
pub fn scan_ma(
    grid: &[(usize, usize)],
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    mas: &[Vec<f64>], ma_windows: &[usize], cp: &CostParams,
) -> ScanResult {
    scan_grid(grid.len(), |k| {
        let (si, li) = grid[k];
        let si_idx = ma_windows.iter().position(|&w| w == si).unwrap_or(0);
        let li_idx = ma_windows.iter().position(|&w| w == li).unwrap_or(0);
        strategies::bt_ma_ls(c, o, h, l, &mas[si_idx], &mas[li_idx], cp)
    })
}

/// RSI strategy scan.
pub fn scan_rsi(
    grid: &[(usize, f64, f64)],
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    rsis: &[Vec<f64>], rsi_windows: &[usize], cp: &CostParams,
) -> ScanResult {
    scan_grid(grid.len(), |k| {
        let (rp, os_v, ob_v) = grid[k];
        let rp_idx = rsi_windows.iter().position(|&w| w == rp).unwrap_or(0);
        strategies::bt_rsi_ls(c, o, h, l, &rsis[rp_idx], os_v, ob_v, cp)
    })
}

/// MACD strategy scan.
pub fn scan_macd(
    grid: &[(usize, usize, usize)],
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    macd_lines: &[Vec<f64>], pair_idx: &[usize], cp: &CostParams,
) -> ScanResult {
    scan_grid(grid.len(), |k| {
        let sig_span = grid[k].2;
        strategies::bt_macd_precomp(c, o, h, l, &macd_lines[pair_idx[k]], sig_span, cp)
    })
}

/// Drift strategy scan.
pub fn scan_drift(
    grid: &[(usize, f64, usize)],
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    up_prefix: &[i64], cp: &CostParams,
) -> ScanResult {
    scan_grid(grid.len(), |k| {
        let (lb, dt, hp) = grid[k];
        strategies::bt_drift_precomp(c, o, h, l, up_prefix, lb, dt, hp, cp)
    })
}

/// RAMOM strategy scan.
pub fn scan_ramom(
    grid: &[(usize, usize, f64, f64)],
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    vols: &[Vec<f64>], vol_windows: &[usize], cp: &CostParams,
) -> ScanResult {
    scan_grid(grid.len(), |k| {
        let (mp, vp, ez, xz) = grid[k];
        let vp_idx = vol_windows.iter().position(|&w| w == vp).unwrap_or(0);
        strategies::bt_ramom_precomp(c, o, h, l, mp, &vols[vp_idx], ez, xz, cp)
    })
}

/// Bollinger strategy scan.
pub fn scan_bollinger(
    grid: &[(usize, f64)],
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    mas: &[Vec<f64>], stds: &[Vec<f64>],
    ma_windows: &[usize], std_windows: &[usize], cp: &CostParams,
) -> ScanResult {
    scan_grid(grid.len(), |k| {
        let (p, ns) = grid[k];
        let p_ma = ma_windows.iter().position(|&w| w == p).unwrap_or(0);
        let p_std = std_windows.iter().position(|&w| w == p).unwrap_or(0);
        strategies::bt_bollinger_precomp(c, o, h, l, &mas[p_ma], &stds[p_std], ns, p, cp)
    })
}

/// DualMom strategy scan.
pub fn scan_dualmom(
    grid: &[(usize, usize)],
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    cp: &CostParams,
) -> ScanResult {
    scan_grid(grid.len(), |k| {
        let (fl, sl) = grid[k];
        strategies::bt_dualmom_ls(c, o, h, l, fl, sl, cp)
    })
}

/// MESA strategy scan.
pub fn scan_mesa(
    grid: &[(f64, f64)],
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    cp: &CostParams,
) -> ScanResult {
    scan_grid(grid.len(), |k| {
        let (fl, sl) = grid[k];
        strategies::bt_mesa_ls(c, o, h, l, fl, sl, cp)
    })
}

/// ZScore strategy scan.
pub fn scan_zscore(
    grid: &[(usize, f64, f64, f64)],
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    mas: &[Vec<f64>], stds: &[Vec<f64>],
    ma_windows: &[usize], std_windows: &[usize], cp: &CostParams,
) -> ScanResult {
    scan_grid(grid.len(), |k| {
        let (lb, ez, xz, sz) = grid[k];
        let lb_ma = ma_windows.iter().position(|&w| w == lb).unwrap_or(0);
        let lb_std = std_windows.iter().position(|&w| w == lb).unwrap_or(0);
        strategies::bt_zscore_precomp(c, o, h, l, &mas[lb_ma], &stds[lb_std], lb, ez, xz, sz, cp)
    })
}

/// Consensus strategy scan.
pub fn scan_consensus(
    grid: &[(usize, usize, usize, usize, f64, f64, usize)],
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    mas: &[Vec<f64>], rsis: &[Vec<f64>],
    ma_windows: &[usize], rsi_windows: &[usize], cp: &CostParams,
) -> ScanResult {
    scan_grid(grid.len(), |k| {
        let (ms, ml, rp, mom_lb, os_v, ob_v, vt) = grid[k];
        let ms_idx = ma_windows.iter().position(|&w| w == ms).unwrap_or(0);
        let ml_idx = ma_windows.iter().position(|&w| w == ml).unwrap_or(0);
        let rp_idx = rsi_windows.iter().position(|&w| w == rp).unwrap_or(0);
        strategies::bt_consensus_ls(
            c, o, h, l, &mas[ms_idx], &mas[ml_idx], &rsis[rp_idx],
            mom_lb, os_v, ob_v, vt as i32, cp,
        )
    })
}
