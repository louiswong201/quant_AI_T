use rayon::prelude::*;
use super::indicators;

/// Sparse MA precomputation — only computes for given window sizes.
pub fn sparse_ma(close: &[f64], windows: &[usize]) -> Vec<Vec<f64>> {
    let n = close.len();
    let mut cs = vec![0.0; n + 1];
    for i in 0..n {
        cs[i + 1] = cs[i] + close[i];
    }
    windows
        .par_iter()
        .map(|&w| {
            let mut out = vec![f64::NAN; n];
            if w >= 2 && w <= n {
                let inv_w = 1.0 / w as f64;
                for i in (w - 1)..n {
                    out[i] = (cs[i + 1] - cs[i - w + 1]) * inv_w;
                }
            }
            out
        })
        .collect()
}

/// Sparse EMA precomputation.
pub fn sparse_ema(close: &[f64], windows: &[usize]) -> Vec<Vec<f64>> {
    let n = close.len();
    windows
        .par_iter()
        .map(|&s| {
            let mut out = vec![f64::NAN; n];
            if s >= 2 && n > 0 {
                let k = 2.0 / (s as f64 + 1.0);
                out[0] = close[0];
                for i in 1..n {
                    out[i] = close[i] * k + out[i - 1] * (1.0 - k);
                }
            }
            out
        })
        .collect()
}

/// Sparse RSI precomputation.
pub fn sparse_rsi(close: &[f64], windows: &[usize]) -> Vec<Vec<f64>> {
    let n = close.len();
    windows
        .par_iter()
        .map(|&p| {
            let mut out = vec![f64::NAN; n];
            if p < 2 || n <= p {
                return out;
            }
            let pf = p as f64;
            let mut gs = 0.0;
            let mut ls = 0.0;
            for i in 1..=p {
                let d = close[i] - close[i - 1];
                if d > 0.0 { gs += d; } else { ls -= d; }
            }
            let mut ag = gs / pf;
            let mut al = ls / pf;
            out[p] = if al == 0.0 { 100.0 } else { 100.0 - 100.0 / (1.0 + ag / al) };
            for i in (p + 1)..n {
                let d = close[i] - close[i - 1];
                let g = if d > 0.0 { d } else { 0.0 };
                let l = if d < 0.0 { -d } else { 0.0 };
                ag = (ag * (pf - 1.0) + g) / pf;
                al = (al * (pf - 1.0) + l) / pf;
                out[i] = if al == 0.0 { 100.0 } else { 100.0 - 100.0 / (1.0 + ag / al) };
            }
            out
        })
        .collect()
}

/// Sparse rolling max precomputation.
pub fn sparse_rolling_max(arr: &[f64], windows: &[usize]) -> Vec<Vec<f64>> {
    windows
        .par_iter()
        .map(|&w| {
            if w >= 2 && w <= arr.len() {
                indicators::rolling_max_1d(arr, w)
            } else {
                vec![f64::NAN; arr.len()]
            }
        })
        .collect()
}

/// Sparse rolling min precomputation.
pub fn sparse_rolling_min(arr: &[f64], windows: &[usize]) -> Vec<Vec<f64>> {
    windows
        .par_iter()
        .map(|&w| {
            if w >= 2 && w <= arr.len() {
                indicators::rolling_min_1d(arr, w)
            } else {
                vec![f64::NAN; arr.len()]
            }
        })
        .collect()
}

/// Sparse rolling std precomputation.
pub fn sparse_rolling_std(close: &[f64], windows: &[usize]) -> Vec<Vec<f64>> {
    let n = close.len();
    windows
        .par_iter()
        .map(|&w| {
            let mut out = vec![f64::NAN; n];
            if w < 2 { return out; }
            let wf = w as f64;
            let mut s = 0.0;
            let mut s2 = 0.0;
            for i in 0..n {
                s += close[i];
                s2 += close[i] * close[i];
                if i >= w {
                    s -= close[i - w];
                    s2 -= close[i - w] * close[i - w];
                }
                if i >= w - 1 {
                    let m = s / wf;
                    out[i] = (s2 / wf - m * m).max(0.0).sqrt();
                }
            }
            out
        })
        .collect()
}

/// Sparse rolling volatility precomputation.
pub fn sparse_rolling_vol(close: &[f64], windows: &[usize]) -> Vec<Vec<f64>> {
    let n = close.len();
    let mut rets = vec![0.0; n];
    for i in 1..n {
        rets[i] = if close[i - 1] > 0.0 { close[i] / close[i - 1] - 1.0 } else { 0.0 };
    }
    windows
        .par_iter()
        .map(|&vp| {
            let mut out = vec![f64::NAN; n];
            if vp < 2 { return out; }
            let vpf = vp as f64;
            let mut s = 0.0;
            let mut s2 = 0.0;
            for i in 0..vp {
                s += rets[i];
                s2 += rets[i] * rets[i];
            }
            if vp > 0 {
                let m = s / vpf;
                out[vp - 1] = (s2 / vpf - m * m).max(1e-20).sqrt();
            }
            for i in vp..n {
                s += rets[i] - rets[i - vp];
                s2 += rets[i] * rets[i] - rets[i - vp] * rets[i - vp];
                let m = s / vpf;
                out[i] = (s2 / vpf - m * m).max(1e-20).sqrt();
            }
            out
        })
        .collect()
}

/// Precompute MACD lines for unique (fast, slow) EMA pairs.
pub fn macd_lines(ema_data: &[Vec<f64>], ema_windows: &[usize], pairs: &[(usize, usize)]) -> Vec<Vec<f64>> {
    let n = if ema_data.is_empty() { 0 } else { ema_data[0].len() };
    pairs
        .par_iter()
        .map(|&(fi, si)| {
            let fi_idx = ema_windows.iter().position(|&w| w == fi);
            let si_idx = ema_windows.iter().position(|&w| w == si);
            match (fi_idx, si_idx) {
                (Some(fi_i), Some(si_i)) => {
                    let mut out = vec![0.0; n];
                    for i in 0..n {
                        out[i] = ema_data[fi_i][i] - ema_data[si_i][i];
                    }
                    out
                }
                _ => vec![f64::NAN; n],
            }
        })
        .collect()
}

/// Precompute all KAMA arrays for unique (erp, fsc, ssc) combos.
pub fn all_kama(close: &[f64], unique_params: &[(usize, usize, usize)]) -> Vec<Vec<f64>> {
    unique_params
        .par_iter()
        .map(|&(er_p, fast_sc, slow_sc)| {
            indicators::compute_kama(close, er_p, fast_sc, slow_sc)
        })
        .collect()
}
