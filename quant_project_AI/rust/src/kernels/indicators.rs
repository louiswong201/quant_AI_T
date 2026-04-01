/// Exponential moving average.
pub fn ema(arr: &[f64], span: usize) -> Vec<f64> {
    let n = arr.len();
    let mut out = vec![0.0; n];
    if n == 0 {
        return out;
    }
    let k = 2.0 / (span as f64 + 1.0);
    out[0] = arr[0];
    for i in 1..n {
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k);
    }
    out
}

/// Simple rolling mean.
pub fn rolling_mean(arr: &[f64], w: usize) -> Vec<f64> {
    let n = arr.len();
    let mut out = vec![f64::NAN; n];
    let mut s = 0.0;
    for i in 0..n {
        s += arr[i];
        if i >= w {
            s -= arr[i - w];
        }
        if i >= w - 1 {
            out[i] = s / w as f64;
        }
    }
    out
}

/// Rolling standard deviation.
pub fn rolling_std(arr: &[f64], w: usize) -> Vec<f64> {
    let n = arr.len();
    let mut out = vec![f64::NAN; n];
    let mut s = 0.0;
    let mut s2 = 0.0;
    let wf = w as f64;
    for i in 0..n {
        s += arr[i];
        s2 += arr[i] * arr[i];
        if i >= w {
            s -= arr[i - w];
            s2 -= arr[i - w] * arr[i - w];
        }
        if i >= w - 1 {
            let m = s / wf;
            out[i] = (s2 / wf - m * m).max(0.0).sqrt();
        }
    }
    out
}

/// ATR (Average True Range) using Wilder's smoothing.
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut tr_arr = vec![0.0; n];
    tr_arr[0] = high[0] - low[0];
    for i in 1..n {
        tr_arr[i] = (high[i] - low[i])
            .max((high[i] - close[i - 1]).abs())
            .max((low[i] - close[i - 1]).abs());
    }
    let mut out = vec![f64::NAN; n];
    let mut s = 0.0;
    for i in 0..period {
        s += tr_arr[i];
    }
    out[period - 1] = s / period as f64;
    let pf = period as f64;
    for i in period..n {
        out[i] = (out[i - 1] * (pf - 1.0) + tr_arr[i]) / pf;
    }
    out
}

/// RSI using Wilder's smoothing.
pub fn rsi_wilder(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n < period + 1 {
        return out;
    }
    let mut gs = 0.0;
    let mut ls = 0.0;
    let pf = period as f64;
    for i in 1..=period {
        let d = close[i] - close[i - 1];
        if d > 0.0 {
            gs += d;
        } else {
            ls -= d;
        }
    }
    let mut ag = gs / pf;
    let mut al = ls / pf;
    out[period] = if al == 0.0 {
        100.0
    } else {
        100.0 - 100.0 / (1.0 + ag / al)
    };
    for i in (period + 1)..n {
        let d = close[i] - close[i - 1];
        let g = if d > 0.0 { d } else { 0.0 };
        let l = if d < 0.0 { -d } else { 0.0 };
        ag = (ag * (pf - 1.0) + g) / pf;
        al = (al * (pf - 1.0) + l) / pf;
        out[i] = if al == 0.0 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + ag / al)
        };
    }
    out
}

/// Prefix-sum of up-bars for O(1) drift ratio lookup.
pub fn up_prefix(close: &[f64]) -> Vec<i64> {
    let n = close.len();
    let mut psum = vec![0i64; n + 1];
    for i in 1..n {
        psum[i + 1] = psum[i] + if close[i] > close[i - 1] { 1 } else { 0 };
    }
    psum
}

/// O(n) rolling max using block decomposition.
pub fn rolling_max_1d(arr: &[f64], w: usize) -> Vec<f64> {
    let n = arr.len();
    let mut out = vec![f64::NAN; n];
    if w < 1 || w > n {
        return out;
    }
    let mut prefix = vec![0.0; n];
    let mut suffix = vec![0.0; n];
    let mut bi = 0;
    while bi < n {
        let be = (bi + w).min(n);
        suffix[bi] = arr[bi];
        for j in (bi + 1)..be {
            suffix[j] = if arr[j] > suffix[j - 1] {
                arr[j]
            } else {
                suffix[j - 1]
            };
        }
        prefix[be - 1] = arr[be - 1];
        if be >= 2 {
            for j in (bi..=(be - 2)).rev() {
                prefix[j] = if arr[j] > prefix[j + 1] {
                    arr[j]
                } else {
                    prefix[j + 1]
                };
            }
        }
        bi += w;
    }
    for i in (w - 1)..n {
        let p = prefix[i - w + 1];
        let s = suffix[i];
        out[i] = if p > s { p } else { s };
    }
    out
}

/// O(n) rolling min using block decomposition.
pub fn rolling_min_1d(arr: &[f64], w: usize) -> Vec<f64> {
    let n = arr.len();
    let mut out = vec![f64::NAN; n];
    if w < 1 || w > n {
        return out;
    }
    let mut prefix = vec![0.0; n];
    let mut suffix = vec![0.0; n];
    let mut bi = 0;
    while bi < n {
        let be = (bi + w).min(n);
        suffix[bi] = arr[bi];
        for j in (bi + 1)..be {
            suffix[j] = if arr[j] < suffix[j - 1] {
                arr[j]
            } else {
                suffix[j - 1]
            };
        }
        prefix[be - 1] = arr[be - 1];
        if be >= 2 {
            for j in (bi..=(be - 2)).rev() {
                prefix[j] = if arr[j] < prefix[j + 1] {
                    arr[j]
                } else {
                    prefix[j + 1]
                };
            }
        }
        bi += w;
    }
    for i in (w - 1)..n {
        let p = prefix[i - w + 1];
        let s = suffix[i];
        out[i] = if p < s { p } else { s };
    }
    out
}

/// Compute MAMA/FAMA arrays for MESA strategy.
pub fn compute_mama_fama(close: &[f64], fl: f64, slow_lim: f64) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut mama = vec![0.0; n];
    let mut fama = vec![0.0; n];
    if n < 40 {
        return (mama, fama);
    }
    let mut smooth = vec![0.0; n];
    let mut det = vec![0.0; n];
    let mut i1 = vec![0.0; n];
    let mut q1 = vec![0.0; n];
    let mut ji = vec![0.0; n];
    let mut jq = vec![0.0; n];
    let mut i2 = vec![0.0; n];
    let mut q2 = vec![0.0; n];
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    let mut per = vec![0.0; n];
    let mut sp = vec![0.0; n];
    let mut ph = vec![0.0; n];

    for i in 0..6.min(n) {
        mama[i] = close[i];
        fama[i] = close[i];
        per[i] = 6.0;
        sp[i] = 6.0;
    }
    for i in 6..n {
        smooth[i] =
            (4.0 * close[i] + 3.0 * close[i - 1] + 2.0 * close[i - 2] + close[i - 3]) / 10.0;
        let adj = 0.075 * per[i - 1] + 0.54;
        det[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i.saturating_sub(2)]
            - 0.5769 * smooth[i.saturating_sub(4)]
            - 0.0962 * smooth[i.saturating_sub(6)])
            * adj;
        i1[i] = det[i.saturating_sub(3)];
        q1[i] = (0.0962 * det[i] + 0.5769 * det[i.saturating_sub(2)]
            - 0.5769 * det[i.saturating_sub(4)]
            - 0.0962 * det[i.saturating_sub(6)])
            * adj;
        ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i.saturating_sub(2)]
            - 0.5769 * i1[i.saturating_sub(4)]
            - 0.0962 * i1[i.saturating_sub(6)])
            * adj;
        jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i.saturating_sub(2)]
            - 0.5769 * q1[i.saturating_sub(4)]
            - 0.0962 * q1[i.saturating_sub(6)])
            * adj;
        i2[i] = i1[i] - jq[i];
        q2[i] = q1[i] + ji[i];
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1];
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1];
        re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1];
        im[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1];
        re[i] = 0.2 * re[i] + 0.8 * re[i - 1];
        im[i] = 0.2 * im[i] + 0.8 * im[i - 1];
        per[i] = if im[i] != 0.0 && re[i] != 0.0 {
            2.0 * std::f64::consts::PI / (im[i] / re[i]).atan()
        } else {
            per[i - 1]
        };
        if per[i] > 1.5 * per[i - 1] {
            per[i] = 1.5 * per[i - 1];
        }
        if per[i] < 0.67 * per[i - 1] {
            per[i] = 0.67 * per[i - 1];
        }
        if per[i] < 6.0 {
            per[i] = 6.0;
        }
        if per[i] > 50.0 {
            per[i] = 50.0;
        }
        per[i] = 0.2 * per[i] + 0.8 * per[i - 1];
        sp[i] = 0.33 * per[i] + 0.67 * sp[i - 1];
        ph[i] = if i1[i] != 0.0 {
            (q1[i] / i1[i]).atan() * 180.0 / std::f64::consts::PI
        } else {
            ph[i - 1]
        };
        let mut dp = ph[i - 1] - ph[i];
        if dp < 1.0 {
            dp = 1.0;
        }
        let mut alpha = fl / dp;
        if alpha < slow_lim {
            alpha = slow_lim;
        }
        if alpha > fl {
            alpha = fl;
        }
        mama[i] = alpha * close[i] + (1.0 - alpha) * mama[i - 1];
        fama[i] = 0.5 * alpha * mama[i] + (1.0 - 0.5 * alpha) * fama[i - 1];
    }
    (mama, fama)
}

/// Compute KAMA array for given parameters.
pub fn compute_kama(close: &[f64], er_p: usize, fast_sc: usize, slow_sc: usize) -> Vec<f64> {
    let n = close.len();
    let mut kama = vec![f64::NAN; n];
    if n < er_p + 2 {
        return kama;
    }
    let fc = 2.0 / (fast_sc as f64 + 1.0);
    let sc_v = 2.0 / (slow_sc as f64 + 1.0);
    kama[er_p - 1] = close[er_p - 1];
    for i in er_p..n {
        let d = (close[i] - close[i - er_p]).abs();
        let mut v = 0.0;
        for j in 1..=er_p {
            v += (close[i - j + 1] - close[i - j]).abs();
        }
        let er = if v > 0.0 { d / v } else { 0.0 };
        let sc2 = (er * (fc - sc_v) + sc_v).powi(2);
        kama[i] = kama[i - 1] + sc2 * (close[i] - kama[i - 1]);
    }
    kama
}
