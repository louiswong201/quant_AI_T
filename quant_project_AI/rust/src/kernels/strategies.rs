use super::helpers::*;
use super::indicators;

/// MA crossover strategy.
pub fn bt_ma_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    ma_s: &[f64], ma_l: &[f64], cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    for i in 1..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let (s0, l0, s1, l1) = (ma_s[i - 1], ma_l[i - 1], ma_s[i], ma_l[i]);
        if s0.is_nan() || l0.is_nan() || s1.is_nan() || l1.is_nan() { continue; }
        if s0 <= l0 && s1 > l1 {
            if st.pos == 0 { st.pend = 1; } else if st.pos == -1 { st.pend = 3; }
        } else if s0 >= l0 && s1 < l1 {
            if st.pos == 0 { st.pend = -1; } else if st.pos == 1 { st.pend = -3; }
        }
    }
    st.finalize(c, cp)
}

/// RSI strategy.
pub fn bt_rsi_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    rsi: &[f64], os_v: f64, ob_v: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    for i in 1..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let r = rsi[i];
        if r.is_nan() { continue; }
        if r < os_v {
            if st.pos == 0 { st.pend = 1; } else if st.pos == -1 { st.pend = 3; }
        } else if r > ob_v {
            if st.pos == 0 { st.pend = -1; } else if st.pos == 1 { st.pend = -3; }
        } else if st.pos == 1 && r > 50.0 {
            st.pend = 2;
        } else if st.pos == -1 && r < 50.0 {
            st.pend = 2;
        }
    }
    st.finalize(c, cp)
}

/// MACD strategy (standalone version computing indicators internally).
pub fn bt_macd_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    ef: &[f64], es: &[f64], sig_span: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut ml = vec![0.0; n];
    let mut sl_arr = vec![0.0; n];
    for i in 0..n { ml[i] = ef[i] - es[i]; }
    let k = 2.0 / (sig_span as f64 + 1.0);
    sl_arr[0] = ml[0];
    for i in 1..n { sl_arr[i] = ml[i] * k + sl_arr[i - 1] * (1.0 - k); }

    let mut st = SimState::new();
    for i in 1..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let (mp, sp, mc, sc) = (ml[i - 1], sl_arr[i - 1], ml[i], sl_arr[i]);
        if mp.is_nan() || sp.is_nan() || mc.is_nan() || sc.is_nan() { continue; }
        if mp <= sp && mc > sc {
            if st.pos == 0 { st.pend = 1; } else if st.pos == -1 { st.pend = 3; }
        } else if mp >= sp && mc < sc {
            if st.pos == 0 { st.pend = -1; } else if st.pos == 1 { st.pend = -3; }
        }
    }
    st.finalize(c, cp)
}

/// MACD with precomputed MACD line.
pub fn bt_macd_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    macd_line: &[f64], sig_span: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let k = 2.0 / (sig_span as f64 + 1.0);
    let k1 = 1.0 - k;
    let mut sp = macd_line[0];
    let mut st = SimState::new();
    for i in 1..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { sp = macd_line[i] * k + sp * k1; continue; }
        let mp = macd_line[i - 1];
        let mc = macd_line[i];
        let sc = mc * k + sp * k1;
        if !mp.is_nan() && !mc.is_nan() {
            if mp <= sp && mc > sc {
                if st.pos == 0 { st.pend = 1; } else if st.pos == -1 { st.pend = 3; }
            } else if mp >= sp && mc < sc {
                if st.pos == 0 { st.pend = -1; } else if st.pos == 1 { st.pend = -3; }
            }
        }
        sp = sc;
    }
    st.finalize(c, cp)
}

/// Drift strategy.
pub fn bt_drift_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    lookback: usize, drift_thr: f64, hold_p: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    let mut hc: usize = 0;
    for i in lookback..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { hc = 0; continue; }
        let mut up = 0usize;
        for j in 1..=lookback {
            if c[i - j + 1] > c[i - j] { up += 1; }
        }
        let ratio = up as f64 / lookback as f64;
        if st.pos != 0 {
            hc += 1;
            if hc >= hold_p { st.pend = 2; hc = 0; }
        }
        if st.pos == 0 && st.pend == 0 {
            if ratio >= drift_thr { st.pend = -1; hc = 0; }
            else if ratio <= 1.0 - drift_thr { st.pend = 1; hc = 0; }
        }
    }
    st.finalize(c, cp)
}

/// Drift with precomputed up-prefix.
pub fn bt_drift_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    up_prefix: &[i64], lookback: usize, drift_thr: f64, hold_p: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    let mut hc: usize = 0;
    for i in lookback..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { hc = 0; continue; }
        let up = up_prefix[i + 1] - up_prefix[i - lookback + 1];
        let ratio = up as f64 / lookback as f64;
        if st.pos != 0 {
            hc += 1;
            if hc >= hold_p { st.pend = 2; hc = 0; }
        }
        if st.pos == 0 && st.pend == 0 {
            if ratio >= drift_thr { st.pend = -1; hc = 0; }
            else if ratio <= 1.0 - drift_thr { st.pend = 1; hc = 0; }
        }
    }
    st.finalize(c, cp)
}

/// RAMOM strategy.
pub fn bt_ramom_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    mom_p: usize, vol_p: usize, ez: f64, xz: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    let start = mom_p.max(vol_p);
    for i in start..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let prev_r = c[i - mom_p];
        if prev_r <= 0.0 { continue; }
        let mom = c[i] / prev_r - 1.0;
        let mut s = 0.0;
        let mut s2 = 0.0;
        for j in 0..vol_p {
            let r = if i > j && c[i - j - 1] > 0.0 {
                c[i - j] / c[i - j - 1] - 1.0
            } else {
                0.0
            };
            s += r;
            s2 += r * r;
        }
        let m = if vol_p > 0 { s / vol_p as f64 } else { 0.0 };
        let vol = (s2 / (vol_p.max(1) as f64) - m * m).max(1e-20).sqrt();
        let z = mom / vol;
        if st.pos == 0 {
            if z > ez { st.pend = 1; } else if z < -ez { st.pend = -1; }
        } else if st.pos == 1 && z < xz {
            st.pend = 2;
        } else if st.pos == -1 && z > -xz {
            st.pend = 2;
        }
    }
    st.finalize(c, cp)
}

/// RAMOM with precomputed rolling volatility.
pub fn bt_ramom_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    mom_p: usize, vol_arr: &[f64], ez: f64, xz: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    for i in mom_p..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let prev = c[i - mom_p];
        if prev <= 0.0 { continue; }
        let mom = c[i] / prev - 1.0;
        let vol = vol_arr[i];
        if vol.is_nan() || vol < 1e-20 { continue; }
        let z = mom / vol;
        if st.pos == 0 {
            if z > ez { st.pend = 1; } else if z < -ez { st.pend = -1; }
        } else if st.pos == 1 && z < xz {
            st.pend = 2;
        } else if st.pos == -1 && z > -xz {
            st.pend = 2;
        }
    }
    st.finalize(c, cp)
}

/// Turtle strategy.
pub fn bt_turtle_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    entry_p: usize, exit_p: usize, atr_p: usize, atr_stop: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let aa = indicators::atr(h, l, c, atr_p);
    let mut st = SimState::new();
    let start = entry_p.max(exit_p).max(atr_p);
    for i in start..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let mut eh = f64::NEG_INFINITY;
        let mut el = f64::INFINITY;
        for j in 1..=entry_p {
            if h[i - j] > eh { eh = h[i - j]; }
            if l[i - j] < el { el = l[i - j]; }
        }
        let mut xl = f64::INFINITY;
        let mut xh = f64::NEG_INFINITY;
        for j in 1..=exit_p {
            if l[i - j] < xl { xl = l[i - j]; }
            if h[i - j] > xh { xh = h[i - j]; }
        }
        let a = aa[i];
        if a.is_nan() { continue; }
        if st.pos == 1 {
            if c[i] < st.ep / cp.sb - atr_stop * a || c[i] < xl { st.pend = 2; }
        } else if st.pos == -1 {
            if c[i] > st.ep / cp.ss + atr_stop * a || c[i] > xh { st.pend = 2; }
        }
        if st.pos == 0 && st.pend == 0 {
            if c[i] > eh { st.pend = 1; } else if c[i] < el { st.pend = -1; }
        }
    }
    st.finalize(c, cp)
}

/// Turtle with precomputed arrays.
pub fn bt_turtle_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    atr_arr: &[f64], rmax_entry: &[f64], rmin_entry: &[f64],
    rmin_exit: &[f64], rmax_exit: &[f64],
    entry_p: usize, exit_p: usize, atr_stop: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    let start = entry_p.max(exit_p);
    for i in start..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let eh = rmax_entry[i - 1];
        let el = rmin_entry[i - 1];
        let xl = rmin_exit[i - 1];
        let xh = rmax_exit[i - 1];
        let a = atr_arr[i];
        if a.is_nan() || eh.is_nan() || el.is_nan() { continue; }
        if st.pos == 1 && cp.sb > 0.0 {
            if c[i] < st.ep / cp.sb - atr_stop * a || c[i] < xl { st.pend = 2; }
        } else if st.pos == -1 && cp.ss > 0.0 {
            if c[i] > st.ep / cp.ss + atr_stop * a || c[i] > xh { st.pend = 2; }
        }
        if st.pos == 0 && st.pend == 0 {
            if c[i] > eh { st.pend = 1; } else if c[i] < el { st.pend = -1; }
        }
    }
    st.finalize(c, cp)
}

/// Bollinger Bands strategy.
pub fn bt_bollinger_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    period: usize, num_std: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let ma = indicators::rolling_mean(c, period);
    let sd = indicators::rolling_std(c, period);
    bt_bollinger_precomp(c, o, h, l, &ma, &sd, num_std, period, cp)
}

/// Bollinger with precomputed MA and std.
pub fn bt_bollinger_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    ma_arr: &[f64], std_arr: &[f64], num_std: f64, period: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    for i in period..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let m = ma_arr[i];
        let s = std_arr[i];
        if m.is_nan() || s.is_nan() || s < 1e-10 { continue; }
        let u = m + num_std * s;
        let lo = m - num_std * s;
        if st.pos == 0 {
            if c[i] < lo { st.pend = 1; } else if c[i] > u { st.pend = -1; }
        } else if st.pos == 1 && c[i] >= m {
            st.pend = 2;
        } else if st.pos == -1 && c[i] <= m {
            st.pend = 2;
        }
    }
    st.finalize(c, cp)
}

/// Keltner Channel strategy.
pub fn bt_keltner_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    ema_p: usize, atr_p: usize, atr_m: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let ea = indicators::ema(c, ema_p);
    let aa = indicators::atr(h, l, c, atr_p);
    bt_keltner_precomp(c, o, h, l, &ea, &aa, atr_m, ema_p, atr_p, cp)
}

/// Keltner with precomputed EMA and ATR.
pub fn bt_keltner_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    ema_arr: &[f64], atr_arr: &[f64], atr_m: f64,
    _ema_p: usize, atr_p: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    let start = _ema_p.max(atr_p);
    for i in start..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let e = ema_arr[i];
        let a = atr_arr[i];
        if e.is_nan() || a.is_nan() { continue; }
        if st.pos == 0 {
            if c[i] > e + atr_m * a { st.pend = 1; } else if c[i] < e - atr_m * a { st.pend = -1; }
        } else if st.pos == 1 && c[i] < e {
            st.pend = 2;
        } else if st.pos == -1 && c[i] > e {
            st.pend = 2;
        }
    }
    st.finalize(c, cp)
}

/// MultiFactor strategy.
pub fn bt_multifactor_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    rsi_p: usize, mom_p: usize, vol_p: usize, lt: f64, st_thr: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let rsi = indicators::rsi_wilder(c, rsi_p);
    let n = c.len();
    let mut st = SimState::new();
    let start = rsi_p.max(mom_p).max(vol_p).max(2);
    for i in start..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let r = rsi[i];
        if r.is_nan() { continue; }
        let prev_mf = c[i - mom_p];
        let mom = if prev_mf > 0.0 { c[i] / prev_mf - 1.0 } else { 0.0 };
        let rs = (100.0 - r) / 100.0;
        let ms = (mom.max(-0.5).min(0.5)) + 0.5;
        let mut s2 = 0.0;
        for j in 0..vol_p {
            let ret = if i > j && c[i - j - 1] > 0.0 { c[i - j] / c[i - j - 1] - 1.0 } else { 0.0 };
            s2 += ret * ret;
        }
        let vs = (1.0 - (s2 / vol_p.max(1) as f64).sqrt() * 20.0).max(0.0);
        let comp = (rs + ms + vs) / 3.0;
        if st.pos == 0 {
            if comp > lt { st.pend = 1; } else if comp < st_thr { st.pend = -1; }
        } else if st.pos == 1 && comp < 0.5 {
            st.pend = 2;
        } else if st.pos == -1 && comp > 0.5 {
            st.pend = 2;
        }
    }
    st.finalize(c, cp)
}

/// MultiFactor with precomputed RSI and vol.
pub fn bt_multifactor_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    rsi_arr: &[f64], mom_p: usize, vol_arr: &[f64], lt: f64, st_thr: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    let start = mom_p.max(2);
    for i in start..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let r = rsi_arr[i];
        if r.is_nan() { continue; }
        let rs = (100.0 - r) / 100.0;
        let prev_mf = c[i - mom_p];
        let mom = if prev_mf > 0.0 { c[i] / prev_mf - 1.0 } else { 0.0 };
        let ms = (mom.max(-0.5).min(0.5)) + 0.5;
        let v = vol_arr[i];
        let vs = if !v.is_nan() { (1.0 - v * 20.0).max(0.0) } else { 0.5 };
        let comp = (rs + ms + vs) / 3.0;
        if st.pos == 0 {
            if comp > lt { st.pend = 1; } else if comp < st_thr { st.pend = -1; }
        } else if st.pos == 1 && comp < 0.5 {
            st.pend = 2;
        } else if st.pos == -1 && comp > 0.5 {
            st.pend = 2;
        }
    }
    st.finalize(c, cp)
}

/// VolRegime strategy.
pub fn bt_volregime_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    atr_p: usize, vol_thr: f64, ma_s: usize, ma_l: usize, _rsi_p: usize,
    rsi_os: f64, rsi_ob: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let aa = indicators::atr(h, l, c, atr_p);
    let ra = indicators::rsi_wilder(c, 14);
    let ms_a = indicators::rolling_mean(c, ma_s);
    let ml_a = indicators::rolling_mean(c, ma_l);
    bt_volregime_precomp(c, o, h, l, &aa, &ra, &ms_a, &ml_a, vol_thr, rsi_os, rsi_ob,
                         atr_p.max(15).max(ma_l), cp)
}

/// VolRegime with precomputed arrays.
pub fn bt_volregime_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    atr_arr: &[f64], rsi_arr: &[f64], ma_s_arr: &[f64], ma_l_arr: &[f64],
    vol_thr: f64, rsi_os: f64, rsi_ob: f64, start: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    let mut mode: i32 = 0;
    for i in start..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let a = atr_arr[i];
        if a.is_nan() || c[i] <= 0.0 { continue; }
        let hv = a / c[i] > vol_thr;
        if hv {
            let r = rsi_arr[i];
            if !r.is_nan() {
                if st.pos == 0 {
                    if r < rsi_os { st.pend = 1; mode = 1; }
                    else if r > rsi_ob { st.pend = -1; mode = 1; }
                } else if st.pos == 1 && mode == 1 && r > 50.0 { st.pend = 2; }
                else if st.pos == -1 && mode == 1 && r < 50.0 { st.pend = 2; }
            }
        } else {
            let s_ = ma_s_arr[i]; let l_ = ma_l_arr[i];
            let s0 = ma_s_arr[i - 1]; let l0 = ma_l_arr[i - 1];
            if !s_.is_nan() && !l_.is_nan() && !s0.is_nan() && !l0.is_nan() {
                if st.pos == 0 {
                    if s0 <= l0 && s_ > l_ { st.pend = 1; mode = 0; }
                    else if s0 >= l0 && s_ < l_ { st.pend = -1; mode = 0; }
                } else if st.pos == 1 && mode == 0 && s_ < l_ { st.pend = 2; }
                else if st.pos == -1 && mode == 0 && s_ > l_ { st.pend = 2; }
            }
        }
        if st.pos != 0 && st.pend == 0 {
            if (mode == 0 && hv) || (mode == 1 && !hv) { st.pend = 2; }
        }
    }
    st.finalize(c, cp)
}

/// MESA strategy.
pub fn bt_mesa_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    fl: f64, slow_lim: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    if n < 40 { return (0.0, 0.0, 0); }
    let (mama, fama) = indicators::compute_mama_fama(c, fl, slow_lim);
    bt_mesa_precomp(c, o, h, l, fl, slow_lim, &mama, &fama, cp)
}

/// MESA with precomputed MAMA/FAMA (reapplied with fl/slow_lim).
pub fn bt_mesa_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    _fl: f64, _slow_lim: f64,
    mama: &[f64], fama: &[f64], cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    for i in 7..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        if st.pos == 0 {
            if mama[i] > fama[i] && mama[i - 1] <= fama[i - 1] { st.pend = 1; }
            else if mama[i] < fama[i] && mama[i - 1] >= fama[i - 1] { st.pend = -1; }
        } else if st.pos == 1 && mama[i] < fama[i] && mama[i - 1] >= fama[i - 1] {
            st.pend = -3;
        } else if st.pos == -1 && mama[i] > fama[i] && mama[i - 1] <= fama[i - 1] {
            st.pend = 3;
        }
    }
    st.finalize(c, cp)
}

/// KAMA strategy.
pub fn bt_kama_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    er_p: usize, fast_sc: usize, slow_sc: usize, atr_sm: f64, atr_p: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    if n < er_p + 2 { return (0.0, 0.0, 0); }
    let kama = indicators::compute_kama(c, er_p, fast_sc, slow_sc);
    let av = indicators::atr(h, l, c, atr_p);
    bt_kama_precomp(c, o, h, l, &kama, &av, atr_sm, er_p, atr_p, cp)
}

/// KAMA with precomputed arrays.
pub fn bt_kama_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    kama_arr: &[f64], atr_arr: &[f64], atr_sm: f64,
    er_p: usize, atr_p: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    let start = (er_p + 2).max(atr_p);
    for i in start..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let k = kama_arr[i]; let kp = kama_arr[i - 1]; let a = atr_arr[i];
        if k.is_nan() || kp.is_nan() || a.is_nan() { continue; }
        if st.pos == 1 {
            if c[i] < st.ep / cp.sb - atr_sm * a || k < kp { st.pend = 2; }
        } else if st.pos == -1 {
            if c[i] > st.ep / cp.ss + atr_sm * a || k > kp { st.pend = 2; }
        }
        if st.pos == 0 && st.pend == 0 {
            if c[i] > k && k > kp { st.pend = 1; }
            else if c[i] < k && k < kp { st.pend = -1; }
        }
    }
    st.finalize(c, cp)
}

/// Donchian Channel strategy.
pub fn bt_donchian_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    entry_p: usize, atr_p: usize, atr_m: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    if n < entry_p + atr_p { return (0.0, 0.0, 0); }
    let av = indicators::atr(h, l, c, atr_p);
    let mut dh = vec![f64::NAN; n];
    let mut dl = vec![f64::NAN; n];
    for i in (entry_p - 1)..n {
        let mut mx = h[i]; let mut mn = l[i];
        for j in 1..entry_p {
            if h[i - j] > mx { mx = h[i - j]; }
            if l[i - j] < mn { mn = l[i - j]; }
        }
        dh[i] = mx; dl[i] = mn;
    }

    let mut st = SimState::new();
    let mut ts = 0.0f64;
    let start = entry_p.max(atr_p);
    for i in start..n {
        let (pos, ep, tr, tc, liq) = fx_lev(st.pend, st.pos, st.ep, o[i], st.tr, cp);
        if st.pend == 1 && !av[i].is_nan() { ts = o[i] - atr_m * av[i]; }
        else if st.pend == -1 && !av[i].is_nan() { ts = o[i] + atr_m * av[i]; }
        st.pos = pos; st.ep = ep; st.tr = tr; st.nt += tc; st.pend = 0;
        if liq { st.pos = 0; st.ep = 0.0; continue; }
        let (pos2, ep2, tr2, tc2) = sl_exit(st.pos, st.ep, st.tr, h[i], l[i], cp);
        st.pos = pos2; st.ep = ep2; st.tr = tr2; st.nt += tc2;
        let d1 = dh[i - 1]; let d2 = dl[i - 1]; let a = av[i];
        if !d1.is_nan() && !d2.is_nan() && !a.is_nan() {
            if st.pos == 1 {
                let ns = c[i] - atr_m * a;
                if ns > ts { ts = ns; }
                if c[i] < ts { st.pend = 2; }
            } else if st.pos == -1 {
                let ns = c[i] + atr_m * a;
                if ns < ts { ts = ns; }
                if c[i] > ts { st.pend = 2; }
            }
            if st.pos == 0 && st.pend == 0 {
                if c[i] > d1 { st.pend = 1; } else if c[i] < d2 { st.pend = -1; }
            }
        }
        let eq = mtm_lev(st.pos, st.tr, c[i], st.ep, cp);
        st.track_dd(eq);
    }
    st.finalize(c, cp)
}

/// Donchian with precomputed rolling H/L.
pub fn bt_donchian_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    dh_arr: &[f64], dl_arr: &[f64], atr_arr: &[f64], atr_m: f64,
    entry_p: usize, _atr_p: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    let mut ts = 0.0f64;
    let start = entry_p;
    for i in start..n {
        let (pos, ep, tr, tc, liq) = fx_lev(st.pend, st.pos, st.ep, o[i], st.tr, cp);
        if st.pend == 1 && !atr_arr[i].is_nan() { ts = o[i] - atr_m * atr_arr[i]; }
        else if st.pend == -1 && !atr_arr[i].is_nan() { ts = o[i] + atr_m * atr_arr[i]; }
        st.pos = pos; st.ep = ep; st.tr = tr; st.nt += tc; st.pend = 0;
        if liq { st.pos = 0; st.ep = 0.0; continue; }
        let (pos2, ep2, tr2, tc2) = sl_exit(st.pos, st.ep, st.tr, h[i], l[i], cp);
        st.pos = pos2; st.ep = ep2; st.tr = tr2; st.nt += tc2;
        let d1 = dh_arr[i - 1]; let d2 = dl_arr[i - 1]; let a = atr_arr[i];
        if !d1.is_nan() && !d2.is_nan() && !a.is_nan() {
            if st.pos == 1 {
                let ns = c[i] - atr_m * a;
                if ns > ts { ts = ns; }
                if c[i] < ts { st.pend = 2; }
            } else if st.pos == -1 {
                let ns = c[i] + atr_m * a;
                if ns < ts { ts = ns; }
                if c[i] > ts { st.pend = 2; }
            }
            if st.pos == 0 && st.pend == 0 {
                if c[i] > d1 { st.pend = 1; } else if c[i] < d2 { st.pend = -1; }
            }
        }
        let eq = mtm_lev(st.pos, st.tr, c[i], st.ep, cp);
        st.track_dd(eq);
    }
    st.finalize(c, cp)
}

/// ZScore strategy.
pub fn bt_zscore_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    lookback: usize, ez: f64, xz: f64, sz: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    if n < lookback + 2 { return (0.0, 0.0, 0); }
    let rm = indicators::rolling_mean(c, lookback);
    let rs = indicators::rolling_std(c, lookback);
    bt_zscore_precomp(c, o, h, l, &rm, &rs, lookback, ez, xz, sz, cp)
}

/// ZScore with precomputed mean and std.
pub fn bt_zscore_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    rm_arr: &[f64], rs_arr: &[f64], lookback: usize,
    ez: f64, xz: f64, sz: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    for i in lookback..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let m = rm_arr[i]; let sd = rs_arr[i];
        if sd == 0.0 || sd.is_nan() { continue; }
        let z = (c[i] - m) / sd;
        if st.pos == 1 && (z > -xz || z > sz) { st.pend = 2; }
        else if st.pos == -1 && (z < xz || z < -sz) { st.pend = 2; }
        if st.pos == 0 && st.pend == 0 {
            if z < -ez { st.pend = 1; } else if z > ez { st.pend = -1; }
        }
    }
    st.finalize(c, cp)
}

/// MomBreak strategy.
pub fn bt_mombreak_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    hp: usize, prox: f64, atr_p: usize, atr_t: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    if n < hp.max(atr_p) + 2 { return (0.0, 0.0, 0); }
    let mut rh = vec![f64::NAN; n]; let mut rl = vec![f64::NAN; n];
    for i in (hp - 1)..n {
        let mut mx = h[i]; let mut mn = l[i];
        for j in 1..hp {
            if h[i - j] > mx { mx = h[i - j]; }
            if l[i - j] < mn { mn = l[i - j]; }
        }
        rh[i] = mx; rl[i] = mn;
    }
    let av = indicators::atr(h, l, c, atr_p);
    bt_mombreak_precomp(c, o, h, l, &av, &rh, &rl, hp, prox, atr_t, cp)
}

/// MomBreak with precomputed arrays.
pub fn bt_mombreak_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    atr_arr: &[f64], rh: &[f64], rl: &[f64],
    hp: usize, prox: f64, atr_t: f64, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    let mut ts_l = 0.0f64;
    let mut ts_s = 1e18f64;
    let start = hp;
    for i in start..n {
        let (pos, ep, tr, tc, liq) = fx_lev(st.pend, st.pos, st.ep, o[i], st.tr, cp);
        if st.pend == 1 && !atr_arr[i].is_nan() { ts_l = o[i] - atr_t * atr_arr[i]; }
        else if st.pend == -1 && !atr_arr[i].is_nan() { ts_s = o[i] + atr_t * atr_arr[i]; }
        st.pos = pos; st.ep = ep; st.tr = tr; st.nt += tc; st.pend = 0;
        if liq { st.pos = 0; st.ep = 0.0; continue; }
        let (pos2, ep2, tr2, tc2) = sl_exit(st.pos, st.ep, st.tr, h[i], l[i], cp);
        st.pos = pos2; st.ep = ep2; st.tr = tr2; st.nt += tc2;
        let hv = rh[i]; let lv = rl[i]; let a = atr_arr[i];
        if !hv.is_nan() && !lv.is_nan() && !a.is_nan() {
            if st.pos == 1 {
                let ns = c[i] - atr_t * a;
                if ns > ts_l { ts_l = ns; }
                if c[i] < ts_l { st.pend = 2; }
            } else if st.pos == -1 {
                let ns = c[i] + atr_t * a;
                if ns < ts_s { ts_s = ns; }
                if c[i] > ts_s { st.pend = 2; }
            }
            if st.pos == 0 && st.pend == 0 {
                if c[i] >= hv * (1.0 - prox) { st.pend = 1; }
                else if c[i] <= lv * (1.0 + prox) { st.pend = -1; }
            }
        }
        let eq = mtm_lev(st.pos, st.tr, c[i], st.ep, cp);
        st.track_dd(eq);
    }
    st.finalize(c, cp)
}

/// RegimeEMA strategy.
pub fn bt_regime_ema_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    atr_p: usize, vt: f64, fe_p: usize, se_p: usize, te_p: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    if n < atr_p.max(se_p.max(te_p)) + 2 { return (0.0, 0.0, 0); }
    let av = indicators::atr(h, l, c, atr_p);
    let ef = indicators::ema(c, fe_p);
    let es = indicators::ema(c, se_p);
    let et = indicators::ema(c, te_p);
    bt_regime_ema_precomp(c, o, h, l, &av, &ef, &es, &et, vt, atr_p, se_p, te_p, cp)
}

/// RegimeEMA with precomputed arrays.
pub fn bt_regime_ema_precomp(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    atr_arr: &[f64], ef: &[f64], es: &[f64], et: &[f64],
    vt: f64, atr_p: usize, se_p: usize, te_p: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let mut st = SimState::new();
    let start = atr_p.max(se_p.max(te_p));
    for i in start..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let a = atr_arr[i];
        if a.is_nan() || c[i] <= 0.0 { continue; }
        let hv = a / c[i] > vt;
        if hv {
            if st.pos == 0 {
                if ef[i] > es[i] && ef[i - 1] <= es[i - 1] { st.pend = 1; }
                else if ef[i] < es[i] && ef[i - 1] >= es[i - 1] { st.pend = -1; }
            } else if st.pos == 1 && ef[i] < es[i] { st.pend = 2; }
            else if st.pos == -1 && ef[i] > es[i] { st.pend = 2; }
        } else {
            let pd = if et[i].abs() > 1e-20 { (c[i] - et[i]) / et[i] } else { 0.0 };
            if st.pos == 0 {
                if pd < -0.02 { st.pend = 1; } else if pd > 0.02 { st.pend = -1; }
            } else if st.pos == 1 && pd > 0.0 { st.pend = 2; }
            else if st.pos == -1 && pd < 0.0 { st.pend = 2; }
        }
    }
    st.finalize(c, cp)
}

/// DualMom strategy.
pub fn bt_dualmom_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    fast_lb: usize, slow_lb: usize, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    let lb = fast_lb.max(slow_lb);
    if n < lb + 2 { return (0.0, 0.0, 0); }
    let mut st = SimState::new();
    for i in lb..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let pf = c[i - fast_lb]; let ps = c[i - slow_lb];
        let fast_ret = if pf > 0.0 { (c[i] - pf) / pf } else { 0.0 };
        let slow_ret = if ps > 0.0 { (c[i] - ps) / ps } else { 0.0 };
        if st.pos == 0 {
            if fast_ret > 0.0 && slow_ret > 0.0 { st.pend = 1; }
            else if fast_ret < 0.0 && slow_ret < 0.0 { st.pend = -1; }
        } else if st.pos == 1 && (fast_ret < 0.0 || slow_ret < 0.0) {
            st.pend = 2;
        } else if st.pos == -1 && (fast_ret > 0.0 || slow_ret > 0.0) {
            st.pend = 2;
        }
    }
    st.finalize(c, cp)
}

/// Consensus strategy.
pub fn bt_consensus_ls(
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    ma_s_arr: &[f64], ma_l_arr: &[f64], rsi_arr: &[f64],
    mom_lb: usize, rsi_os: f64, rsi_ob: f64, vote_thr: i32, cp: &CostParams,
) -> (f64, f64, i64) {
    let n = c.len();
    if n < mom_lb + 2 { return (0.0, 0.0, 0); }
    let mut st = SimState::new();
    for i in mom_lb..n {
        if process_bar(&mut st, o[i], h[i], l[i], c[i], cp) { continue; }
        let ms = ma_s_arr[i]; let ml = ma_l_arr[i]; let r = rsi_arr[i];
        if ms.is_nan() || ml.is_nan() || r.is_nan() { continue; }
        let mut v: i32 = 0;
        if ms > ml { v += 1; } else if ms < ml { v -= 1; }
        if r < rsi_os { v += 1; } else if r > rsi_ob { v -= 1; }
        let mom_ret = if c[i - mom_lb] > 0.0 {
            (c[i] - c[i - mom_lb]) / c[i - mom_lb]
        } else { 0.0 };
        if mom_ret > 0.02 { v += 1; } else if mom_ret < -0.02 { v -= 1; }
        if st.pos == 0 {
            if v >= vote_thr { st.pend = 1; } else if v <= -vote_thr { st.pend = -1; }
        } else if st.pos == 1 && v <= -1 { st.pend = 2; }
        else if st.pos == -1 && v >= 1 { st.pend = 2; }
    }
    st.finalize(c, cp)
}
