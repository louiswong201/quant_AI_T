use super::helpers::CostParams;
use super::indicators;
use super::strategies;

/// Dispatch to the correct kernel by strategy name.
pub fn eval_kernel(
    name: &str,
    params: &[f64],
    c: &[f64], o: &[f64], h: &[f64], l: &[f64],
    cp: &CostParams,
) -> (f64, f64, i64) {
    match name {
        "MA" => {
            let ma_s = indicators::rolling_mean(c, params[0] as usize);
            let ma_l = indicators::rolling_mean(c, params[1] as usize);
            strategies::bt_ma_ls(c, o, h, l, &ma_s, &ma_l, cp)
        }
        "RSI" => {
            let rsi = indicators::rsi_wilder(c, params[0] as usize);
            strategies::bt_rsi_ls(c, o, h, l, &rsi, params[1], params[2], cp)
        }
        "MACD" => {
            let ef = indicators::ema(c, params[0] as usize);
            let es = indicators::ema(c, params[1] as usize);
            strategies::bt_macd_ls(c, o, h, l, &ef, &es, params[2] as usize, cp)
        }
        "Drift" => {
            strategies::bt_drift_ls(c, o, h, l, params[0] as usize, params[1], params[2] as usize, cp)
        }
        "RAMOM" => {
            strategies::bt_ramom_ls(c, o, h, l, params[0] as usize, params[1] as usize, params[2], params[3], cp)
        }
        "Turtle" => {
            strategies::bt_turtle_ls(c, o, h, l, params[0] as usize, params[1] as usize, params[2] as usize, params[3], cp)
        }
        "Bollinger" => {
            strategies::bt_bollinger_ls(c, o, h, l, params[0] as usize, params[1], cp)
        }
        "Keltner" => {
            strategies::bt_keltner_ls(c, o, h, l, params[0] as usize, params[1] as usize, params[2], cp)
        }
        "MultiFactor" => {
            strategies::bt_multifactor_ls(c, o, h, l, params[0] as usize, params[1] as usize, params[2] as usize, params[3], params[4], cp)
        }
        "VolRegime" => {
            let ms = (params[2] as usize).max(2);
            let ml = (params[3] as usize).max(5);
            if ms >= ml { return (0.0, 0.0, 0); }
            strategies::bt_volregime_ls(c, o, h, l, params[0] as usize, params[1], ms, ml, 14, params[4], params[5], cp)
        }
        "MESA" => {
            strategies::bt_mesa_ls(c, o, h, l, params[0], params[1], cp)
        }
        "KAMA" => {
            strategies::bt_kama_ls(c, o, h, l, params[0] as usize, params[1] as usize, params[2] as usize, params[3], params[4] as usize, cp)
        }
        "Donchian" => {
            strategies::bt_donchian_ls(c, o, h, l, params[0] as usize, params[1] as usize, params[2], cp)
        }
        "ZScore" => {
            strategies::bt_zscore_ls(c, o, h, l, params[0] as usize, params[1], params[2], params[3], cp)
        }
        "MomBreak" => {
            strategies::bt_mombreak_ls(c, o, h, l, params[0] as usize, params[1], params[2] as usize, params[3], cp)
        }
        "RegimeEMA" => {
            let fe = (params[2] as usize).max(2);
            let se = (params[3] as usize).max(5);
            if fe >= se { return (0.0, 0.0, 0); }
            strategies::bt_regime_ema_ls(c, o, h, l, (params[0] as usize).max(5), params[1], fe, se, params[4] as usize, cp)
        }
        "DualMom" => {
            let fl = (params[0] as usize).max(2);
            let sl = (params[1] as usize).max(5);
            if fl >= sl { return (0.0, 0.0, 0); }
            strategies::bt_dualmom_ls(c, o, h, l, fl, sl, cp)
        }
        "Consensus" => {
            let ms = (params[0] as usize).min(200).max(2);
            let ml = (params[1] as usize).min(200).max(5);
            if ms >= ml { return (0.0, 0.0, 0); }
            let rp = (params[2] as usize).min(200).max(2);
            let ma_s = indicators::rolling_mean(c, ms);
            let ma_l = indicators::rolling_mean(c, ml);
            let rsi = indicators::rsi_wilder(c, rp);
            strategies::bt_consensus_ls(c, o, h, l, &ma_s, &ma_l, &rsi, params[3] as usize, params[4], params[5], params[6] as i32, cp)
        }
        _ => (0.0, 0.0, 0),
    }
}
