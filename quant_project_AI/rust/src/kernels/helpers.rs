#[derive(Clone, Copy)]
pub struct CostParams {
    pub sb: f64,
    pub ss: f64,
    pub cm: f64,
    pub lev: f64,
    pub dc: f64,
    pub dc_short: f64,
    pub sl: f64,
    pub pfrac: f64,
    pub sl_slip: f64,
}

#[derive(Clone, Copy)]
pub struct SimState {
    pub pos: i32,
    pub ep: f64,
    pub tr: f64,
    pub pend: i32,
    pub pk: f64,
    pub mdd: f64,
    pub nt: i64,
}

impl SimState {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            pos: 0,
            ep: 0.0,
            tr: 1.0,
            pend: 0,
            pk: 1.0,
            mdd: 0.0,
            nt: 0,
        }
    }

    #[inline(always)]
    pub fn finalize(mut self, c: &[f64], cp: &CostParams) -> (f64, f64, i64) {
        let n = c.len();
        if self.pos == 1 && self.ep > 0.0 {
            let raw = (c[n - 1] * cp.ss * (1.0 - cp.cm)) / (self.ep * (1.0 + cp.cm));
            let dep = deploy(self.tr, cp.pfrac);
            self.tr += dep * ((raw - 1.0) * cp.lev);
            if self.tr < 0.01 {
                self.tr = 0.01;
            }
            self.nt += 1;
        } else if self.pos == -1 && self.ep > 0.0 {
            let raw = (self.ep * (1.0 - cp.cm)) / (c[n - 1] * cp.sb * (1.0 + cp.cm));
            let dep = deploy(self.tr, cp.pfrac);
            self.tr += dep * ((raw - 1.0) * cp.lev);
            if self.tr < 0.01 {
                self.tr = 0.01;
            }
            self.nt += 1;
        }
        ((self.tr - 1.0) * 100.0, self.mdd, self.nt)
    }

    #[inline(always)]
    pub fn track_dd(&mut self, eq: f64) {
        if eq > self.pk {
            self.pk = eq;
        }
        if self.pk > 0.0 {
            let dd = (self.pk - eq) / self.pk * 100.0;
            if dd > self.mdd {
                self.mdd = dd;
            }
        }
    }
}

#[inline(always)]
pub fn deploy(tr: f64, pfrac: f64) -> f64 {
    let mut d = tr * pfrac;
    if tr > 1.0 && d > pfrac {
        d = pfrac;
    }
    d
}

/// Fill execution and daily cost deduction.
/// Returns (pos, ep, tr, trade_count, liquidated).
#[inline(always)]
pub fn fx_lev(
    pend: i32,
    mut pos: i32,
    mut ep: f64,
    oi: f64,
    mut tr: f64,
    cp: &CostParams,
) -> (i32, f64, f64, i64, bool) {
    if pos != 0 {
        let deployed = deploy(tr, cp.pfrac);
        let mut cost = deployed * cp.dc;
        if pos < 0 {
            cost += deployed * cp.dc_short;
        }
        tr -= cost;
        if tr < 0.01 {
            return (0, 0.0, 0.01, 0, true);
        }
    }
    if pend == 0 {
        return (pos, ep, tr, 0, false);
    }
    let mut tc: i64 = 0;
    if pend.abs() >= 2 {
        let deployed = deploy(tr, cp.pfrac);
        if pos == 1 && ep > 0.0 {
            let raw = (oi * cp.ss * (1.0 - cp.cm)) / (ep * (1.0 + cp.cm));
            let pnl = (raw - 1.0) * cp.lev;
            tr += deployed * pnl;
            if tr < 0.01 {
                tr = 0.01;
            }
            tc = 1;
        } else if pos == -1 && ep > 0.0 && oi > 0.0 {
            let raw = (ep * (1.0 - cp.cm)) / (oi * cp.sb * (1.0 + cp.cm));
            let pnl = (raw - 1.0) * cp.lev;
            tr += deployed * pnl;
            if tr < 0.01 {
                tr = 0.01;
            }
            tc = 1;
        }
        pos = 0;
    }
    if pend == 1 || pend == 3 {
        ep = oi * cp.sb;
        pos = 1;
    } else if pend == -1 || pend == -3 {
        ep = oi * cp.ss;
        pos = -1;
    }
    (pos, ep, tr, tc, false)
}

/// Stop-loss exit check using intra-bar high/low.
/// Returns (pos, ep, tr, trade_count).
#[inline(always)]
pub fn sl_exit(
    pos: i32,
    ep: f64,
    mut tr: f64,
    hi: f64,
    li: f64,
    cp: &CostParams,
) -> (i32, f64, f64, i64) {
    if pos == 0 || ep <= 0.0 {
        return (pos, ep, tr, 0);
    }
    let pnl = if pos == 1 {
        let raw = (li * cp.ss * (1.0 - cp.cm)) / (ep * (1.0 + cp.cm));
        (raw - 1.0) * cp.lev
    } else {
        let denom = hi * cp.sb * (1.0 + cp.cm);
        let raw = if denom > 0.0 {
            (ep * (1.0 - cp.cm)) / denom
        } else {
            1.0
        };
        (raw - 1.0) * cp.lev
    };
    if pnl >= -cp.sl {
        return (pos, ep, tr, 0);
    }
    let actual_loss = cp.sl + cp.sl_slip;
    let deployed = deploy(tr, cp.pfrac);
    tr -= deployed * actual_loss;
    if tr < 0.01 {
        tr = 0.01;
    }
    (0, 0.0, tr, 1)
}

/// Mark-to-market equity with leverage.
#[inline(always)]
pub fn mtm_lev(pos: i32, tr: f64, ci: f64, ep: f64, cp: &CostParams) -> f64 {
    let deployed = deploy(tr, cp.pfrac);
    if pos == 1 && ep > 0.0 {
        let raw = (ci * cp.ss * (1.0 - cp.cm)) / (ep * (1.0 + cp.cm));
        let mut pnl = (raw - 1.0) * cp.lev;
        if pnl < -cp.sl {
            pnl = -cp.sl;
        }
        return tr + deployed * pnl;
    }
    if pos == -1 && ep > 0.0 && ci > 0.0 {
        let raw = (ep * (1.0 - cp.cm)) / (ci * cp.sb * (1.0 + cp.cm));
        let mut pnl = (raw - 1.0) * cp.lev;
        if pnl < -cp.sl {
            pnl = -cp.sl;
        }
        return tr + deployed * pnl;
    }
    tr
}

#[inline(always)]
pub fn score(ret: f64, dd: f64, nt: i64) -> f64 {
    let dd_denom = if dd > 1.0 { dd } else { 1.0 };
    let nt_factor = if (nt as f64) < 20.0 {
        nt as f64 / 20.0
    } else {
        1.0
    };
    ret / dd_denom * nt_factor
}

/// Common bar processing: fx_lev + sl_exit. Returns true if liquidated.
#[inline(always)]
pub fn process_bar(
    st: &mut SimState,
    oi: f64,
    hi: f64,
    li: f64,
    ci: f64,
    cp: &CostParams,
) -> bool {
    let (pos, ep, tr, tc, liq) = fx_lev(st.pend, st.pos, st.ep, oi, st.tr, cp);
    st.pos = pos;
    st.ep = ep;
    st.tr = tr;
    st.nt += tc;
    st.pend = 0;
    if liq {
        st.pos = 0;
        st.ep = 0.0;
        return true;
    }
    let (pos2, ep2, tr2, tc2) = sl_exit(st.pos, st.ep, st.tr, hi, li, cp);
    st.pos = pos2;
    st.ep = ep2;
    st.tr = tr2;
    st.nt += tc2;
    let eq = mtm_lev(st.pos, st.tr, ci, st.ep, cp);
    st.track_dd(eq);
    false
}
