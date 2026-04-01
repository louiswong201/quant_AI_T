/// Welford's online algorithm for incremental mean/variance.
pub struct WelfordAccumulator {
    count: usize,
    mean: f64,
    m2: f64,
}

impl WelfordAccumulator {
    pub fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }

    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn remove(&mut self, x: f64) {
        if self.count == 0 { return; }
        let delta = x - self.mean;
        self.count -= 1;
        if self.count == 0 {
            self.mean = 0.0;
            self.m2 = 0.0;
            return;
        }
        self.mean -= delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 -= delta * delta2;
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        (self.m2 / self.count as f64).max(0.0)
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

/// Incremental RSI state (Wilder's smoothing).
pub struct IncrementalRsi {
    period: usize,
    avg_gain: f64,
    avg_loss: f64,
    prev_close: f64,
    count: usize,
    initial_gains: f64,
    initial_losses: f64,
}

impl IncrementalRsi {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            avg_gain: 0.0,
            avg_loss: 0.0,
            prev_close: f64::NAN,
            count: 0,
            initial_gains: 0.0,
            initial_losses: 0.0,
        }
    }

    pub fn update(&mut self, close: f64) -> f64 {
        if self.prev_close.is_nan() {
            self.prev_close = close;
            self.count = 1;
            return f64::NAN;
        }
        self.count += 1;
        let d = close - self.prev_close;
        self.prev_close = close;
        let gain = if d > 0.0 { d } else { 0.0 };
        let loss = if d < 0.0 { -d } else { 0.0 };

        if self.count <= self.period {
            self.initial_gains += gain;
            self.initial_losses += loss;
            if self.count == self.period {
                self.avg_gain = self.initial_gains / self.period as f64;
                self.avg_loss = self.initial_losses / self.period as f64;
                return if self.avg_loss == 0.0 {
                    100.0
                } else {
                    100.0 - 100.0 / (1.0 + self.avg_gain / self.avg_loss)
                };
            }
            return f64::NAN;
        }

        let pf = self.period as f64;
        self.avg_gain = (self.avg_gain * (pf - 1.0) + gain) / pf;
        self.avg_loss = (self.avg_loss * (pf - 1.0) + loss) / pf;
        if self.avg_loss == 0.0 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + self.avg_gain / self.avg_loss)
        }
    }
}

/// Incremental ATR state.
pub struct IncrementalAtr {
    period: usize,
    prev_close: f64,
    count: usize,
    initial_sum: f64,
    atr_value: f64,
}

impl IncrementalAtr {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            prev_close: f64::NAN,
            count: 0,
            initial_sum: 0.0,
            atr_value: f64::NAN,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let tr = if self.prev_close.is_nan() {
            high - low
        } else {
            (high - low)
                .max((high - self.prev_close).abs())
                .max((low - self.prev_close).abs())
        };
        self.prev_close = close;
        self.count += 1;

        if self.count <= self.period {
            self.initial_sum += tr;
            if self.count == self.period {
                self.atr_value = self.initial_sum / self.period as f64;
                return self.atr_value;
            }
            return f64::NAN;
        }

        let pf = self.period as f64;
        self.atr_value = (self.atr_value * (pf - 1.0) + tr) / pf;
        self.atr_value
    }
}

/// Incremental EMA state.
pub struct IncrementalEma {
    span: usize,
    k: f64,
    value: f64,
    initialized: bool,
}

impl IncrementalEma {
    pub fn new(span: usize) -> Self {
        Self {
            span,
            k: 2.0 / (span as f64 + 1.0),
            value: 0.0,
            initialized: false,
        }
    }

    pub fn update(&mut self, x: f64) -> f64 {
        if !self.initialized {
            self.value = x;
            self.initialized = true;
        } else {
            self.value = x * self.k + self.value * (1.0 - self.k);
        }
        self.value
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}

/// CCI (Commodity Channel Index).
pub fn cci(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n < period { return out; }
    let mut tp = vec![0.0; n];
    for i in 0..n {
        tp[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    for i in (period - 1)..n {
        let mut sum = 0.0;
        for j in 0..period {
            sum += tp[i - j];
        }
        let mean = sum / period as f64;
        let mut mad = 0.0;
        for j in 0..period {
            mad += (tp[i - j] - mean).abs();
        }
        mad /= period as f64;
        out[i] = if mad > 1e-10 {
            (tp[i] - mean) / (0.015 * mad)
        } else {
            0.0
        };
    }
    out
}

/// Williams %R.
pub fn williams_r(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n < period { return out; }
    for i in (period - 1)..n {
        let mut hh = high[i];
        let mut ll = low[i];
        for j in 1..period {
            if high[i - j] > hh { hh = high[i - j]; }
            if low[i - j] < ll { ll = low[i - j]; }
        }
        let range = hh - ll;
        out[i] = if range > 1e-10 {
            -100.0 * (hh - close[i]) / range
        } else {
            0.0
        };
    }
    out
}

/// Stochastic %K.
pub fn stochastic_k(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n < period { return out; }
    for i in (period - 1)..n {
        let mut hh = high[i];
        let mut ll = low[i];
        for j in 1..period {
            if high[i - j] > hh { hh = high[i - j]; }
            if low[i - j] < ll { ll = low[i - j]; }
        }
        let range = hh - ll;
        out[i] = if range > 1e-10 {
            100.0 * (close[i] - ll) / range
        } else {
            50.0
        };
    }
    out
}
