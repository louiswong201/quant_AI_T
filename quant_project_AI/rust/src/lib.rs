mod kernels;
mod indicators;
mod io_accel;
mod rolling;
mod fill_sim;

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

use kernels::helpers::CostParams;

// =====================================================================
//  PyO3 bindings — eval_kernel
// =====================================================================

#[pyfunction]
#[pyo3(signature = (name, params, c, o, h, l, sb, ss, cm, lev, dc, dc_short=0.0, sl=0.80, pfrac=1.0, sl_slip=0.0))]
fn eval_kernel(
    name: &str,
    params: Vec<f64>,
    c: PyReadonlyArray1<f64>,
    o: PyReadonlyArray1<f64>,
    h: PyReadonlyArray1<f64>,
    l: PyReadonlyArray1<f64>,
    sb: f64, ss: f64, cm: f64, lev: f64, dc: f64,
    dc_short: f64, sl: f64, pfrac: f64, sl_slip: f64,
) -> (f64, f64, i64) {
    let cp = CostParams { sb, ss, cm, lev, dc, dc_short, sl, pfrac, sl_slip };
    let c_s = c.as_slice().unwrap();
    let o_s = o.as_slice().unwrap();
    let h_s = h.as_slice().unwrap();
    let l_s = l.as_slice().unwrap();
    kernels::dispatch::eval_kernel(name, &params, c_s, o_s, h_s, l_s, &cp)
}

// =====================================================================
//  PyO3 bindings — indicator functions
// =====================================================================

#[pyfunction]
fn rust_ema<'py>(py: Python<'py>, arr: PyReadonlyArray1<f64>, span: usize) -> Bound<'py, PyArray1<f64>> {
    let result = kernels::indicators::ema(arr.as_slice().unwrap(), span);
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn rust_rolling_mean<'py>(py: Python<'py>, arr: PyReadonlyArray1<f64>, w: usize) -> Bound<'py, PyArray1<f64>> {
    let result = kernels::indicators::rolling_mean(arr.as_slice().unwrap(), w);
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn rust_rolling_std<'py>(py: Python<'py>, arr: PyReadonlyArray1<f64>, w: usize) -> Bound<'py, PyArray1<f64>> {
    let result = kernels::indicators::rolling_std(arr.as_slice().unwrap(), w);
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn rust_atr<'py>(py: Python<'py>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, period: usize) -> Bound<'py, PyArray1<f64>> {
    let result = kernels::indicators::atr(
        high.as_slice().unwrap(), low.as_slice().unwrap(), close.as_slice().unwrap(), period,
    );
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn rust_rsi<'py>(py: Python<'py>, close: PyReadonlyArray1<f64>, period: usize) -> Bound<'py, PyArray1<f64>> {
    let result = kernels::indicators::rsi_wilder(close.as_slice().unwrap(), period);
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn rust_cci<'py>(py: Python<'py>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, period: usize) -> Bound<'py, PyArray1<f64>> {
    let result = crate::indicators::technical::cci(
        high.as_slice().unwrap(), low.as_slice().unwrap(), close.as_slice().unwrap(), period,
    );
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn rust_williams_r<'py>(py: Python<'py>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, period: usize) -> Bound<'py, PyArray1<f64>> {
    let result = crate::indicators::technical::williams_r(
        high.as_slice().unwrap(), low.as_slice().unwrap(), close.as_slice().unwrap(), period,
    );
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn rust_stochastic_k<'py>(py: Python<'py>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, period: usize) -> Bound<'py, PyArray1<f64>> {
    let result = crate::indicators::technical::stochastic_k(
        high.as_slice().unwrap(), low.as_slice().unwrap(), close.as_slice().unwrap(), period,
    );
    PyArray1::from_vec_bound(py, result)
}

// =====================================================================
//  PyO3 bindings — fill simulator
// =====================================================================

#[pyfunction]
#[pyo3(signature = (reference_price, order_shares, bar_volume, daily_volatility, participation_rate_cap=0.1, is_buy=true))]
fn rust_market_impact_price(
    reference_price: f64, order_shares: f64, bar_volume: f64,
    daily_volatility: f64, participation_rate_cap: f64, is_buy: bool,
) -> f64 {
    fill_sim::market_impact_price(
        reference_price, order_shares, bar_volume,
        daily_volatility, participation_rate_cap, is_buy,
    )
}

#[pyfunction]
fn rust_compute_fill_price(reference_price: f64, slippage_bps: f64, is_buy: bool) -> f64 {
    fill_sim::compute_fill_price(reference_price, slippage_bps, is_buy)
}

#[pyfunction]
fn rust_limit_fill_probability(limit_price: f64, bar_high: f64, bar_low: f64, is_buy: bool) -> f64 {
    fill_sim::limit_fill_probability(limit_price, bar_high, bar_low, is_buy)
}

#[pyfunction]
fn rust_check_liquidity(order_shares: f64, bar_volume: f64, max_participation_rate: f64) -> f64 {
    fill_sim::check_liquidity(order_shares, bar_volume, max_participation_rate)
}

// =====================================================================
//  PyO3 bindings — rolling buffer
// =====================================================================

#[pyclass]
struct RollingBuffer {
    inner: rolling::RollingOhlcv,
}

#[pymethods]
impl RollingBuffer {
    #[new]
    fn new(capacity: usize) -> Self {
        Self { inner: rolling::RollingOhlcv::new(capacity) }
    }

    fn append(&mut self, o: f64, h: f64, l: f64, c: f64, v: f64) {
        self.inner.append(o, h, l, c, v);
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn latest_close(&self) -> Option<f64> {
        self.inner.latest_close()
    }

    fn to_arrays<'py>(&self, py: Python<'py>) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ) {
        let (o, h, l, c, v) = self.inner.to_arrays();
        (
            PyArray1::from_vec_bound(py, o),
            PyArray1::from_vec_bound(py, h),
            PyArray1::from_vec_bound(py, l),
            PyArray1::from_vec_bound(py, c),
            PyArray1::from_vec_bound(py, v),
        )
    }
}

// =====================================================================
//  PyO3 module definition
// =====================================================================

#[pymodule]
fn quant_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eval_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ema, m)?)?;
    m.add_function(wrap_pyfunction!(rust_rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rust_rolling_std, m)?)?;
    m.add_function(wrap_pyfunction!(rust_atr, m)?)?;
    m.add_function(wrap_pyfunction!(rust_rsi, m)?)?;
    m.add_function(wrap_pyfunction!(rust_cci, m)?)?;
    m.add_function(wrap_pyfunction!(rust_williams_r, m)?)?;
    m.add_function(wrap_pyfunction!(rust_stochastic_k, m)?)?;
    m.add_function(wrap_pyfunction!(rust_market_impact_price, m)?)?;
    m.add_function(wrap_pyfunction!(rust_compute_fill_price, m)?)?;
    m.add_function(wrap_pyfunction!(rust_limit_fill_probability, m)?)?;
    m.add_function(wrap_pyfunction!(rust_check_liquidity, m)?)?;
    m.add_class::<RollingBuffer>()?;
    Ok(())
}
