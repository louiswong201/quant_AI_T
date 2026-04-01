/// Fill simulation with market impact model.

/// Almgren-Chriss style market impact.
/// Returns the impact-adjusted execution price.
pub fn market_impact_price(
    reference_price: f64,
    order_shares: f64,
    bar_volume: f64,
    daily_volatility: f64,
    participation_rate_cap: f64,
    is_buy: bool,
) -> f64 {
    if bar_volume <= 0.0 || reference_price <= 0.0 {
        return reference_price;
    }
    let participation = (order_shares.abs() / bar_volume).min(participation_rate_cap);
    let temporary_impact = daily_volatility * participation.sqrt();
    let permanent_impact = 0.1 * daily_volatility * participation;
    let total_impact = temporary_impact + permanent_impact;
    if is_buy {
        reference_price * (1.0 + total_impact)
    } else {
        reference_price * (1.0 - total_impact)
    }
}

/// Compute execution price with slippage and commission.
pub fn compute_fill_price(
    reference_price: f64,
    slippage_bps: f64,
    is_buy: bool,
) -> f64 {
    let slip = slippage_bps / 10000.0;
    if is_buy {
        reference_price * (1.0 + slip)
    } else {
        reference_price * (1.0 - slip)
    }
}

/// Probabilistic limit fill: returns fill probability based on
/// how deep the bar's range penetrates the limit price.
pub fn limit_fill_probability(
    limit_price: f64,
    bar_high: f64,
    bar_low: f64,
    is_buy: bool,
) -> f64 {
    let range = bar_high - bar_low;
    if range < 1e-10 {
        return if is_buy && bar_low <= limit_price { 1.0 }
               else if !is_buy && bar_high >= limit_price { 1.0 }
               else { 0.0 };
    }
    if is_buy {
        if bar_low > limit_price { return 0.0; }
        if bar_high <= limit_price { return 1.0; }
        (limit_price - bar_low) / range
    } else {
        if bar_high < limit_price { return 0.0; }
        if bar_low >= limit_price { return 1.0; }
        (bar_high - limit_price) / range
    }
}

/// Check if an order can be filled within liquidity constraints.
pub fn check_liquidity(
    order_shares: f64,
    bar_volume: f64,
    max_participation_rate: f64,
) -> f64 {
    let max_fillable = bar_volume * max_participation_rate;
    order_shares.abs().min(max_fillable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_price_buy() {
        let p = compute_fill_price(100.0, 10.0, true);
        assert!((p - 100.1).abs() < 1e-10);
    }

    #[test]
    fn test_fill_price_sell() {
        let p = compute_fill_price(100.0, 10.0, false);
        assert!((p - 99.9).abs() < 1e-10);
    }

    #[test]
    fn test_market_impact() {
        let p = market_impact_price(100.0, 1000.0, 100000.0, 0.02, 0.1, true);
        assert!(p > 100.0);
    }

    #[test]
    fn test_limit_fill_prob() {
        assert_eq!(limit_fill_probability(50.0, 55.0, 45.0, true), 0.5);
        assert_eq!(limit_fill_probability(50.0, 55.0, 45.0, false), 0.5);
    }
}
