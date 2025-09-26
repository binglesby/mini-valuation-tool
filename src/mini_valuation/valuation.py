from __future__ import annotations
import numpy as np
import pandas as pd
from .settings import get_settings

S = get_settings()


def cagr(series: pd.Series, years_min: int = 3) -> float:
    s = series.dropna()
    if len(s) < years_min:
        return 0.03  # conservative default if too little data
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    n = len(s) - 1
    if first <= 0 or n <= 0:
        return 0.03
    return (last / first) ** (1 / n) - 1


def avg_margin(series_num: pd.Series, series_den: pd.Series, years_min: int = 3) -> float:
    if series_num.empty or series_den.empty:
        return 0.15
    df = pd.DataFrame({"num": series_num, "den": series_den}).dropna()
    if len(df) < years_min or (df["den"] <= 0).any():
        return 0.15
    return float((df["num"] / df["den"]).tail(years_min).mean())


def project_revenue(last_rev: float, growth: float, years: int = 5) -> np.ndarray:
    rev = []
    cur = last_rev
    for _ in range(years):
        cur = cur * (1 + growth)
        rev.append(cur)
    return np.array(rev, dtype=float)


def dcf(
    revenue_series: pd.Series,
    ebit_series: pd.Series,
    d_and_a_series: pd.Series | None,
    shares_out: float,
    net_debt: float,
    wacc: float = S.DEFAULT_WACC,
    tax_rate: float = S.TAX_RATE,
    terminal_g: float = S.DEFAULT_TERMINAL_G,
    capex_pct_sales: float = S.CAPEX_PCT_SALES,
    delta_wc_pct_sales: float = S.DELTA_WC_PCT_SALES,
    years: int = 5,
    override_growth: float | None = None,
    override_margin: float | None = None,
) -> dict:
    last_rev = float(revenue_series.dropna().iloc[-1]) if len(revenue_series.dropna()) else 0.0
    g = override_growth if override_growth is not None else cagr(revenue_series.tail(4))
    ebit_margin = (
        override_margin if override_margin is not None else avg_margin(ebit_series, revenue_series)
    )

    rev = project_revenue(last_rev, g, years)
    ebit = rev * ebit_margin
    d_and_a = (
        d_and_a_series.dropna().iloc[-1]
        if (d_and_a_series is not None and len(d_and_a_series.dropna()))
        else 0.02 * last_rev
    )
    d_and_a_proj = np.full(years, float(d_and_a))
    capex = rev * capex_pct_sales
    delta_wc = rev * delta_wc_pct_sales

    nopat = ebit * (1 - tax_rate)
    fcf = nopat + d_and_a_proj - capex - delta_wc

    disc = np.array([(1 + wacc) ** t for t in range(1, years + 1)], dtype=float)
    pv_fcfs = fcf / disc

    # Terminal value (perpetuity growth on year N FCF)
    tv = fcf[-1] * (1 + terminal_g) / (wacc - terminal_g) if wacc > terminal_g else 0.0
    pv_tv = tv / ((1 + wacc) ** years)

    ev = pv_fcfs.sum() + pv_tv
    equity = ev - net_debt
    per_share = equity / shares_out if shares_out > 0 else np.nan

    return {
        "g": g,
        "ebit_margin": ebit_margin,
        "rev": rev,
        "fcf": fcf,
        "pv_fcfs": pv_fcfs,
        "pv_tv": pv_tv,
        "enterprise_value": ev,
        "equity_value": equity,
        "per_share": per_share,
    }


def multiples_implied_price(net_income: float, pe: float, shares_out: float) -> float | float:
    if shares_out <= 0:
        return np.nan
    equity = net_income * pe
    return equity / shares_out
