from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from mini_valuation.settings import get_settings
from mini_valuation.data import fetch_financials
from mini_valuation.valuation import dcf, multiples_implied_price
from mini_valuation.sensitivities import growth_wacc_table
from mini_valuation.viz import line_fcf, heatmap_sensitivity

S = get_settings()
st.set_page_config(page_title="Mini Valuation Tool", layout="wide")
st.title("Mini Valuation Tool")
st.write("Created by Ben Inglesby")
st.caption("Educational use only. Not investment advice.")

with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL").strip()
    run_mode = st.radio("Model", ["DCF", "Multiples"], horizontal=True)
    wacc = st.slider("WACC", 0.04, 0.14, S.DEFAULT_WACC, 0.005)
    term_g = st.slider("Terminal growth", 0.0, 0.04, S.DEFAULT_TERMINAL_G, 0.005)
    st.divider()
    st.write("Sensitivity (DCF)")
    g_min, g_max = st.slider("Revenue growth range", -0.02, 0.20, (0.03, 0.10), 0.005)
    w_min, w_max = st.slider("WACC range", 0.05, 0.15, (0.07, 0.11), 0.005)
    st.divider()
    st.write("Assumptions")
    st.text(f"Tax rate: {S.TAX_RATE:.0%}")
    st.text(f"Capex: {S.CAPEX_PCT_SALES:.0%} of revenue")
    st.text(f"ΔWC: {S.DELTA_WC_PCT_SALES:.0%} of revenue")
    st.text(f"Fallback P/E: {S.FALLBACK_PE:.1f}")

if not ticker:
    st.stop()

data = fetch_financials(ticker)
st.subheader(data["name"])
st.write(f"Current price: **{data['price']:.2f}**")

income: pd.DataFrame = data["income"]
balance: pd.DataFrame = data["balance"]
cash: pd.DataFrame = data["cash"]
shares: float = float(data["shares"])


# Pull core line items with resilience to missing labels
def pick(df: pd.DataFrame, options: list[str]) -> pd.Series:
    for c in options:
        if c in df.columns:
            return df[c].astype(float)
    return pd.Series(dtype=float)


revenue = pick(income, ["Total Revenue", "Operating Revenue"])
ebit = pick(income, ["Ebit", "EBIT"])
net_income = pick(income, ["Net Income", "NetIncome"])
dna = pick(cash, ["Depreciation & Amortization", "DepreciationAndAmortization"])
cash_bal = pick(balance, ["Cash And Cash Equivalents", "Cash"]).replace({np.nan: 0}).tail(1).sum()
debt = (
    pick(balance, ["Short Long Term Debt", "Long Term Debt", "Total Debt"])
    .replace({np.nan: 0})
    .tail(1)
    .sum()
)
net_debt = float(debt - cash_bal)

if run_mode == "DCF":
    result = dcf(
        revenue_series=revenue.squeeze(),
        ebit_series=ebit.squeeze(),
        d_and_a_series=dna.squeeze() if not dna.empty else None,
        shares_out=shares,
        net_debt=net_debt,
        wacc=float(wacc),
        terminal_g=float(term_g),
    )
    implied = float(result["per_share"])
    upside = (implied / data["price"] - 1) * 100 if data["price"] > 0 else np.nan

    col1, col2, col3 = st.columns(3)
    col1.metric("Implied price (DCF)", f"{implied:,.2f}", f"{upside:,.1f}% vs spot")
    col2.metric("Enterprise Value (PV)", f"{result['enterprise_value']:,.0f}")
    col3.metric("Equity Value", f"{result['equity_value']:,.0f}")

    years = list(range(1, len(result["fcf"]) + 1))
    st.plotly_chart(line_fcf(years, result["fcf"]), use_container_width=True)

    st.markdown("#### Sensitivity")
    g_grid = np.linspace(g_min, g_max, 7)
    w_grid = np.linspace(w_min, w_max, 7)
    sens = growth_wacc_table(
        revenue, ebit, dna if not dna.empty else None, shares, net_debt, g_grid, w_grid, term_g
    )
    st.plotly_chart(heatmap_sensitivity(sens), use_container_width=True)
    st.dataframe(sens.style.format("{:.2f}"))

else:
    ni = float(net_income.tail(1).sum())
    implied = multiples_implied_price(ni, pe=S.FALLBACK_PE, shares_out=shares)
    upside = (implied / data["price"] - 1) * 100 if data["price"] > 0 else np.nan
    st.metric("Implied price (P/E)", f"{implied:,.2f}", f"{upside:,.1f}% vs spot")
    st.caption(
        "Sector median P/E not always available via yfinance; using S&P 500 median fallback (20x)."
    )

st.divider()
st.markdown(
    "**Notes**: Uses last reported annuals. Simple heuristics for Capex (5% of revenue) and ΔWC (1% of revenue). Tax 25%. UI lets you adjust WACC and terminal growth."
)

# Inject Newsreader for title with robust selectors
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Newsreader:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root { --newsreader-font: 'Newsreader', serif; }
/* Target Streamlit title variants */
h1,
div[data-testid="stMarkdownContainer"] h1,
.stMarkdown h1,
[data-testid="stAppViewContainer"] h1 {
  font-family: var(--newsreader-font) !important;
  font-weight: 400 !important;
}
</style>
""",
    unsafe_allow_html=True,
)
