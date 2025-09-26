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
st.set_page_config(
    page_title="Mini Valuation Tool", layout="wide", initial_sidebar_state="expanded"
)
st.title("Valuation Tool")
st.write("Created by Ben Inglesby. Educational use only. Not investment advice.")

# Model selector just under the caption
try:
    run_mode = st.segmented_control("Select Model", options=["DCF", "Multiples"], default="DCF")
except AttributeError:
    run_mode = st.radio("Select Model", ["DCF", "Multiples"], horizontal=True)
# Style segmented control and radio fallback
st.markdown(
    """
<style>
/* Title: bold and slightly larger */
h1, div[data-testid="stMarkdownContainer"] h1, .stMarkdown h1, [data-testid="stAppViewContainer"] h1 {
  font-weight: 700 !important;
  font-size: 2rem !important;
}
/* Segmented control selected pill in blue */
[data-testid="stSegmentedControl"] label[data-checked="true"] {
  background: #0A84FF !important; color: #fff !important;
}
[data-testid="stSegmentedControl"] label { border-radius: 999px !important; }
/* Radio fallback: style as pills */
[role="radiogroup"] label {
  background: rgba(0,0,0,0.04);
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  margin-right: 0.35rem;
}
[role="radiogroup"] input[type="radio"] { visibility: hidden; width: 0; height: 0; }
[role="radiogroup"] label[data-checked="true"] { background: #0A84FF; color: #fff; }
/* Always keep sidebar open: hide collapse control */
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL").strip()
    wacc = st.slider("WACC", 0.04, 0.14, S.DEFAULT_WACC, 0.005)
    term_g = st.slider("Terminal growth", 0.0, 0.04, S.DEFAULT_TERMINAL_G, 0.005)
    st.divider()
    st.write("Sensitivity (DCF)")
    g_min, g_max = st.slider("Revenue growth range", -0.02, 0.20, (0.03, 0.10), 0.005)
    w_min, w_max = st.slider("WACC range", 0.05, 0.15, (0.07, 0.11), 0.005)
    st.divider()
    st.markdown('<div class="assumptions-header">Assumptions</div>', unsafe_allow_html=True)
    tax_rate = st.slider("Tax rate", 0.0, 0.40, float(S.TAX_RATE), 0.005)
    capex_pct_sales = st.slider("Capex (% of revenue)", 0.0, 0.15, float(S.CAPEX_PCT_SALES), 0.005)
    delta_wc_pct_sales = st.slider(
        "ΔWC (% of revenue)", -0.05, 0.10, float(S.DELTA_WC_PCT_SALES), 0.005
    )
    fallback_pe = st.slider("Fallback P/E", 5.0, 40.0, float(S.FALLBACK_PE), 1.0)

if not ticker:
    st.stop()

data = fetch_financials(ticker)

# Company title always visible
st.subheader(data["name"])

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
        tax_rate=float(tax_rate),
        capex_pct_sales=float(capex_pct_sales),
        delta_wc_pct_sales=float(delta_wc_pct_sales),
    )
    implied = float(result["per_share"])
    upside = (implied / data["price"] - 1) * 100 if data["price"] > 0 else np.nan

    # Inline metrics including Current price
    m_price, m1, m2, m3 = st.columns(4)
    m_price.metric("Current price", f"{data['price']:,.2f}")
    m1.metric("Implied price (DCF)", f"{implied:,.2f}", f"{upside:,.1f}% vs spot")
    m2.metric("Enterprise Value (PV)", f"{result['enterprise_value']:,.0f}")
    m3.metric("Equity Value", f"{result['equity_value']:,.0f}")

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
    implied = multiples_implied_price(ni, pe=float(fallback_pe), shares_out=shares)
    upside = (implied / data["price"] - 1) * 100 if data["price"] > 0 else np.nan
    # Inline metrics: Current price and P/E implied
    m_price, m_pe = st.columns(2)
    m_price.metric("Current price", f"{data['price']:,.2f}")
    m_pe.metric("Implied price (P/E)", f"{implied:,.2f}", f"{upside:,.1f}% vs spot")
    st.caption(
        "Sector median P/E not always available via yfinance; using S&P 500 median fallback (20x)."
    )

st.divider()
st.markdown(
    "**Notes**: Uses last reported annuals. Simple heuristics for Capex (5% of revenue) and ΔWC (1% of revenue). Tax 25%. UI lets you adjust WACC and terminal growth."
)

# Global Apple system fonts, fix sidebar toggle icon, and consistent sidebar spacing
st.markdown(
    """
<style>
:root { --apple-font: -apple-system, BlinkMacSystemFont, "SF Pro Text","SF Pro Display","Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", Arial, sans-serif; }
/* Apply system font but preserve Material icon glyphs */
html, body, [data-testid="stAppViewContainer"] :not(.material-icons):not(.material-icons-outlined):not(.material-icons-round):not(.material-icons-sharp):not(.material-symbols-outlined):not(.material-symbols-rounded):not(.material-symbols-sharp) {
  font-family: var(--apple-font) !important;
}
/* Ensure icon fonts render correctly */
.material-icons, .material-icons-outlined, .material-icons-round, .material-icons-sharp,
.material-symbols-outlined, .material-symbols-rounded, .material-symbols-sharp {
  font-family: 'Material Symbols Outlined','Material Icons','Material Icons Outlined', sans-serif !important;
  font-feature-settings: 'liga';
  font-weight: normal;
}
/* Sidebar spacing: moderate and consistent */
[data-testid="stSidebar"] .block-container { padding-top: 0.75rem; padding-bottom: 0.75rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.6rem !important; row-gap: 0.6rem !important; }
/* Assumptions header: bold and slightly larger */
.assumptions-header { font-weight: 600; font-size: 1.05rem; margin-top: 0.25rem; }
</style>
""",
    unsafe_allow_html=True,
)
