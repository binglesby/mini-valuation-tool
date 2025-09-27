from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from mini_valuation.settings import get_settings
from mini_valuation.data import fetch_financials
from mini_valuation.valuation import dcf, multiples_implied_price
from mini_valuation.sensitivities import growth_wacc_table
from mini_valuation.viz import line_fcf, heatmap_sensitivity

# Helpers for currency formatting with compact units and hover full value


def _fmt_money(value: float) -> tuple[str, str]:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "$—", "$—"
    absn = abs(n)
    if absn >= 1_000_000_000_000:
        disp = f"${n/1_000_000_000_000:.3f}T"
    elif absn >= 1_000_000_000:
        disp = f"${n/1_000_000_000:.3f}B"
    elif absn >= 1_000_000:
        disp = f"${n/1_000_000:.3f}M"
    elif absn >= 1_000:
        disp = f"${n/1_000:.3f}K"
    else:
        disp = f"${n:,.2f}"
    full_ = f"${n:,.0f}"
    return disp, full_


# Render a compact currency metric card into a given container
# Shows compact $ value and full value on hover via title attribute
def _metric_card(container, label: str, value: float, delta_pct: float | None = None) -> None:
    disp, full_ = _fmt_money(float(value))
    delta_html = (
        f'<div class="delta">{delta_pct:.1f}% vs spot</div>' if delta_pct is not None else ""
    )
    html = f"""
<div class=\"metric-card\">
  <div class=\"label\">{label}</div>
  <div class=\"value\" title=\"{full_}\">{disp}</div>
  {delta_html}
</div>
"""
    container.markdown(html, unsafe_allow_html=True)


S = get_settings()
st.set_page_config(
    page_title="Mini Valuation Tool", layout="wide", initial_sidebar_state="expanded"
)
st.title("Valuation Tool")
st.write(
    "Created by Ben Inglesby, for educational use only. Input a company's ticker and adjust assumptions in the left sidebar. Not investment advice. "
)

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
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("**Company & Valuation Settings**")
    ticker = st.text_input("Ticker", value="AAPL").strip()
    wacc_pct = st.slider(
        "WACC",
        4.0,
        14.0,
        float(S.DEFAULT_WACC * 100),
        0.5,
        format="%.1f%%",
    )
    wacc = wacc_pct / 100.0
    term_pct = st.slider(
        "Terminal growth",
        0.0,
        4.0,
        float(S.DEFAULT_TERMINAL_G * 100),
        0.5,
        format="%.1f%%",
    )
    term_g = term_pct / 100.0
    fallback_pe = st.slider("Fallback P/E", 5.0, 40.0, float(S.FALLBACK_PE), 1.0)

    st.divider()

    st.markdown("**Sensitivity Ranges**")
    rev_range_pct = st.slider(
        "Revenue growth range",
        -2.0,
        20.0,
        (3.0, 10.0),
        0.5,
        format="%.1f%%",
    )
    g_min, g_max = (rev_range_pct[0] / 100.0, rev_range_pct[1] / 100.0)

    wacc_range_pct = st.slider(
        "WACC range",
        5.0,
        15.0,
        (7.0, 11.0),
        0.5,
        format="%.1f%%",
    )
    w_min, w_max = (wacc_range_pct[0] / 100.0, wacc_range_pct[1] / 100.0)

    st.divider()

    st.markdown("**Operating Assumptions**")
    tax_pct = st.slider("Tax rate", 0.0, 40.0, float(S.TAX_RATE * 100), 0.5, format="%.1f%%")
    tax_rate = tax_pct / 100.0
    capex_pct = st.slider(
        "Capex (% of revenue)", 0.0, 15.0, float(S.CAPEX_PCT_SALES * 100), 0.5, format="%.1f%%"
    )
    capex_pct_sales = capex_pct / 100.0
    delta_wc_pct = st.slider(
        "ΔWC (% of revenue)", -5.0, 10.0, float(S.DELTA_WC_PCT_SALES * 100), 0.5, format="%.1f%%"
    )
    delta_wc_pct_sales = delta_wc_pct / 100.0

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
    _metric_card(m_price, "Current price", float(data["price"]))
    _metric_card(m1, "Implied price (DCF)", implied, upside)
    _metric_card(m2, "Enterprise Value (PV)", float(result["enterprise_value"]))
    _metric_card(m3, "Equity Value", float(result["equity_value"]))

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
    _metric_card(m_price, "Current price", float(data["price"]))
    _metric_card(m_pe, "Implied price (P/E)", implied, upside)
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
/* Reduce extra bottom space in sidebar */
[data-testid="stSidebar"] .block-container { padding-bottom: 0.25rem !important; }
/* Rename sidebar toggle tooltip/label by replacing icon glyph text when present */
[data-testid="stSidebarCollapsedControl"] span[class*="material"],
header [title*="sidebar"] span[class*="material"] { font-family: inherit !important; }
header [title*="sidebar"] span[class*="material"]::before { content: "Controls"; }
</style>
""",
    unsafe_allow_html=True,
)

# Styles for custom metric cards
st.markdown(
    """
<style>
.metric-card { padding: 0.25rem 0; }
.metric-card .label { font-size: 0.95rem; color: rgba(0,0,0,0.6); margin-bottom: 0.15rem; }
.metric-card .value { font-size: 2.1rem; font-weight: 600; line-height: 1.1; }
.metric-card .delta { margin-top: 0.2rem; font-size: 0.9rem; display: inline-block; padding: 0.1rem 0.35rem; border-radius: 999px; background: rgba(0,0,0,0.06); }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght@400" rel="stylesheet">
<style>
/* Prefer icon glyph if available */
header [data-testid="baseButton-secondary"] span[class*="material"],
[data-testid="stSidebarCollapsedControl"] span[class*="material"] {
  font-family: 'Material Symbols Outlined', 'Material Icons' !important;
  font-feature-settings: 'liga';
}
/* Fallback: replace raw icon text with a readable label */
header [data-testid="baseButton-secondary"] span[class*="material"],
[data-testid="stSidebarCollapsedControl"] span[class*="material"] {
  position: relative;
}
header [data-testid="baseButton-secondary"] span[class*="material"]:not([style*="font-family"]):after,
[data-testid="stSidebarCollapsedControl"] span[class*="material"]:not([style*="font-family"]):after {
  content: 'Controls';
  font-family: inherit;
  font-size: 0.9rem;
}
header [data-testid="baseButton-secondary"] span[class*="material"]:not([style*="font-family"]),
[data-testid="stSidebarCollapsedControl"] span[class*="material"]:not([style*="font-family"]) {
  font-size: 0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<style>
/* Reset sidebar toggle button font to default Streamlit font (open and closed states) */
header [data-testid="baseButton-secondary"] span[class*="material"],
[data-testid="stSidebarCollapsedControl"] span[class*="material"] {
  font-family: inherit !important;
  font-feature-settings: normal !important;
  font-size: inherit !important;
}
/* Remove any custom replacement label/content from prior overrides */
header [data-testid="baseButton-secondary"] span[class*="material"]::before,
header [data-testid="baseButton-secondary"] span[class*="material"]::after,
[data-testid="stSidebarCollapsedControl"] span[class*="material"]::before,
[data-testid="stSidebarCollapsedControl"] span[class*="material"]::after { content: none !important; }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<style>
/* Force the sidebar toggle glyph (open + closed states) to render as an icon */
header [data-testid="stExpanderSidebarButton"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"] {
  font-family: 'Material Icons' !important;
  font-weight: normal !important;
  font-style: normal !important;
  font-size: 20px !important;
  line-height: 1 !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  display: inline-block !important;
  white-space: nowrap !important;
  direction: ltr !important;
  font-feature-settings: 'liga' !important; /* enable ligatures */
  -webkit-font-smoothing: antialiased;
}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<style>
/* Pragmatic fallback: hide raw ligature text and show a simple glyph */
header [data-testid="stExpanderSidebarButton"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"] {
  font-size: 0 !important;            /* hide raw text like 'keyboard_double_arrow_right' */
  font-family: inherit !important;     /* avoid icon font conflicts */
}
header [data-testid="stExpanderSidebarButton"] [data-testid="stIconMaterial"]::before,
[data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"]::before {
  content: '»';                        /* simple, readable double-angle icon */
  font-size: 18px;                     /* visible size */
  line-height: 1;                      /* align nicely */
}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<script>
(function () {
  const apply = (root) => {
    const nodes = (root || document).querySelectorAll('[data-testid="stIconMaterial"]');
    nodes.forEach((n) => {
      const t = (n.textContent || '').trim();
      if (t !== '»') {
        n.textContent = '»';            // simple double-angle icon
        n.style.fontSize = '18px';
        n.style.lineHeight = '1';
        n.style.fontFamily = 'inherit';
      }
    });
  };
  const mo = new MutationObserver((muts) => {
    for (const m of muts) apply(m.target);
  });
  apply();
  mo.observe(document.body, { subtree: true, childList: true, characterData: true });
})();
</script>
""",
    unsafe_allow_html=True,
)
