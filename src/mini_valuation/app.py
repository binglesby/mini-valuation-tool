from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

# Ensure src/ is on sys.path when running via script path in Streamlit Cloud
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mini_valuation.settings import get_settings  # noqa: E402
from mini_valuation.data import fetch_financials  # noqa: E402
from mini_valuation.valuation import dcf, multiples_implied_price  # noqa: E402
from mini_valuation.sensitivities import growth_wacc_table  # noqa: E402
from mini_valuation.viz import line_fcf  # noqa: E402

# Helpers for currency formatting with compact units and hover full value


def _fmt_money(value: float) -> tuple[str, str]:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "$—", "$—"
    absn = abs(n)
    if absn >= 1_000_000_000_000:
        disp = f"${n/1_000_000_000_000:.1f}T"
    elif absn >= 1_000_000_000:
        disp = f"${n/1_000_000_000:.1f}B"
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


def _metric_percent(container, label: str, pct_value: float) -> None:
    pct_str = f"{pct_value:.1f}%" if np.isfinite(pct_value) else "—"
    html = f"""
<div class=\"metric-card\">
  <div class=\"label\">{label}</div>
  <div class=\"value\">{pct_str}</div>
</div>
"""
    container.markdown(html, unsafe_allow_html=True)


# Visual styling helpers/constants
BLUE = "rgba(37, 99, 235, 1)"
LTBLU = "rgba(96, 165, 250, 1)"
GREEN = "rgba(34, 197, 94, 1)"
RED = "rgba(239, 68, 68, 1)"
GRID = "rgba(0,0,0,0.06)"

CONFIG_MINIMAL = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "autoScale2d",
        "zoom2d",
        "pan2d",
        "zoomIn2d",
        "zoomOut2d",
        "resetScale2d",
        "toImage",
    ],
}


def _layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID)
    return fig


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

left_panel, main_area = st.columns([0.28, 0.72], gap="large")

# Make the left panel behave like a sticky sidebar
st.markdown(
    """
<style>
[data-testid="column"] > div:has(> div > div > div > p:contains('Company & Valuation Settings')) { position: sticky; top: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

with left_panel:
    st.markdown("**Company & Valuation Settings**")
    ticker = st.text_input("Ticker", value="AAPL").strip()
    wacc_pct = st.slider(
        "WACC", 4.0, 14.0, float(S.DEFAULT_WACC * 100), 0.5, format="%.1f%%", key="wacc_pct"
    )
    wacc = wacc_pct / 100.0
    term_pct = st.slider(
        "Terminal growth",
        0.0,
        4.0,
        float(S.DEFAULT_TERMINAL_G * 100),
        0.5,
        format="%.1f%%",
        key="term_pct",
    )
    term_g = term_pct / 100.0
    fallback_pe = st.slider("Fallback P/E", 5.0, 40.0, float(S.FALLBACK_PE), 1.0)

    st.divider()
    st.markdown("**Sensitivity Ranges**")
    rev_range_pct = st.slider("Revenue growth range", -2.0, 20.0, (3.0, 10.0), 0.5, format="%.1f%%")
    g_min, g_max = (rev_range_pct[0] / 100.0, rev_range_pct[1] / 100.0)

    wacc_range_pct = st.slider("WACC range", 5.0, 15.0, (7.0, 11.0), 0.5, format="%.1f%%")
    w_min, w_max = (wacc_range_pct[0] / 100.0, wacc_range_pct[1] / 100.0)

    st.divider()
    st.markdown("**Operating Assumptions**")
    tax_pct = st.slider(
        "Tax rate", 0.0, 40.0, float(S.TAX_RATE * 100), 0.5, format="%.1f%%", key="tax_pct"
    )
    tax_rate = tax_pct / 100.0
    capex_pct = st.slider(
        "Capex (% of revenue)",
        0.0,
        15.0,
        float(S.CAPEX_PCT_SALES * 100),
        0.5,
        format="%.1f%%",
        key="capex_pct",
    )
    capex_pct_sales = capex_pct / 100.0
    delta_wc_pct = st.slider(
        "ΔWC (% of revenue)",
        -5.0,
        10.0,
        float(S.DELTA_WC_PCT_SALES * 100),
        0.5,
        format="%.1f%%",
        key="delta_wc_pct",
    )
    delta_wc_pct_sales = delta_wc_pct / 100.0

if not ticker:
    st.stop()

data = fetch_financials(ticker)

# Company title always visible
with main_area:
    st.subheader(data["name"])
    # Microcaption of inputs
    st.caption(
        f"Inputs: WACC {wacc*100:.1f}% • g {term_g*100:.1f}% • Tax {tax_rate*100:.0f}% • Capex {capex_pct_sales*100:.0f}% rev • ΔWC {delta_wc_pct_sales*100:.0f}% rev"
    )

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

    # Guardrails
    if float(term_g) >= float(wacc):
        main_area.markdown(
            "<span style='color:#b00020; padding:4px 8px; border:1px solid #b00020; border-radius:6px;'>Unstable terminal math (g ≥ WACC)</span>",
            unsafe_allow_html=True,
        )

    # Summary row (KPIs)
    k1, k2, k3, k4, k5 = main_area.columns(5)
    _metric_card(k1, "Implied price (DCF)", implied, None)
    # Only keep upside pill on KPI row
    _metric_percent(k2, "Upside/Downside vs spot", upside)
    _metric_card(k3, "Enterprise Value (PV)", float(result["enterprise_value"]))
    _metric_card(k4, "Equity Value", float(result["equity_value"]))
    tv_share = (
        float(result["pv_tv"]) / float(result["enterprise_value"])
        if result["enterprise_value"]
        else np.nan
    )
    _metric_percent(
        k5,
        "Terminal value as % of EV",
        tv_share * 100.0 if np.isfinite(tv_share) else np.nan,
    )
    if np.isfinite(tv_share) and (tv_share > 0.85 or tv_share < 0.50):
        hint = "High TV dependence" if tv_share > 0.85 else "Low TV dependence"
        main_area.caption(f"{hint} (typical 60–80%)")

    # Core charts
    c_left, c_mid, c_right = main_area.columns([0.34, 0.33, 0.33])

    # 1) Projected Free Cash Flow (existing)
    years = list(range(1, len(result["fcf"]) + 1))
    # Customize FCF chart: blue line, currency ticks, subtitle
    fig_fcf = line_fcf(years, result["fcf"])
    fig_fcf.update_traces(line=dict(color="#1f77b4"))
    fig_fcf.update_layout(
        title="Projected Free Cash Flow<br><sup>Years 1–5, nominal USD</sup>",
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig_fcf.update_yaxes(tickformat="$~s", gridcolor="#e0e0e0")
    c_left.plotly_chart(fig_fcf, use_container_width=True, config=CONFIG_MINIMAL)

    # 2) EV composition (stacked bar): PV of explicit FCFs vs PV of terminal value
    pv_fcfs_sum = (
        float(np.nansum(result["pv_fcfs"]))
        if isinstance(result["pv_fcfs"], np.ndarray)
        else float(result["pv_fcfs"])
    )  # safety
    pv_tv = float(result["pv_tv"])
    fig_stack = go.Figure()
    fig_stack.add_bar(x=["Mix"], y=[pv_fcfs_sum], name="PV of FCFs", marker_color="#1f77b4")
    fig_stack.add_bar(x=["Mix"], y=[pv_tv], name="PV of Terminal", marker_color="#7fb3d5")
    fig_stack.update_layout(
        barmode="stack",
        barnorm="percent",
        title="EV composition<br><sup>PV of terminal vs PV of explicit</sup>",
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig_stack.update_yaxes(ticksuffix="%", gridcolor="#e0e0e0")
    fig_stack.update_traces(texttemplate="%{percentY:.0%}", textposition="inside")
    c_mid.plotly_chart(fig_stack, use_container_width=True, config=CONFIG_MINIMAL)

    # 3) EV → Equity waterfall
    base_ev = float(result["enterprise_value"])
    fig_wf = go.Figure(
        go.Waterfall(
            name="EV→Equity",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["EV", "+Cash", "-Debt", "Equity"],
            text=[
                f"EV ${base_ev/1e9:,.1f}B",
                f"+Cash ${cash_bal/1e9:,.1f}B",
                f"-Debt ${debt/1e9:,.1f}B",
                f"Equity ${float(result['equity_value'])/1e9:,.1f}B",
            ],
            textposition="outside",
            y=[base_ev, float(cash_bal), -float(debt), 0.0],
            connector=dict(line=dict(color="#9e9e9e")),
            increasing=dict(marker=dict(color="#2e7d32")),  # cash (green)
            decreasing=dict(marker=dict(color="#c62828")),  # debt (red)
            totals=dict(marker=dict(color="#1f77b4")),  # EV/Equity (blue)
        )
    )
    fig_wf.update_layout(
        title="EV → Equity bridge<br><sup>Most recent balance sheet</sup>",
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig_wf.update_yaxes(gridcolor="#e0e0e0", tickformat="$~s")
    # Add implied price annotation
    fig_wf.add_annotation(
        x=3,
        y=float(result["equity_value"]),
        text=f"Shares: {shares:,.0f} • Price/Share: ${implied:,.2f}",
        showarrow=False,
        yshift=20,
    )
    c_right.plotly_chart(fig_wf, use_container_width=True, config=CONFIG_MINIMAL)

    # Sensitivity block
    main_area.markdown("#### Sensitivity")
    g_grid = np.linspace(g_min, g_max, 7)
    w_grid = np.linspace(w_min, w_max, 7)
    sens = growth_wacc_table(
        revenue, ebit, dna if not dna.empty else None, shares, net_debt, g_grid, w_grid, term_g
    )
    # Build heatmap inline for formatting
    x_labels = [f"{x*100:.1f}%" for x in sens.columns.astype(float)]
    y_labels = [f"{float(y)*100:.1f}%" for y in sens.index.astype(float)]
    fig_hm = go.Figure(
        data=go.Heatmap(
            z=sens.values,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu",
            colorbar=dict(title="Price ($)"),
            zmin=np.nanmin(sens.values),
            zmax=np.nanmax(sens.values),
        )
    )
    # Crosshair rectangle around current sliders
    try:
        xi = np.argmin(np.abs(sens.columns.astype(float) - float(wacc)))
        yi = np.argmin(np.abs(sens.index.astype(float) - float(g_grid[3])))
        fig_hm.add_shape(
            type="rect",
            x0=xi - 0.5,
            x1=xi + 0.5,
            y0=yi - 0.5,
            y1=yi + 0.5,
            xref="x",
            yref="y",
            line=dict(color="#000", width=1.5),
        )
    except Exception:
        pass
    fig_hm.update_layout(
        title="Sensitivity heatmap",
        xaxis_title="WACC (cols)",
        yaxis_title="Growth (rows)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    # Full-width heatmap directly under the top row; use_container_width keeps it wide
    main_area.plotly_chart(fig_hm, use_container_width=True, config=CONFIG_MINIMAL)
    main_area.dataframe(sens.style.format("${:,.2f}"))

    # Mini tornado (±50 bps) for WACC and Terminal growth
    main_area.markdown("#### Mini tornado (±50 bps)")

    def _price_at(w: float, g: float) -> float:
        # Clamp to valid space: wacc must exceed terminal_g slightly
        eps = 1e-6
        w = max(float(w), float(g) + eps)
        res = dcf(
            revenue_series=revenue.squeeze(),
            ebit_series=ebit.squeeze(),
            d_and_a_series=dna.squeeze() if not dna.empty else None,
            shares_out=shares,
            net_debt=net_debt,
            wacc=w,
            terminal_g=g,
            tax_rate=float(tax_rate),
            capex_pct_sales=float(capex_pct_sales),
            delta_wc_pct_sales=float(delta_wc_pct_sales),
        )
        return float(res["per_share"])

    bps = 0.005
    # WACC shock
    w_low = max(0.0, float(wacc) - bps)
    w_high = float(wacc) + bps
    p_w_low = _price_at(w_low, float(term_g))
    p_w_high = _price_at(w_high, float(term_g))
    w_min, w_max = (min(p_w_low, p_w_high), max(p_w_low, p_w_high))

    # Terminal growth shock (cap at wacc - tiny eps)
    g_low = max(0.0, float(term_g) - bps)
    g_high = min(float(wacc) - 1e-6, float(term_g) + bps)
    p_g_low = _price_at(float(wacc), g_low)
    p_g_high = _price_at(float(wacc), g_high)
    g_min, g_max = (min(p_g_low, p_g_high), max(p_g_low, p_g_high))

    fig_tornado = go.Figure()
    # vertical base line at current implied price
    fig_tornado.add_shape(
        type="line", x0=implied, x1=implied, y0=-0.5, y1=1.5, line=dict(color=GRID, width=1)
    )
    fig_tornado.add_bar(
        y=["WACC ±50 bps"], x=[w_max - w_min], base=[w_min], orientation="h", name="WACC"
    )
    fig_tornado.add_bar(
        y=["Terminal g ±50 bps"],
        x=[g_max - g_min],
        base=[g_min],
        orientation="h",
        name="Terminal g",
    )
    fig_tornado.update_layout(
        barmode="overlay",
        title="Implied Price Sensitivity",
        xaxis_title="Price ($)",
        yaxis_title="",
        showlegend=False,
    )
    # Annotations for endpoints
    fig_tornado.add_annotation(
        x=w_min, y=0, text=f"${w_min:,.2f}", showarrow=False, xanchor="right", yshift=10
    )
    fig_tornado.add_annotation(
        x=w_max, y=0, text=f"${w_max:,.2f}", showarrow=False, xanchor="left", yshift=10
    )
    fig_tornado.add_annotation(
        x=g_min, y=1, text=f"${g_min:,.2f}", showarrow=False, xanchor="right", yshift=10
    )
    fig_tornado.add_annotation(
        x=g_max, y=1, text=f"${g_max:,.2f}", showarrow=False, xanchor="left", yshift=10
    )
    fig_tornado.update_layout(height=220)
    main_area.plotly_chart(fig_tornado, use_container_width=True, config=CONFIG_MINIMAL)

else:
    ni = float(net_income.tail(1).sum())
    implied = multiples_implied_price(ni, pe=float(fallback_pe), shares_out=shares)
    upside = (implied / data["price"] - 1) * 100 if data["price"] > 0 else np.nan
    # Inline metrics: Current price and P/E implied
    m_price, m_pe = main_area.columns(2)
    _metric_card(m_price, "Current price", float(data["price"]))
    _metric_card(m_pe, "Implied price (P/E)", implied, upside)
    main_area.caption(
        "Sector median P/E not always available via yfinance; using S&P 500 median fallback (20x)."
    )

main_area.divider()
main_area.markdown(
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

# Hide native sidebar and make left column sticky (we use the embedded panel)
st.markdown(
    """
<style>
header [data-testid="stExpanderSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }
section.main > div > div > div:first-child { position: sticky; top: 1rem; align-self: flex-start; }
</style>
""",
    unsafe_allow_html=True,
)
