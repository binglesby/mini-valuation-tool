# Mini Valuation Tool

An interactive web app that estimates what a company’s stock could be worth using two common approaches: a Discounted Cash Flow (DCF) model and a simple Multiples model. Designed to be easy to understand, quick to use, and visually clear.

This project is for educational use only and not investment advice.

## What it does (in plain English)

- Enter a stock ticker (e.g., AAPL)
- Adjust a few intuitive assumptions (growth, discount rate, taxes, capex)
- See an estimated price per share from two perspectives (DCF and Multiples)
- Explore visuals that show the drivers behind the valuation

## Key features

- DCF and Multiples side-by-side (toggle between models)
- Live company data pulled from Yahoo Finance via `yfinance`
- Clear, compact UI with sticky controls and consistent formatting
- Adjustable assumptions: WACC, terminal growth, tax rate, capex % revenue, Δ working capital % revenue, fallback P/E
- Visuals built with Plotly:
  - Projected free cash flow (line)
  - Enterprise Value composition (PV of FCF vs. Terminal Value)
  - EV → Equity bridge (cash and debt adjustments)
  - Growth vs. WACC sensitivity heatmap
  - Mini “tornado” showing ±50 bps sensitivity
- Helpful formatting for readability: $K/$M/$B/$T units, percent axes, minimal chart clutter

## How it works (at a high level)

- Pulls recent financials (income statement, balance sheet, cash flow) for the selected ticker
- Computes historical growth/margins and projects free cash flow
- Discounts future cash flows and a terminal value to estimate Enterprise Value (EV)
- Moves from EV to Equity Value (cash/debt), then to an implied price per share
- Multiples model provides a quick cross-check using a user-set fallback P/E

## Why this is useful

This tool makes a complex topic approachable. It helps a non-technical audience see how a few key assumptions influence a valuation, with clean visuals and simple language.

## Try it locally (2–3 minutes)

```bash
# 1) Create a virtual environment (Python 3.11)
python -m venv .venv
source .venv/bin/activate

# 2) Install dependencies (fast, via uv)
pip install -U pip uv
uv pip install -e ".[dev]"

# 3) (Optional) enable formatters/linters
pre-commit install

# 4) Run the app
./scripts/run_app.sh
```

Then open the printed local URL in your browser (it starts with `http://localhost:`). Type a ticker (e.g., AAPL) and explore the charts.

## Screens you’ll see

- Company header with a model selector (DCF or Multiples)
- A compact controls panel to set assumptions
- A summary row with key metrics (implied price, EV, equity value, upside)
- Charts: projected FCF, EV composition, EV → Equity waterfall, sensitivity heatmap, and a small tornado chart

## Tech (kept simple)

- Streamlit for the web app UI
- Plotly for interactive charts
- yfinance + pandas for data
- Pydantic for safe settings

## Notes & disclaimers

- For learning only; simplified assumptions; may not reflect real-world complexities
- Data quality and availability depend on the public Yahoo Finance API
- Results change as assumptions change; no guarantees or recommendations

## Developer quickstart (same as above, condensed)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip uv
uv pip install -e ".[dev]"
pre-commit install
./scripts/run_app.sh
```

If you prefer plain pip over uv, replace the install step with:

```bash
pip install -e ".[dev]"
```
