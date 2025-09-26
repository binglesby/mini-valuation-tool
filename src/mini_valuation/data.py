from __future__ import annotations
import pandas as pd
import yfinance as yf
from functools import lru_cache


@lru_cache(maxsize=64)
def _ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)


def fetch_financials(ticker: str) -> dict[str, pd.DataFrame | float]:
    tkr = _ticker(ticker)
    income = tkr.financials or pd.DataFrame()
    balance = tkr.balance_sheet or pd.DataFrame()
    cash = tkr.cashflow or pd.DataFrame()
    info = tkr.info or {}
    shares_out = float(info.get("sharesOutstanding") or 0.0)
    price = float(info.get("currentPrice") or 0.0)
    sector = info.get("sector") or ""
    long_name = info.get("longName") or ticker.upper()

    # Normalize columns to datetime index (latest first in yfinance)
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.columns = pd.to_datetime(df.columns)
        return df.T.sort_index()

    return {
        "income": _norm(income),
        "balance": _norm(balance),
        "cash": _norm(cash),
        "shares": shares_out,
        "price": price,
        "sector": sector,
        "name": long_name,
    }


def trailing_metric(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    # Use the latest row
    return float(df.iloc[-1].get(col, 0.0))
