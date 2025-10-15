from __future__ import annotations
import time
from functools import lru_cache
import pandas as pd
import yfinance as yf

try:
    from yfinance.exceptions import YFRateLimitError  # type: ignore
except Exception:  # pragma: no cover - older yfinance

    class YFRateLimitError(Exception):  # fallback type
        pass


@lru_cache(maxsize=64)
def _ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)


def _as_df(obj: object) -> pd.DataFrame:
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame()


def _with_retries(fn, attempts: int = 3, base_delay: float = 0.6):
    last_err: Exception | None = None
    for i in range(max(1, attempts)):
        try:
            return fn()
        except (YFRateLimitError, Exception) as err:  # broad catch to keep UI resilient
            last_err = err
            if i == attempts - 1:
                break
            time.sleep(base_delay * (2**i))
    if last_err:
        raise last_err
    return None


def _from_fast_info(fi: object) -> tuple[float, float]:
    if fi is None:
        return 0.0, 0.0

    def _get(key: str):
        if hasattr(fi, "get"):
            try:
                return fi.get(key)  # type: ignore[attr-defined]
            except Exception:
                return None
        return getattr(fi, key, None)

    # Attempt common keys; tolerate missing fields across yfinance versions
    price = None
    for k in ("last_price", "regular_market_price", "previous_close"):
        price = _get(k)
        if price is not None:
            break

    shares = None
    for k in ("shares", "sharesOutstanding", "shares_outstanding"):
        shares = _get(k)
        if shares is not None:
            break

    # Derive shares from market cap if available
    if (shares is None or float(shares or 0) == 0.0) and price not in (None, 0, 0.0):
        mcap = _get("market_cap")
        try:
            if mcap is not None and float(price) != 0:
                shares = float(mcap) / float(price)
        except Exception:
            shares = shares or 0.0

    try:
        return float(shares or 0.0), float(price or 0.0)
    except Exception:
        return 0.0, 0.0


def fetch_financials(ticker: str) -> dict[str, pd.DataFrame | float]:
    tkr = _ticker(ticker)

    # Financial statements with retry/backoff; return empty on failure
    try:
        income = _as_df(_with_retries(lambda: getattr(tkr, "financials", None)))
    except Exception:
        income = pd.DataFrame()
    try:
        balance = _as_df(_with_retries(lambda: getattr(tkr, "balance_sheet", None)))
    except Exception:
        balance = pd.DataFrame()
    try:
        cash = _as_df(_with_retries(lambda: getattr(tkr, "cashflow", None)))
    except Exception:
        cash = pd.DataFrame()

    # Prefer fast_info to avoid heavy .info call
    shares_out, price = 0.0, 0.0
    try:
        fi = _with_retries(lambda: getattr(tkr, "fast_info", None))
    except Exception:
        fi = None
    s_fast, p_fast = _from_fast_info(fi)
    shares_out = float(s_fast or 0.0)
    price = float(p_fast or 0.0)

    # Best-effort enrichments; swallow rate-limit errors
    sector = ""
    long_name = ticker.upper()
    try:
        info = _with_retries(lambda: getattr(tkr, "info", {}) or {}, attempts=2)
        if not shares_out:
            shares_out = float(info.get("sharesOutstanding") or 0.0)
        if not price:
            price = float(info.get("currentPrice") or 0.0)
        sector = info.get("sector") or sector
        long_name = info.get("longName") or long_name
    except Exception:
        # Fall back to defaults when rate limited
        pass

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
