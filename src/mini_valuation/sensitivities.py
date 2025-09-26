from __future__ import annotations
import numpy as np
import pandas as pd
from .valuation import dcf


def growth_wacc_table(
    revenue_series: pd.Series,
    ebit_series: pd.Series,
    d_and_a_series: pd.Series | None,
    shares_out: float,
    net_debt: float,
    growth_range: np.ndarray,
    wacc_range: np.ndarray,
    terminal_g: float,
) -> pd.DataFrame:
    rows = []
    for g in growth_range:
        row = {}
        for w in wacc_range:
            res = dcf(
                revenue_series,
                ebit_series,
                d_and_a_series,
                shares_out,
                net_debt,
                wacc=w,
                terminal_g=terminal_g,
                override_growth=float(g),
            )
            row[round(float(w), 4)] = float(res["per_share"])
        rows.append(pd.Series(row, name=round(float(g), 4)))
    return pd.DataFrame(rows)
