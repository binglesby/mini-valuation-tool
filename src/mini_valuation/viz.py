import numpy as np
import pandas as pd
import plotly.graph_objects as go


def line_fcf(years: list[int], fcf: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=fcf, mode="lines+markers", name="Projected FCF"))
    fig.update_layout(title="Projected Free Cash Flow", xaxis_title="Year", yaxis_title="FCF")
    return fig


def heatmap_sensitivity(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=[str(c) for c in df.columns],
            y=[str(i) for i in df.index],
            colorbar=dict(title="Price"),
        )
    )
    fig.update_layout(title="Sensitivity: Growth (rows) vs WACC (cols)")
    return fig
