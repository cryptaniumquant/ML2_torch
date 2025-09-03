#!/usr/bin/env python3
"""
Plot signals frequency and ratio together with price using Plotly.
- Filters the price range to the time span covered by signals.
- Shows:
  * Minute Close price (primary Y axis)
  * Daily trades count (0 or 2) as bars (secondary Y axis)
  * Daily Buy/Sell ratio as a line (secondary Y axis)
Outputs: signals_price_plot.html
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === Configuration ===
SIGNALS_FILE = "data/sol_new.csv"           # must contain columns: 'date' and one of ['predicted','actual','signal']
SIGNAL_COLUMN = "predicted"                  # choose which signal column to use
PRICE_FILE = "data/SOLUSDT_1m_20210101_to_20250820.csv"  # must contain 'Date' and 'Close'
OUTPUT_HTML = "signals_price_plot.html"

# Signals mapping: 0=sell,1=hold,2=buy


def load_signals(path: str, column: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])  # expects 'date'
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {path}. Available: {list(df.columns)}")
    df = df.sort_values("date").reset_index(drop=True)
    # Normalize to Int64 signals
    sig = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    return pd.DataFrame({"date": df["date"], "signal": sig})


def load_price(path: str) -> pd.DataFrame:
    # Known schema from workspace: Open,High,Low,Close,Volume,Date
    df = pd.read_csv(path, parse_dates=["Date"])  # capital D
    if "Close" not in df.columns:
        raise ValueError(f"Column 'Close' not found in {path}. Available: {list(df.columns)}")
    return df[["Date", "Close"]].rename(columns={"Date": "date"}).sort_values("date").reset_index(drop=True)


def compute_daily_stats(sig_df: pd.DataFrame) -> pd.DataFrame:
    d = sig_df.copy()
    d["day"] = d["date"].dt.date
    # Counts per day
    daily = d.groupby("day").agg(
        buys=("signal", lambda s: int((s == 2).sum())),
        sells=("signal", lambda s: int((s == 0).sum())),
    )
    daily["trades"] = daily["buys"] + daily["sells"]
    # Buy/Sell ratio; avoid division by zero
    daily["buy_sell_ratio"] = daily.apply(lambda r: (r.buys / r.sells) if r.sells > 0 else np.nan, axis=1)
    # For plotting as datetime on x
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    return daily


def make_figure(price_df: pd.DataFrame, daily_stats: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Price line (primary y)
    fig.add_trace(
        go.Scatter(
            x=price_df["date"],
            y=price_df["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#1f77b4", width=1.5),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Close: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Daily trades as bars (secondary y)
    bar_width_ms = 24 * 60 * 60 * 1000 * 0.8
    fig.add_trace(
        go.Bar(
            x=daily_stats.index,
            y=daily_stats["trades"],
            name="Trades/day",
            marker_color="#ff7f0e",
            opacity=0.45,
            width=bar_width_ms,
            hovertemplate="%{x|%Y-%m-%d}<br>Trades: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Buy/Sell ratio line (secondary y)
    fig.add_trace(
        go.Scatter(
            x=daily_stats.index,
            y=daily_stats["buy_sell_ratio"],
            mode="lines+markers",
            name="Buy/Sell ratio",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=5),
            hovertemplate="%{x|%Y-%m-%d}<br>Buy/Sell: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Price with Daily Signal Frequency and Buy/Sell Ratio",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=60, b=40),
        hovermode="x unified",
        bargap=0.15,
        template="plotly_white",
    )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Close", secondary_y=False)
    fig.update_yaxes(title_text="Signals / Ratio", secondary_y=True)

    return fig


def main():
    # Load data
    sig_df = load_signals(SIGNALS_FILE, SIGNAL_COLUMN)
    if sig_df.empty:
        print("Signals file is empty or invalid")
        return

    # Determine time window from signals
    t0, t1 = sig_df["date"].min(), sig_df["date"].max()

    price_df = load_price(PRICE_FILE)
    # Restrict to signals time window (with a small margin of one day for context)
    pad = pd.Timedelta(days=0)
    mask = (price_df["date"] >= (t0 - pad)) & (price_df["date"] <= (t1 + pad))
    price_win = price_df.loc[mask].reset_index(drop=True)
    if price_win.empty:
        print("No price data in the signals time window.")
        return

    # Daily stats
    daily_stats = compute_daily_stats(sig_df)
    # Restrict daily stats to days present in the price window
    days_mask = (daily_stats.index.date >= price_win["date"].min().date()) & (
        daily_stats.index.date <= price_win["date"].max().date()
    )
    daily_stats = daily_stats.loc[days_mask]

    fig = make_figure(price_win, daily_stats)

    # Output to HTML
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
    print(f"Saved interactive chart to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
