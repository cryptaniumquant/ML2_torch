#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Configuration: set input CSV and which signal column to use
FILE_PATH = "data/btc_new.csv"
SIGNAL_COLUMN = "predicted"  # options: 'predicted', 'actual', or 'signal'
PRICE_FILE = "data/BTCUSDT_1m_20210106_to_20250825.csv"

# Bar frequency for segmentation (e.g., '4H', '1H')
BAR_FREQ = "12H"

# Segmentation thresholds (quantiles)
# Hours with |hourly_return| <= RET_FLAT_Q quantile of |hourly_return| are 'Flat'
RET_FLAT_Q = 0.75
# Hours with intrahour volatility >= VOL_HIGH_Q quantile are 'High' vol, else 'Low'
VOL_HIGH_Q = 0.75

@dataclass
class DayMetrics:
    date: pd.Timestamp
    minutes: int
    buys: int
    sells: int
    trades: int
    p50_gap_min: Optional[float]
    p75_gap_min: Optional[float]
    faotd_50: Optional[float]
    faotd_75: Optional[float]


def load_signals(path: str, column: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])  # expects 'date' column
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in CSV. Available: {list(df.columns)}")

    # Coerce to integer signals 0/1/2
    sig = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    df = df.assign(signal=sig)
    # sort by time to ensure ordering
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "signal"]]


def load_price(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])  # expects 'Date', 'Close'
    if "Close" not in df.columns:
        raise ValueError(f"Column 'Close' not found in {path}. Available: {list(df.columns)}")
    return df[["Date", "Close"]].rename(columns={"Date": "date"}).sort_values("date").reset_index(drop=True)


def interarrival_gaps_minutes(times: pd.Series) -> pd.Series:
    if times.size < 2:
        return pd.Series(dtype=float)
    gaps = times.sort_values().diff().dropna().dt.total_seconds() / 60.0
    # filter any zero/negative (shouldn't occur if strict minutes, but keep robust)
    gaps = gaps[gaps > 0]
    return gaps


def compute_day_metrics(day: pd.DataFrame) -> DayMetrics:
    d = day.copy()
    minutes = len(d)
    buys = int((d.signal == 2).sum())
    sells = int((d.signal == 0).sum())
    trades = int((d.signal.isin([0, 2])).sum())

    trade_times = d.loc[d.signal.isin([0, 2]), "date"]
    gaps = interarrival_gaps_minutes(trade_times)

    p50_gap = float(gaps.quantile(0.50)) if not gaps.empty else None
    p75_gap = float(gaps.quantile(0.75)) if not gaps.empty else None

    fa50 = (minutes / p50_gap) if (p50_gap and p50_gap > 0) else None
    fa75 = (minutes / p75_gap) if (p75_gap and p75_gap > 0) else None

    return DayMetrics(
        date=pd.Timestamp(d.date.iloc[0].date()),
        minutes=minutes,
        buys=buys,
        sells=sells,
        trades=trades,
        p50_gap_min=p50_gap,
        p75_gap_min=p75_gap,
        faotd_50=fa50,
        faotd_75=fa75,
    )


def compute_run_lengths(directions: pd.Series) -> list:
    """Compute run lengths for a sequence of +1/-1 directions (ignores NaN).

    Example: [+1,+1,+1,-1,-1,+1] -> [3,2,1]
    """
    runs: list[int] = []
    prev = None
    count = 0
    for v in directions.dropna():
        if prev is None or v != prev:
            if count > 0:
                runs.append(count)
            prev = v
            count = 1
        else:
            count += 1
    if count > 0:
        runs.append(count)
    return runs


def compute_hourly_counts(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return hourly counts for buys(2) and sells(0) over the full time span, including zero hours.

    Output index is hourly datetime from min to max (inclusive). Missing hours are filled with 0.
    """
    tmp = df.copy()
    tmp["hour"] = tmp["date"].dt.floor("H")

    buys = tmp.loc[tmp["signal"] == 2].groupby("hour").size()
    sells = tmp.loc[tmp["signal"] == 0].groupby("hour").size()

    if tmp["hour"].empty:
        return pd.Series(dtype=int), pd.Series(dtype=int)

    start = tmp["hour"].min()
    end = tmp["hour"].max()
    full_idx = pd.date_range(start=start, end=end, freq="H")

    buys = buys.reindex(full_idx, fill_value=0)
    sells = sells.reindex(full_idx, fill_value=0)
    buys.index.name = sells.index.name = "hour"
    return buys.astype(int), sells.astype(int)


def build_hourly_features(price_df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """Compute hourly trend (Up/Down/Flat) and volatility (High/Low) features.

    - Trend is based on hourly close-to-close return with Flat defined by RET_FLAT_Q of |returns|.
    - Volatility is std of 1m log-returns within each hour; High/Low split by VOL_HIGH_Q.
    """
    # Restrict to signals window (pad optional)
    mask = (price_df["date"] >= t0.floor(BAR_FREQ)) & (price_df["date"] <= t1.ceil(BAR_FREQ))
    px = price_df.loc[mask].copy()
    if px.empty:
        return pd.DataFrame(columns=["hour", "ret", "abs_ret", "vol", "trend", "vol_level", "segment"]).set_index("hour")

    # Minute log returns
    px["logp"] = np.log(px["Close"].astype(float))
    px["ret1m"] = px["logp"].diff()
    px["hour"] = px["date"].dt.floor(BAR_FREQ)

    # Hourly close and return
    hourly_close = px.groupby("hour")["Close"].last()
    hourly_ret = hourly_close.pct_change().fillna(0.0)
    abs_ret = hourly_ret.abs()

    # Intrahour volatility: std of 1m log-returns per hour
    grp_min = px.groupby("hour")["ret1m"]
    vol_h = grp_min.std().fillna(0.0)
    n_min = grp_min.count().reindex(vol_h.index).fillna(0).astype(int)
    # Convert to 4H log-return std by sqrt(n) scaling of 1m std, then to percent
    vol4h = (vol_h * np.sqrt(np.maximum(n_min, 1))).astype(float)
    vol4h_pct = vol4h * 100.0

    # Thresholds
    ret_flat_thr = float(abs_ret.quantile(RET_FLAT_Q)) if len(abs_ret) else 0.0
    vol_high_thr = float(vol_h.quantile(VOL_HIGH_Q)) if len(vol_h) else 0.0

    trend = pd.Series(index=hourly_ret.index, dtype=object)
    trend[abs_ret <= ret_flat_thr] = "Flat"
    trend[(abs_ret > ret_flat_thr) & (hourly_ret > 0)] = "Up"
    trend[(abs_ret > ret_flat_thr) & (hourly_ret < 0)] = "Down"

    vol_level = pd.Series(np.where(vol_h >= vol_high_thr, "High", "Low"), index=vol_h.index)

    feats = pd.DataFrame({
        "ret": hourly_ret,
        "abs_ret": abs_ret,
        "vol": vol_h,
        "n_min": n_min,
        "ret_pct": hourly_ret * 100.0,
        "ret_pct_abs": abs_ret * 100.0,
        "vol4h_pct": vol4h_pct,
        "trend": trend,
        "vol_level": vol_level,
    })
    feats.index.name = "hour"
    feats["segment"] = feats.apply(lambda r: f"{r['trend']}-{r['vol_level']}", axis=1)
    feats["segment"] = feats["segment"].where(feats["trend"].notna(), other="Unknown")

    feats.attrs["ret_flat_thr"] = ret_flat_thr
    feats.attrs["vol_high_thr"] = vol_high_thr
    return feats


def aggregate_by_segment(signals: pd.DataFrame, feats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate buy/sell counts per segment.

    Returns (sum_table, mean_table, median_table, thresholds).

    sum_table columns: [hours, buys, sells, total, long_share, long_short_ratio]
    mean/median tables (per 4H window within each segment):
      columns: [windows, buys, sells, total, long_share, long_short_ratio]
    thresholds: dict with ret_flat_thr and vol_high_thr
    """
    if feats.empty:
        empty_sum = pd.DataFrame(columns=["segment", "hours", "buys", "sells", "total", "long_share", "long_short_ratio"]).set_index("segment")
        empty_avg = pd.DataFrame(columns=["segment", "windows", "buys", "sells", "total", "long_share", "long_short_ratio"]).set_index("segment")
        return empty_sum, empty_avg, empty_avg, {}

    # Hourly counts of signals
    sig = signals.copy()
    sig["hour"] = sig["date"].dt.floor(BAR_FREQ)
    buys_h = sig.loc[sig["signal"] == 2].groupby("hour").size()
    sells_h = sig.loc[sig["signal"] == 0].groupby("hour").size()

    # Align to all hours in feats
    idx = feats.index
    buys_h = buys_h.reindex(idx, fill_value=0).astype(int)
    sells_h = sells_h.reindex(idx, fill_value=0).astype(int)

    df = feats.copy()
    df["buys"] = buys_h
    df["sells"] = sells_h
    df["total"] = df["buys"] + df["sells"]
    # Per-window shares/ratios
    df["long_share_win"] = np.where(df["total"] > 0, df["buys"] / df["total"], np.nan)
    df["lsr_win"] = np.where(df["sells"] > 0, df["buys"] / df["sells"], np.nan)

    # Group by segment
    grp = df.groupby("segment")
    agg = grp.aggregate(hours=("ret", "size"), buys=("buys", "sum"), sells=("sells", "sum"))
    agg["total"] = agg["buys"] + agg["sells"]
    agg["long_share"] = np.where(agg["total"] > 0, agg["buys"] / agg["total"], np.nan)
    agg["long_short_ratio"] = np.where(agg["sells"] > 0, agg["buys"] / agg["sells"], np.nan)

    # Per-segment averages across 4H windows
    mean_tbl = grp.aggregate(
        windows=("ret", "size"),
        buys=("buys", "mean"),
        sells=("sells", "mean"),
        total=("total", "mean"),
        ret_pct=("ret_pct", "mean"),
        ret_pct_abs=("ret_pct_abs", "mean"),
        vol4h_pct=("vol4h_pct", "mean"),
        long_share=("long_share_win", "mean"),
        long_short_ratio=("lsr_win", "mean"),
    )
    median_tbl = grp.aggregate(
        windows=("ret", "size"),
        buys=("buys", "median"),
        sells=("sells", "median"),
        total=("total", "median"),
        ret_pct=("ret_pct", "median"),
        ret_pct_abs=("ret_pct_abs", "median"),
        vol4h_pct=("vol4h_pct", "median"),
        long_share=("long_share_win", "median"),
        long_short_ratio=("lsr_win", "median"),
    )

    # Order segments if present
    order = ["Up-High", "Up-Low", "Down-High", "Down-Low", "Flat-High", "Flat-Low", "Unknown"]
    agg = agg.reindex([s for s in order if s in agg.index])
    mean_tbl = mean_tbl.reindex([s for s in order if s in mean_tbl.index])
    median_tbl = median_tbl.reindex([s for s in order if s in median_tbl.index])

    thresholds = {
        "ret_flat_thr": feats.attrs.get("ret_flat_thr", np.nan),
        "vol_high_thr": feats.attrs.get("vol_high_thr", np.nan),
    }
    return agg, mean_tbl, median_tbl, thresholds


def print_segmentation_report(seg_sum: pd.DataFrame, seg_mean: pd.DataFrame, seg_median: pd.DataFrame, thresholds: dict):
    print("=== Market Segmentation by Trend and Volatility ===")
    print(f"Bar frequency: {BAR_FREQ}")
    print(f"Flat threshold |return per {BAR_FREQ}| <= {thresholds.get('ret_flat_thr', float('nan')):.6f}")
    print(f"High vol threshold (std 1m log-ret within {BAR_FREQ}) >= {thresholds.get('vol_high_thr', float('nan')):.6f}")
    print()
    if seg_sum.empty:
        print("No data for the selected window.")
        return

    def fmt_tbl(tbl: pd.DataFrame, share_cols=("long_share", "long_short_ratio")) -> pd.DataFrame:
        out = tbl.copy()
        # Round key numeric columns for readability
        for col in [c for c in out.columns if c in ("buys", "sells", "total", "ret_pct", "ret_pct_abs", "vol4h_pct")]:
            out[col] = out[col].astype(float).round(2)
        for col in share_cols:
            if col in out.columns:
                out[col] = out[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")
        return out

    print("-- Totals per segment --")
    print(fmt_tbl(seg_sum).to_string())
    print()

    if not seg_mean.empty:
        print("-- Averages per 4H (mean) --")
        print(fmt_tbl(seg_mean).to_string())
        print()

    if not seg_median.empty:
        print("-- Averages per 4H (median) --")
        print(fmt_tbl(seg_median).to_string())
        print()


def compute_trend_run_stats(feats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute duration stats of contiguous trend runs (ignoring volatility level).

    Returns (per_trend_table, overall_table), with columns:
      runs, mean_bars, median_bars, mean_hours, median_hours
    """
    if feats.empty or feats["trend"].dropna().empty:
        empty = pd.DataFrame(columns=["runs", "mean_bars", "median_bars", "mean_hours", "median_hours"], index=["Up", "Down", "Flat"])
        overall = pd.DataFrame(columns=["runs", "mean_bars", "median_bars", "mean_hours", "median_hours"], index=["All"]) 
        return empty, overall

    s = feats.sort_index()["trend"].fillna("Unknown").values
    # Compute run lengths and labels
    lengths = []
    labels = []
    if len(s) > 0:
        cur = s[0]
        cnt = 1
        for x in s[1:]:
            if x == cur:
                cnt += 1
            else:
                labels.append(cur)
                lengths.append(cnt)
                cur = x
                cnt = 1
        labels.append(cur)
        lengths.append(cnt)

    runs_df = pd.DataFrame({"trend": labels, "bars": lengths})
    # Filter to only the three trends of interest
    runs_df = runs_df[runs_df["trend"].isin(["Up", "Down", "Flat"])].reset_index(drop=True)

    hours_per_bar = pd.to_timedelta(BAR_FREQ).total_seconds() / 3600.0
    runs_df["hours"] = runs_df["bars"] * hours_per_bar

    # Per-trend stats
    per_trend = runs_df.groupby("trend").agg(
        runs=("bars", "size"),
        mean_bars=("bars", "mean"),
        median_bars=("bars", "median"),
        mean_hours=("hours", "mean"),
        median_hours=("hours", "median"),
    )

    # Overall stats
    overall = pd.DataFrame({
        "runs": [runs_df.shape[0]],
        "mean_bars": [runs_df["bars"].mean()],
        "median_bars": [runs_df["bars"].median()],
        "mean_hours": [runs_df["hours"].mean()],
        "median_hours": [runs_df["hours"].median()],
    }, index=["All"])

    return per_trend, overall


def print_trend_run_stats(per_trend: pd.DataFrame, overall: pd.DataFrame):
    print("=== Trend Run Duration Stats (by trend, ignoring volatility) ===")
    if per_trend.empty:
        print("No trend runs found.")
        return
    def fmt(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in ["mean_bars", "median_bars", "mean_hours", "median_hours"]:
            if col in out.columns:
                out[col] = out[col].astype(float).round(2)
        return out
    print("-- Per trend --")
    print(fmt(per_trend).to_string())
    if not overall.empty:
        print()
        print("-- Overall --")
        print(fmt(overall).to_string())


def analyze(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    # Per-day grouping
    df = df.copy()
    df["day"] = df["date"].dt.date

    day_rows = []
    for day, grp in df.groupby("day", sort=True):
        day_rows.append(compute_day_metrics(grp))

    day_df = pd.DataFrame([vars(x) for x in day_rows])

    # Summary stats
    summary = {}
    if not day_df.empty:
        summary["days"] = len(day_df)
        summary["avg_buys_per_day"] = day_df.buys.mean()
        summary["avg_sells_per_day"] = day_df.sells.mean()
        summary["avg_trades_per_day"] = day_df.trades.mean()

        # Inter-arrival overall (concatenate per-day gaps for robustness)
        gaps_all = []
        for _, grp in df.groupby("day", sort=True):
            gaps = interarrival_gaps_minutes(grp.loc[grp.signal.isin([0, 2]), "date"])
            if not gaps.empty:
                gaps_all.append(gaps)
        if gaps_all:
            gaps_all = pd.concat(gaps_all, ignore_index=True)
            summary["gap_min_mean"] = gaps_all.mean()
            summary["gap_min_median"] = gaps_all.median()
            summary["gap_min_p75"] = gaps_all.quantile(0.75)
            # Frequency-Aware Optimal Trades per Day (FAOTD): D / p75_gap
            # Provide both mean of daily FAOTD and global using global p75 gap
            if (day_df.faotd_75.notna()).any():
                summary["faotd_75_daily_mean"] = day_df.faotd_75.mean()
                summary["faotd_75_daily_median"] = day_df.faotd_75.median()
            # Global
            # Estimate average minutes per day from data
            avg_minutes_day = day_df.minutes.mean()
            global_p75 = float(summary["gap_min_p75"]) if summary.get("gap_min_p75") else None
            if global_p75 and global_p75 > 0:
                summary["faotd_75_global"] = avg_minutes_day / global_p75
        else:
            summary["gap_min_mean"] = np.nan
            summary["gap_min_median"] = np.nan
            summary["gap_min_p75"] = np.nan
            summary["faotd_75_daily_mean"] = np.nan
            summary["faotd_75_daily_median"] = np.nan
            summary["faotd_75_global"] = np.nan

        # Global run-lengths over the entire history (ignoring holds)
        df_sorted = df.sort_values("date")
        mask = df_sorted["signal"].isin([0, 2])
        dirs = df_sorted.loc[mask, "signal"].map({2: 1, 0: -1}).astype("Int64")
        runs = compute_run_lengths(dirs)
        if len(runs) > 0:
            runs_s = pd.Series(runs, dtype=float)
            summary["runlen_count"] = int(len(runs))
            summary["runlen_mean"] = float(runs_s.mean())
            summary["runlen_median"] = float(runs_s.median())
            summary["runlen_p75"] = float(runs_s.quantile(0.95))
            p75_val = summary["runlen_p75"]
            summary["global_cap_runlen_p75"] = int(np.floor(p75_val)) if (p75_val and p75_val > 0) else np.nan
        else:
            summary["runlen_count"] = 0
            summary["runlen_mean"] = np.nan
            summary["runlen_median"] = np.nan
            summary["runlen_p75"] = np.nan
            summary["global_cap_runlen_p75"] = np.nan

        # Hourly buy/sell counts and their quantiles over entire period
        buys_h, sells_h = compute_hourly_counts(df)
        summary["hours_total"] = int(len(buys_h)) if len(buys_h) else 0
        if len(buys_h):
            summary["buys_hour_mean"] = float(buys_h.mean())
            summary["buys_hour_p50"] = float(buys_h.quantile(0.50))
            summary["buys_hour_p95"] = float(buys_h.quantile(0.95))
        else:
            summary["buys_hour_mean"] = np.nan
            summary["buys_hour_p50"] = np.nan
            summary["buys_hour_p95"] = np.nan
        if len(sells_h):
            summary["sells_hour_mean"] = float(sells_h.mean())
            summary["sells_hour_p50"] = float(sells_h.quantile(0.50))
            summary["sells_hour_p95"] = float(sells_h.quantile(0.95))
        else:
            summary["sells_hour_mean"] = np.nan
            summary["sells_hour_p50"] = np.nan
            summary["sells_hour_p95"] = np.nan

    return day_df, summary


def print_report(day_df: pd.DataFrame, summary: dict):
    print("=== Daily Buy/Sell/Trade Counts ===")
    print(day_df[["date", "minutes", "buys", "sells", "trades"]].to_string(index=False))
    print()

    print("=== Averages per day ===")
    print(f"Days: {summary.get('days', 0)}")
    print(f"Avg buys/day:  {summary.get('avg_buys_per_day', float('nan')):.2f}")
    print(f"Avg sells/day: {summary.get('avg_sells_per_day', float('nan')):.2f}")
    print(f"Avg trades/day:{summary.get('avg_trades_per_day', float('nan')):.2f}")
    print()

    print("=== Inter-arrival gaps (minutes) between trade signals (0/2) ===")
    print(f"Mean:   {summary.get('gap_min_mean', float('nan')):.2f}")
    print(f"Median: {summary.get('gap_min_median', float('nan')):.2f}")
    print(f"P75:    {summary.get('gap_min_p75', float('nan')):.2f}")
    print()

    print("=== Frequency-Aware Optimal Trades per Day (FAOTD) ===")
    print("Definition: FAOTD = active_minutes_per_day / P75(inter-arrival minutes). Lower is more conservative.")
    if not np.isnan(summary.get("faotd_75_daily_mean", np.nan)):
        print(f"Daily FAOTD mean:   {summary['faotd_75_daily_mean']:.2f}")
        print(f"Daily FAOTD median: {summary['faotd_75_daily_median']:.2f}")
    if not np.isnan(summary.get("faotd_75_global", np.nan)):
        print(f"Global FAOTD (using global P75 gap): {summary['faotd_75_global']:.2f}")
    print("Suggested integer cap per day (conservative):",
          int(np.floor(summary.get("faotd_75_daily_median", 0))) if not np.isnan(summary.get("faotd_75_daily_median", np.nan)) else "n/a")
    print()

    print("=== Global Position Cap from Signal Run-Lengths (P75) ===")
    if not np.isnan(summary.get("runlen_p75", np.nan)):
        print(f"Run-length P75: {summary['runlen_p75']:.2f}")
        cap_val = summary.get("global_cap_runlen_p75", np.nan)
        print("Global position cap (floor P75):", int(cap_val) if not np.isnan(cap_val) else "n/a")
    else:
        print("Run-lengths not available (insufficient trade signals).")
    print()

    print("=== Hourly Buy/Sell Counts (Quantiles) ===")
    print(f"Hours considered: {summary.get('hours_total', 0)}")
    print("Buys/hour:")
    print(f"  mean: {summary.get('buys_hour_mean', float('nan')):.2f}")
    print(f"  p50:  {summary.get('buys_hour_p50', float('nan')):.2f}")
    print(f"  p95:  {summary.get('buys_hour_p95', float('nan')):.2f}")
    print("Sells/hour:")
    print(f"  mean: {summary.get('sells_hour_mean', float('nan')):.2f}")
    print(f"  p50:  {summary.get('sells_hour_p50', float('nan')):.2f}")
    print(f"  p95:  {summary.get('sells_hour_p95', float('nan')):.2f}")


def main():
    # Load signals
    sig = load_signals(FILE_PATH, SIGNAL_COLUMN)
    if sig.empty:
        print("No signals found.")
        return
    # Load price and build hourly features within signals time window
    px = load_price(PRICE_FILE)
    t0, t1 = sig["date"].min(), sig["date"].max()
    feats = build_hourly_features(px, t0, t1)
    seg_sum, seg_mean, seg_median, thresholds = aggregate_by_segment(sig, feats)
    print_segmentation_report(seg_sum, seg_mean, seg_median, thresholds)
    # Trend duration statistics (by Up/Down/Flat)
    per_trend, overall = compute_trend_run_stats(feats)
    print_trend_run_stats(per_trend, overall)


if __name__ == "__main__":
    main()
