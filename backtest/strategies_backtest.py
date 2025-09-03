import pandas as pd
import numpy as np
import talib
import logging
from copy import deepcopy
from datetime import datetime, timezone
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns

log = logging.getLogger("backtest")

# Global variable to store signals data
_signals_data = None


def load_signals_data(signals_file_path):
    """Load signals data from CSV file and create optimized lookup dictionary"""
    global _signals_data
    if _signals_data is None:
        df = pd.read_csv(signals_file_path)
        df['date'] = pd.to_datetime(df['date'])
        # Create dictionary for O(1) lookup instead of O(n) search
        _signals_data = dict(zip(df['date'], df['predicted'].astype(int)))
        log.info(f"Loaded {len(_signals_data)} signals from {signals_file_path}")
    return _signals_data


def get_signal_for_datetime(target_datetime, signals_dict):
    """Get signal for specific datetime using optimized dictionary lookup"""
    # Direct lookup in dictionary (much faster than pandas operations)
    return signals_dict.get(target_datetime, None)


def ensure_trailing_columns(positions_df: pd.DataFrame) -> pd.DataFrame:
    needed = ['initial_stop', 'initial_take', 'initial_risk']
    for col in needed:
        if col not in positions_df.columns:
            positions_df[col] = np.nan
    return positions_df


def update_trailing_levels(
        positions_df: pd.DataFrame,
        ohlc_df: pd.DataFrame,
        trail_mode: str = 'atr_band',  # 'atr_band' | 'chandelier'
        atr_period: int = 140,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 10.0,
        chandelier_lookback: int = 22,
        breakeven_rr: float | None = 1.0,  # None чтобы отключить
        lock_in_frac: float = 0.2  # доля риска, которую фиксируем после BE
) -> pd.DataFrame:
    """
    Пересчитывает SL/TP для всех открытых позиций на последнем баре.
    Трейлинг — «трещотка»: long SL/TP не понижаются, short SL/TP не повышаются.
    """
    if positions_df.empty:
        return positions_df

    positions_df = ensure_trailing_columns(positions_df)

    high = ohlc_df['High'].values
    low = ohlc_df['Low'].values
    close = ohlc_df['Close'].values
    price = float(close[-1])

    # ATR на последнем баре
    atr_series = talib.ATR(high, low, close, timeperiod=atr_period)
    atr = float(atr_series[-1]) if not np.isnan(atr_series[-1]) else None
    if atr is None or atr <= 0:
        # нет данных — ничего не двигаем
        return positions_df

    # Для Chandelier — локальный экстремум
    hh = pd.Series(high).rolling(chandelier_lookback).max().iloc[-1] if trail_mode == 'chandelier' else None
    ll = pd.Series(low).rolling(chandelier_lookback).min().iloc[-1] if trail_mode == 'chandelier' else None

    open_mask = positions_df['exit_date'].isna()
    for idx, pos in positions_df[open_mask].iterrows():
        side = pos['position_type']
        entry = float(pos['entry_price'])
        curSL = pos['stop_price']
        curTP = pos['take_price']
        iSL = pos['initial_stop'] if pd.notna(pos['initial_stop']) else curSL
        iRisk = pos['initial_risk'] if pd.notna(pos['initial_risk']) else (
            (entry - curSL) if side == 'long' else (curSL - entry)
        )
        # Защита от NaN
        if pd.isna(curSL): curSL = entry - (atr * sl_atr_mult) if side == 'long' else entry + (atr * sl_atr_mult)
        if pd.isna(curTP): curTP = entry + (atr * tp_atr_mult) if side == 'long' else entry - (atr * tp_atr_mult)
        if pd.isna(iRisk) or iRisk <= 0:
            iRisk = (entry - curSL) if side == 'long' else (curSL - entry)

        # --- Кандидаты уровней ---
        if trail_mode == 'chandelier':
            if side == 'long':
                sl_candidate = hh - sl_atr_mult * atr
                tp_candidate = price + tp_atr_mult * atr  # TP всё равно тащим ATR-полосой вверх
            else:
                sl_candidate = ll + sl_atr_mult * atr
                tp_candidate = price - tp_atr_mult * atr
        else:  # 'atr_band'
            if side == 'long':
                sl_candidate = price - sl_atr_mult * atr
                tp_candidate = price + tp_atr_mult * atr
            else:
                sl_candidate = price + sl_atr_mult * atr
                tp_candidate = price - tp_atr_mult * atr

        # --- Трещотка (монотонность) ---
        if side == 'long':
            newSL = max(curSL, sl_candidate)
            newTP = max(curTP, tp_candidate)
        else:
            newSL = min(curSL, sl_candidate)
            newTP = min(curTP, tp_candidate)

        # --- Брейк-ивен и частичная фиксация ---
        if breakeven_rr is not None and iRisk > 0:
            if side == 'long':
                float_pnl = price - entry
                if float_pnl >= breakeven_rr * iRisk:
                    beSL = entry + lock_in_frac * iRisk
                    print(
                        f"  Breakeven triggered for {side} pos {idx}: float_pnl={float_pnl:.2f}, threshold={breakeven_rr * iRisk:.2f}, new BE SL={beSL:.2f}")
                    newSL = max(newSL, beSL)
            else:
                float_pnl = entry - price
                if float_pnl >= breakeven_rr * iRisk:
                    beSL = entry - lock_in_frac * iRisk
                    print(
                        f"  Breakeven triggered for {side} pos {idx}: float_pnl={float_pnl:.2f}, threshold={breakeven_rr * iRisk:.2f}, new BE SL={beSL:.2f}")
                    newSL = min(newSL, beSL)

        # Debug print if levels changed
        if newSL != curSL or newTP != curTP:
            print(
                f"  Trailing update for {side} pos {idx} at {price:.2f}: SL {curSL:.2f} -> {newSL:.2f}, TP {curTP:.2f} -> {newTP:.2f}")

        # Запись обратно
        positions_df.at[idx, 'stop_price'] = newSL
        positions_df.at[idx, 'take_price'] = newTP
        # Если не заполнены стартовые, зафиксируем для будущей логики
        if pd.isna(positions_df.at[idx, 'initial_stop']):
            positions_df.at[idx, 'initial_stop'] = curSL
        if pd.isna(positions_df.at[idx, 'initial_take']):
            positions_df.at[idx, 'initial_take'] = curTP
        if pd.isna(positions_df.at[idx, 'initial_risk']) or positions_df.at[idx, 'initial_risk'] <= 0:
            positions_df.at[idx, 'initial_risk'] = iRisk

    return positions_df


def calculate_exponential_close_count(net_position_size, base_close=1, exp_factor=0.1):
    """
    Calculate how many positions to close based on net position size using exponential scaling

    Args:
        net_position_size: Absolute value of net positions (longs - shorts)
        base_close: Base number of positions to close (default 1)
        exp_factor: Exponential scaling factor (default 0.1)

    Returns:
        Number of positions to close (at least 1, capped at net_position_size)
    """
    if net_position_size <= 0:
        return 0

    # Exponential scaling: base_close * e^(exp_factor * net_position_size)
    exponential_close = int(base_close * np.exp(exp_factor * net_position_size))

    # Cap at available positions and ensure at least 1
    return min(max(exponential_close, 1), net_position_size)


def ml_strategy_torch(Bars_df: pd.DataFrame,
                      current_date,
                      symbol,
                      series_coin,
                      fixed_trade_size,
                      pctOfReinvest,
                      signals_dict,
                      max_pos_count,
                      strategy_name,
                      state,
                      positions_df,
                      current_balance,
                      trading_mode="signal_based",
                      stop_loss_pct=0.02,
                      take_profit_pct=0.04,
                      exp_factor=0.05,
                      override_signal=None):
    oco = (state or {}).get('OCO', [])

    symbol = symbol.upper()
    close, high, low = Bars_df.iloc[-1][['Close', 'High', 'Low']]
    current_price = close
    current_date = Bars_df['Date'].iloc[-1]
    # Use original 1-minute data for ATR calculation instead of resampling
    df_for_atr = Bars_df[['Open', 'Close', 'High', 'Low']].copy()

    series_coin = Bars_df['Close']
    log.debug('len=%s, limit=%s, %s', len(oco), max_pos_count, state)

    # Signal-based position management - no TP/SL, only signal-driven closes

    # Get signal for current datetime (signals_dict already loaded)
    current_datetime = pd.to_datetime(current_date)

    # Use override signal if provided (for realistic timing), otherwise get current signal
    if override_signal is not None:
        predicted_class = override_signal
    else:
        predicted_class = get_signal_for_datetime(current_datetime, signals_dict)

    # Get open positions (not closed yet)
    open_positions = positions_df[positions_df['exit_date'].isna()]

    # Trading logic based on mode
    if trading_mode == "signal_based":
        # Signal-based mode: only signals control positions, no SL/TP
        if predicted_class is not None:
            positions_df, current_balance = handle_signal_based_trading(predicted_class, current_price,
                                                                        current_datetime, symbol, fixed_trade_size,
                                                                        positions_df, current_balance, max_pos_count,
                                                                        exp_factor)
    elif trading_mode == "sl_tp":
        # 1) Сначала подтянем трейлинг уровни на текущем баре
        positions_df = update_trailing_levels(
            positions_df=positions_df,
            ohlc_df=Bars_df,  # последние ~300 баров
            trail_mode='atr_band',  # или 'chandelier'
            atr_period=140,
            sl_atr_mult=8.5,
            tp_atr_mult=8.0,
            chandelier_lookback=22,
            breakeven_rr=1.0,  # None, чтобы отключить
            lock_in_frac=0.2
        )

        # 2) Потом проверяем срабатывание OCO с учётом обновлённых уровней
        current_ohlc = {
            'Open': Bars_df.iloc[-1]['Open'], 'High': Bars_df.iloc[-1]['High'],
            'Low': Bars_df.iloc[-1]['Low'], 'Close': Bars_df.iloc[-1]['Close']
        }
        current_balance = check_sl_tp_exits(
            positions_df, current_ohlc, current_datetime, current_balance,
            stop_loss_pct, take_profit_pct
        )

        # 3) И уже затем можем открыть новые позиции по новому сигналу
        if predicted_class is not None:
            positions_df, current_balance = handle_sl_tp_trading(
                predicted_class, current_price, current_datetime, symbol, fixed_trade_size,
                positions_df, current_balance, max_pos_count, stop_loss_pct, take_profit_pct
            )
    return positions_df, state, current_balance


def check_sl_tp_exits(positions_df, current_ohlc, current_datetime, current_balance, stop_loss_pct, take_profit_pct):
    """Check and execute OCO (One-Cancels-Other) stop-loss and take-profit exits using full OHLC data"""
    open_positions = positions_df[positions_df['exit_date'].isna()].copy()

    # Extract OHLC prices
    open_price = current_ohlc['Open']
    high_price = current_ohlc['High']
    low_price = current_ohlc['Low']
    close_price = current_ohlc['Close']

    for idx, pos in open_positions.iterrows():
        entry_price = pos['entry_price']
        position_type = pos['position_type']
        order_qty = pos['order_qty']

        # Use stored SL/TP levels from position (OCO logic)
        stop_price = pos.get('stop_price', np.nan)
        take_price = pos.get('take_price', np.nan)

        # Skip positions without SL/TP levels (signal-based positions)
        if pd.isna(stop_price) or pd.isna(take_price):
            continue

        should_close = False
        exit_reason = ""
        exit_price = close_price  # Default to close price

        if position_type == 'long':
            # Long position: check if intrabar price hit SL or TP
            if low_price <= stop_price:
                should_close = True
                exit_reason = "Stop Loss"
                exit_price = stop_price  # Exit at SL level
            elif high_price >= take_price:
                should_close = True
                exit_reason = "Take Profit"
                exit_price = take_price  # Exit at TP level

        else:  # short position
            # Short position: check if intrabar price hit SL or TP
            if high_price >= stop_price:
                should_close = True
                exit_reason = "Stop Loss"
                exit_price = stop_price  # Exit at SL level
            elif low_price <= take_price:
                should_close = True
                exit_reason = "Take Profit"
                exit_price = take_price  # Exit at TP level

        if should_close:
            # Calculate PnL using actual exit price
            if position_type == 'long':
                pnl = (exit_price - entry_price) * order_qty
            else:  # short
                pnl = (entry_price - exit_price) * order_qty

            # Update position (OCO: one order executed, other cancelled)
            positions_df.loc[idx, 'exit_date'] = current_datetime
            positions_df.loc[idx, 'exit_price'] = exit_price
            positions_df.loc[idx, 'pnl'] = pnl

            # Update balance
            current_balance += pnl

            print(
                f"  {exit_reason}: Closing {position_type.upper()} | Entry=${entry_price:.2f}, Exit=${exit_price:.2f}, SL=${stop_price:.2f}, TP=${take_price:.2f}, PnL=${pnl:.2f}")

    return current_balance


def handle_signal_based_trading(predicted_class, price, current_datetime, symbol, fixed_trade_size,
                                positions_df, current_balance, max_pos_count, exp_factor=0.05):
    """Handle signal-based trading logic with exponential position closing"""

    # Calculate current net position
    open_positions = positions_df[positions_df['exit_date'].isna()]
    open_longs = open_positions[open_positions['position_type'] == 'long']
    open_shorts = open_positions[open_positions['position_type'] == 'short']
    net_long_positions = len(open_longs)
    net_short_positions = len(open_shorts)

    if predicted_class == 2:  # Buy signal - close shorts exponentially OR open long position
        if not open_shorts.empty:
            # Calculate how many shorts to close exponentially
            close_count = calculate_exponential_close_count(net_short_positions, base_close=1, exp_factor=exp_factor)
            close_count = min(close_count, len(open_shorts))  # Don't exceed available positions

            print(f"  Exponential closing: {close_count} SHORT positions (net shorts: {net_short_positions})")

            # Close multiple short positions (FIFO - oldest first)
            shorts_to_close = open_shorts.head(close_count)
            total_pnl = 0

            for idx, pos in shorts_to_close.iterrows():
                entry_price = pos['entry_price']
                order_qty = pos['order_qty']

                # Calculate PnL for short position
                pnl = (entry_price - price) * order_qty
                total_pnl += pnl

                # Update position
                positions_df.loc[idx, 'exit_date'] = current_datetime
                positions_df.loc[idx, 'exit_price'] = price
                positions_df.loc[idx, 'pnl'] = pnl

                print(
                    f"    Closing SHORT: Entry=${entry_price:.2f}, Exit=${price:.2f}, Qty={order_qty:.4f}, PnL=${pnl:.2f}")

            # Update balance
            current_balance += total_pnl
            print(f"  Total PnL from {close_count} SHORT closures: ${total_pnl:.2f}")

        else:
            # No shorts to close, open new long position if under limit
            total_open_positions = len(positions_df[positions_df['exit_date'].isna()])
            if total_open_positions < max_pos_count:
                order_qty = fixed_trade_size / price
                new_position = pd.DataFrame({
                    'entry_date': [current_datetime],
                    'entry_price': [price],
                    'symbol': [symbol],
                    'order_qty': [order_qty],
                    'position_type': ['long'],
                    'exit_date': [pd.NaT],
                    'exit_price': [np.nan],
                    'pnl': [np.nan],
                    'stop_price': [np.nan],
                    'take_price': [np.nan]
                })
                if len(positions_df) == 0:
                    positions_df = new_position
                else:
                    positions_df = pd.concat([positions_df, new_position], ignore_index=True)

                print(f"  Opening LONG position: Entry=${price:.2f}, Qty={order_qty:.4f}")

    elif predicted_class == 0:  # Sell signal - close longs exponentially OR open short position
        if not open_longs.empty:
            # Calculate how many longs to close exponentially
            close_count = calculate_exponential_close_count(net_long_positions, base_close=1, exp_factor=exp_factor)
            close_count = min(close_count, len(open_longs))  # Don't exceed available positions

            print(f"  Exponential closing: {close_count} LONG positions (net longs: {net_long_positions})")

            # Close multiple long positions (FIFO - oldest first)
            longs_to_close = open_longs.head(close_count)
            total_pnl = 0

            for idx, pos in longs_to_close.iterrows():
                entry_price = pos['entry_price']
                order_qty = pos['order_qty']

                # Calculate PnL for long position
                pnl = (price - entry_price) * order_qty
                total_pnl += pnl

                # Update position
                positions_df.loc[idx, 'exit_date'] = current_datetime
                positions_df.loc[idx, 'exit_price'] = price
                positions_df.loc[idx, 'pnl'] = pnl

                print(
                    f"    Closing LONG: Entry=${entry_price:.2f}, Exit=${price:.2f}, Qty={order_qty:.4f}, PnL=${pnl:.2f}")

            # Update balance
            current_balance += total_pnl
            print(f"  Total PnL from {close_count} LONG closures: ${total_pnl:.2f}")

        else:
            # No longs to close, open new short position if under limit
            total_open_positions = len(positions_df[positions_df['exit_date'].isna()])
            if total_open_positions < max_pos_count:
                order_qty = fixed_trade_size / price
                new_position = pd.DataFrame({
                    'entry_date': [current_datetime],
                    'entry_price': [price],
                    'symbol': [symbol],
                    'order_qty': [order_qty],
                    'position_type': ['short'],
                    'exit_date': [pd.NaT],
                    'exit_price': [np.nan],
                    'pnl': [np.nan],
                    'stop_price': [np.nan],
                    'take_price': [np.nan]
                })
                if len(positions_df) == 0:
                    positions_df = new_position
                else:
                    positions_df = pd.concat([positions_df, new_position], ignore_index=True)

                print(f"  Opening SHORT position: Entry=${price:.2f}, Qty={order_qty:.4f}")

    # Debug output for signal processing
    if predicted_class in [0, 2]:
        # Recalculate after operations
        updated_open = positions_df[positions_df['exit_date'].isna()]
        updated_longs = len(updated_open[updated_open['position_type'] == 'long'])
        updated_shorts = len(updated_open[updated_open['position_type'] == 'short'])
        net_position = updated_longs - updated_shorts

        action = {0: 'SELL (close long/open short)', 2: 'BUY (close short/open long)'}[predicted_class]
        print(
            f"Signal {predicted_class} ({action}) at {current_datetime.strftime('%Y-%m-%d %H:%M')} | Total: {len(updated_open)} (L:{updated_longs}, S:{updated_shorts}, Net:{net_position:+d})")

    return positions_df, current_balance


def handle_sl_tp_trading(predicted_class, price, current_datetime, symbol, fixed_trade_size,
                         positions_df, current_balance, max_pos_count, stop_loss_pct, take_profit_pct):
    """Handle SL/TP trading logic - only open positions on signals, close only on SL/TP"""
    if predicted_class == 2:  # Buy signal - open long position only
        open_positions = len(positions_df[positions_df['exit_date'].isna()])
        if open_positions < max_pos_count:
            order_qty = fixed_trade_size / price

            # LONG
            stop_price = price * (1 - stop_loss_pct)
            take_price = price * (1 + take_profit_pct)
            initial_risk = price - stop_price

            new_position = pd.DataFrame({
                'entry_date': [current_datetime],
                'entry_price': [price],
                'symbol': [symbol],
                'order_qty': [order_qty],
                'position_type': ['long'],
                'exit_date': [pd.NaT],
                'exit_price': [np.nan],
                'pnl': [np.nan],
                'stop_price': [stop_price],
                'take_price': [take_price],
                'initial_stop': [stop_price],
                'initial_take': [take_price],
                'initial_risk': [initial_risk]
            })

            if len(positions_df) == 0:
                positions_df = new_position
            else:
                positions_df = pd.concat([positions_df, new_position], ignore_index=True)

            print(
                f"  Opening LONG: Entry=${price:.2f}, SL=${stop_price:.2f}, TP=${take_price:.2f}, Qty={order_qty:.4f}")

    elif predicted_class == 0:  # Sell signal - open short position only
        open_positions = len(positions_df[positions_df['exit_date'].isna()])
        if open_positions < max_pos_count:
            order_qty = fixed_trade_size / price

            # Calculate SL/TP levels
            stop_price = price * (1 + stop_loss_pct)
            take_price = price * (1 - take_profit_pct)
            initial_risk = stop_price - price

            new_position = pd.DataFrame({
                'entry_date': [current_datetime],
                'entry_price': [price],
                'symbol': [symbol],
                'order_qty': [order_qty],
                'position_type': ['short'],
                'exit_date': [pd.NaT],
                'exit_price': [np.nan],
                'pnl': [np.nan],
                'stop_price': [stop_price],
                'take_price': [take_price],
                'initial_stop': [stop_price],
                'initial_take': [take_price],
                'initial_risk': [initial_risk]
            })

            if len(positions_df) == 0:
                positions_df = new_position
            else:
                positions_df = pd.concat([positions_df, new_position], ignore_index=True)

            print(
                f"  Opening SHORT: Entry=${price:.2f}, SL=${stop_price:.2f}, TP=${take_price:.2f}, Qty={order_qty:.4f}")

    # Debug output
    if predicted_class in [0, 2]:
        action = {0: 'OPEN SHORT', 2: 'OPEN LONG'}[predicted_class]
        open_count = len(positions_df[positions_df['exit_date'].isna()])
        open_longs = len(positions_df[(positions_df['exit_date'].isna()) & (positions_df['position_type'] == 'long')])
        open_shorts = len(positions_df[(positions_df['exit_date'].isna()) & (positions_df['position_type'] == 'short')])
        print(
            f"Signal {predicted_class} ({action}) at {current_datetime.strftime('%Y-%m-%d %H:%M')} | Total: {open_count} (L:{open_longs}, S:{open_shorts})")

    return positions_df, current_balance


def create_pnl_visualization(positions_df, init_deposit, final_balance):
    """Create comprehensive PnL visualization with cumulative position tracking"""
    if len(positions_df) == 0:
        print("No trades to visualize")
        return

    # Filter completed trades
    completed_trades = positions_df[positions_df['pnl'].notna()].copy()
    all_positions = positions_df.copy()

    if len(completed_trades) == 0:
        print("No completed trades to visualize")
        return

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Signal-Based Trading Performance Analysis', fontsize=16, fontweight='bold')

    # 1. Cumulative Position Size Over Time
    all_positions['entry_date'] = pd.to_datetime(all_positions['entry_date'])
    all_positions = all_positions.sort_values('entry_date')

    # Calculate cumulative position size
    position_timeline = []
    cumulative_qty = 0

    for _, row in all_positions.iterrows():
        if pd.notna(row['exit_date']):
            # Position opened
            position_timeline.append({
                'date': row['entry_date'],
                'qty_change': row['order_qty'],
                'action': 'open'
            })
            # Position closed
            position_timeline.append({
                'date': pd.to_datetime(row['exit_date']),
                'qty_change': -row['order_qty'],
                'action': 'close'
            })
        else:
            # Still open position
            position_timeline.append({
                'date': row['entry_date'],
                'qty_change': row['order_qty'],
                'action': 'open'
            })

    # Sort by date and calculate cumulative position
    position_timeline = sorted(position_timeline, key=lambda x: x['date'])
    dates = []
    cumulative_positions = []
    cumulative_qty = 0

    for event in position_timeline:
        cumulative_qty += event['qty_change']
        dates.append(event['date'])
        cumulative_positions.append(cumulative_qty)

    if dates:
        ax1.plot(dates, cumulative_positions, linewidth=2, marker='o', markersize=3, label='Cumulative Position Size')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Cumulative Position Size Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Position Size (Qty)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. Cumulative PnL over time
    completed_trades['exit_date'] = pd.to_datetime(completed_trades['exit_date'])
    completed_trades = completed_trades.sort_values('exit_date')
    completed_trades['cumulative_pnl'] = completed_trades['pnl'].cumsum()
    completed_trades['cumulative_balance'] = init_deposit + completed_trades['cumulative_pnl']

    ax2.plot(completed_trades['exit_date'], completed_trades['cumulative_pnl'],
             linewidth=2, marker='o', markersize=4, label='Cumulative PnL', color='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_title('Cumulative PnL Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative PnL ($)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 3. Balance Evolution
    ax3.plot(completed_trades['exit_date'], completed_trades['cumulative_balance'],
             linewidth=2, color='blue', marker='s', markersize=4)
    ax3.axhline(y=init_deposit, color='orange', linestyle='--', alpha=0.7, label=f'Initial: ${init_deposit}')
    ax3.set_title('Account Balance Evolution')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Balance ($)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Format x-axis dates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # 4. PnL Statistics
    ax4.axis('off')

    # Calculate statistics
    total_trades = len(completed_trades)
    winning_trades = len(completed_trades[completed_trades['pnl'] > 0])
    losing_trades = len(completed_trades[completed_trades['pnl'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    avg_win = completed_trades[completed_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = completed_trades[completed_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

    total_pnl = completed_trades['pnl'].sum()
    max_win = completed_trades['pnl'].max()
    max_loss = completed_trades['pnl'].min()

    profit_factor = abs(
        avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')

    # Create statistics text
    stats_text = f"""
    TRADING STATISTICS

    Total Trades: {total_trades}
    Winning Trades: {winning_trades}
    Losing Trades: {losing_trades}
    Win Rate: {win_rate:.1f}%

    Total PnL: ${total_pnl:.2f}
    Average Win: ${avg_win:.2f}
    Average Loss: ${avg_loss:.2f}

    Best Trade: ${max_win:.2f}
    Worst Trade: ${max_loss:.2f}

    Profit Factor: {profit_factor:.2f}

    Initial Balance: ${init_deposit:.2f}
    Final Balance: ${final_balance:.2f}
    Return: {((final_balance - init_deposit) / init_deposit * 100):.2f}%
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    plt.savefig('pnl_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 PnL visualization saved as 'pnl_analysis.png'")

    # Show the plot
    plt.show()

    return fig


def print_trading_statistics(positions_df, init_deposit):
    """
    Print comprehensive trading statistics to terminal
    """
    completed_trades = positions_df[pd.notna(positions_df['pnl'])]

    if completed_trades.empty:
        print("\n=== No completed trades to analyze ===")
        return

    total_trades = len(completed_trades)
    winning_trades = len(completed_trades[completed_trades['pnl'] > 0])
    losing_trades = len(completed_trades[completed_trades['pnl'] < 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    avg_win = completed_trades[completed_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = completed_trades[completed_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

    profit_factor = abs(
        avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else float('inf')

    total_pnl = completed_trades['pnl'].sum()
    completed_trades = completed_trades.copy()
    completed_trades['cumulative_pnl'] = completed_trades['pnl'].cumsum()
    max_drawdown = completed_trades['cumulative_pnl'].cummin().min() if not completed_trades.empty else 0

    return_pct = (total_pnl / init_deposit) * 100

    print("\n" + "=" * 60)
    print("                    TRADING STATISTICS")
    print("=" * 60)
    print(f"Total Trades:        {total_trades}")
    print(f"Winning Trades:      {winning_trades}")
    print(f"Losing Trades:       {losing_trades}")
    print(f"Win Rate:            {win_rate:.1f}%")
    print("-" * 60)
    print(f"Average Win:         ${avg_win:.2f}")
    print(f"Average Loss:        ${avg_loss:.2f}")
    print(f"Profit Factor:       {profit_factor:.2f}")
    print("-" * 60)
    print(f"Total PnL:           ${total_pnl:.2f}")
    print(f"Max Drawdown:        ${max_drawdown:.2f}")
    print(f"Initial Deposit:     ${init_deposit:.2f}")
    print(f"Final Balance:       ${init_deposit + total_pnl:.2f}")
    print(f"Return:              {return_pct:.2f}%")
    print("=" * 60)


def create_plotly_chart(positions_df, balance_history, price_data, init_deposit, symbol, trail_history=None):
    """
    Create interactive Plotly chart with PnL %, position size, and price
    """
    if positions_df.empty and not balance_history:
        print("No data to visualize")
        return

    # Create subplots with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} Price & Position Size', 'PnL %'),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Prepare price data
    if price_data is not None and not price_data.empty:
        fig.add_trace(
            go.Scatter(
                x=price_data['Date'],
                y=price_data['Close'],
                name='Price',
                line=dict(color='blue', width=1),
                yaxis='y'
            ),
            row=1, col=1
        )

    # Calculate position timeline with long/short distinction
    position_timeline = []

    for _, row in positions_df.iterrows():
        if pd.notna(row['entry_date']):
            # +1 for long, -1 for short
            position_change = 1 if row['position_type'] == 'long' else -1
            position_timeline.append({
                'date': row['entry_date'],
                'position_change': position_change
            })

        if pd.notna(row['exit_date']) and pd.notna(row['pnl']):
            # Opposite of entry for closing
            position_change = -1 if row['position_type'] == 'long' else 1
            position_timeline.append({
                'date': row['exit_date'],
                'position_change': position_change
            })

    if position_timeline:
        timeline_df = pd.DataFrame(position_timeline)
        timeline_df = timeline_df.sort_values('date')
        timeline_df['cumulative_positions'] = timeline_df['position_change'].cumsum()

        fig.add_trace(
            go.Scatter(
                x=timeline_df['date'],
                y=timeline_df['cumulative_positions'],
                name='Position Size',
                line=dict(color='orange', width=2),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )

    # Add entry markers
    entries = positions_df[pd.notna(positions_df['entry_date'])]
    for _, row in entries.iterrows():
        color = 'green' if row['position_type'] == 'long' else 'red'
        marker_symbol = 'triangle-up' if row['position_type'] == 'long' else 'triangle-down'
        fig.add_trace(
            go.Scatter(
                x=[row['entry_date']],
                y=[row['entry_price']],
                mode='markers',
                marker=dict(symbol=marker_symbol, color=color, size=10),
                name=f'{row["position_type"].capitalize()} Entry #{_}',
                showlegend=False
            ),
            row=1, col=1
        )

    # Add exit markers
    exits = positions_df[pd.notna(positions_df['exit_date'])]
    for _, row in exits.iterrows():
        pnl_color = 'green' if row['pnl'] > 0 else 'red'
        fig.add_trace(
            go.Scatter(
                x=[row['exit_date']],
                y=[row['exit_price']],
                mode='markers',
                marker=dict(symbol='x', color=pnl_color, size=10),
                name=f'Exit #{_} (PnL: {row["pnl"]:.2f})',
                showlegend=False
            ),
            row=1, col=1
        )

    # Add trailing levels if available (only in test mode)
    if trail_history:
        for pos_idx, history in trail_history.items():
            if not history:
                continue
            hist_df = pd.DataFrame(history)
            pos_type = positions_df.loc[pos_idx, 'position_type']
            color = 'green' if pos_type == 'long' else 'red'
            fig.add_trace(
                go.Scatter(
                    x=hist_df['date'],
                    y=hist_df['sl'],
                    name=f'{pos_type.capitalize()} SL #{pos_idx}',
                    line=dict(color=color, width=1, dash='dot'),
                    opacity=0.5,
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=hist_df['date'],
                    y=hist_df['tp'],
                    name=f'{pos_type.capitalize()} TP #{pos_idx}',
                    line=dict(color=color, width=1, dash='dash'),
                    opacity=0.5,
                    showlegend=False
                ),
                row=1, col=1
            )

    # Calculate PnL percentage
    if balance_history:
        balance_df = pd.DataFrame(balance_history)
        balance_df['realized_pnl_pct'] = (balance_df['realized_pnl'] / init_deposit) * 100
        balance_df['total_pnl_pct'] = (balance_df['total_pnl'] / init_deposit) * 100

        # Add realized PnL line
        fig.add_trace(
            go.Scatter(
                x=balance_df['date'],
                y=balance_df['realized_pnl_pct'],
                name='Realized PnL %',
                line=dict(color='green', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )

        # Add total PnL (realized + unrealized) line
        fig.add_trace(
            go.Scatter(
                x=balance_df['date'],
                y=balance_df['total_pnl_pct'],
                name='Total PnL % (Real + Unreal)',
                line=dict(color='blue', width=2, dash='dot'),
                fill='tonexty'
            ),
            row=2, col=1
        )

        # Add zero line for PnL
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f'{symbol} Trading Strategy Analysis',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Position Size", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="PnL (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    # Save and show
    filename = 'trading_analysis.html'
    fig.write_html(filename)
    print(f"\nInteractive chart saved as {filename}")
    fig.show()


def main(test_mode=False):
    # Load price data
    price_file = "data/SOLUSDT_1m_20210101_to_20250820.csv"
    signals_file = "data/sol_new.csv"

    print("Loading data...")
    df = pd.read_csv(price_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date").reset_index(drop=True)

    # Filter data to match signal timeframe (signals start from 2025-03-04)
    signals_start_date = pd.to_datetime('2025-04-10')
    df = df[df['Date'] >= signals_start_date].reset_index(drop=True)

    print(f"Price data loaded: {len(df)} rows from {df['Date'].min()} to {df['Date'].max()}")

    # Load signals once at startup (major optimization)
    print("Loading signals data...")
    signals_dict = load_signals_data(signals_file)
    print(f"Signals loaded: {len(signals_dict)} signals")

    # Debug: Check signal distribution
    signal_values = list(signals_dict.values())
    signal_counts = pd.Series(signal_values).value_counts().to_dict()
    print(f"Signal distribution: {signal_counts}")

    if test_mode:
        print("\n=== TEST MODE: Running on first 1000 candles ===\n")
        df = df.head(1000)

    init_deposit = 100_000
    balance = init_deposit

    max_pos_count = 400  # Maximum positions (defines leverage: 200 pos * $1000 = $200k max exposure on $100k)
    fixed_trade_size = 1000  # Fixed $1000 per trade
    pctOfReinvest = 0.0  # No reinvestment

    # Trading mode selection
    trading_mode = "sl_tp"  # Options: "signal_based" or "sl_tp"

    # Stop-loss and take-profit percentages (only used in sl_tp mode)
    stop_loss_pct = 0.02  # 2% stop loss
    take_profit_pct = 0.05  # 8% take profit

    # Exponential closing factor (only used in signal_based mode)
    # Controls how aggressively positions are closed based on net position size
    # Higher values = more aggressive closing, Lower values = more conservative
    # Examples: 0.01 (conservative), 0.05 (moderate), 0.1 (aggressive)
    exp_factor = 0.01  # Exponential scaling factor for position closing
    state = {"OCO": []}
    positions_df = pd.DataFrame(columns=[
        'entry_date', 'entry_price', 'symbol', 'order_qty', 'position_type',
        'exit_date', 'exit_price', 'pnl', 'stop_price', 'take_price',
        'initial_stop', 'initial_take', 'initial_risk'
    ])

    # Trail history for visualization (only in test_mode to avoid memory issues)
    trail_history = {} if test_mode else None

    # Start backtest from a reasonable point (need some history for ATR calculation)
    start_idx = max(300, 0)
    end_idx = len(df)

    print(f"Starting backtest from index {start_idx} to {end_idx} ({end_idx - start_idx} iterations)")

    # Check if we have buy signals (class 2) in our timeframe
    buy_signals_count = sum(1 for v in signal_counts.values() if v == 2)
    if signal_counts.get(2, 0) == 0:
        print("⚠️ WARNING: No class 2 (buy) signals found in data!")
    else:
        print(f"✓ Found {signal_counts.get(2, 0)} buy signals (class 2)")

    # Pre-convert data to numpy arrays for faster access
    dates = df['Date'].values
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    opens = df['Open'].values
    volumes = df['Volume'].values
    # Main backtest loop - get signal on candle i-1, execute on candle i
    previous_signal = None

    for i in tqdm(range(start_idx, len(df)), desc="Backtest Progress", unit="candle"):
        # Get signal from previous candle (realistic timing)
        if i > 0:
            previous_datetime = pd.to_datetime(df.iloc[i - 1]['Date'])
            previous_signal = get_signal_for_datetime(previous_datetime, signals_dict)

        # Execute trades based on previous candle's signal on current candle
        positions_df, state, balance = ml_strategy_torch(
            Bars_df=df.iloc[max(0, i - 300):i + 1],
            current_date=df.iloc[i]['Date'],
            symbol="SOLUSDT",
            series_coin=df.iloc[max(0, i - 1):i + 1]['Close'],
            fixed_trade_size=fixed_trade_size,
            pctOfReinvest=pctOfReinvest,
            signals_dict=signals_dict,
            max_pos_count=max_pos_count,
            strategy_name='ml_signals',
            state=deepcopy(state),
            positions_df=positions_df,
            current_balance=balance,
            trading_mode=trading_mode,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            exp_factor=exp_factor,
            override_signal=previous_signal  # Pass signal from previous candle
        )

        # Log trailing levels for visualization (only in test_mode)
        if test_mode and trail_history is not None:
            current_date = df.iloc[i]['Date']
            open_pos = positions_df[positions_df['exit_date'].isna()]
            for idx, pos in open_pos.iterrows():
                trail_history.setdefault(idx, []).append({
                    'date': current_date,
                    'sl': pos['stop_price'],
                    'tp': pos['take_price']
                })

    print("==== Backtest result ====")
    print(f"Start deposit: {init_deposit}")
    print(f"Final balance: {balance}")
    closed_trades = positions_df[positions_df['pnl'].notna()]['pnl'].tolist()
    print(f"Trades: {len(closed_trades)}")
    if closed_trades:
        print(
            f"Avg PnL: {np.mean(closed_trades):.2f}, Win rate: {np.mean([1 if t > 0 else 0 for t in closed_trades]):.2%}")
    print("\nPositions DataFrame:")
    print(positions_df)

    # Print statistics to terminal
    print_trading_statistics(positions_df, init_deposit)

    # Create balance history with both realized and unrealized PnL
    balance_history = []
    current_balance_tracking = init_deposit

    # Get all unique dates from the dataset
    all_dates = df['Date'].sort_values().unique()

    for date in all_dates:
        current_price = df[df['Date'] == date]['Close'].iloc[0]

        # Calculate realized PnL up to this date
        realized_trades = positions_df[(pd.notna(positions_df['exit_date'])) & (positions_df['exit_date'] <= date)]
        realized_pnl = realized_trades['pnl'].sum() if not realized_trades.empty else 0

        # Calculate unrealized PnL for open positions at this date
        open_positions_at_date = positions_df[
            (pd.notna(positions_df['entry_date'])) &
            (positions_df['entry_date'] <= date) &
            ((pd.isna(positions_df['exit_date'])) | (positions_df['exit_date'] > date))
            ]

        unrealized_pnl = 0
        for _, pos in open_positions_at_date.iterrows():
            if pos['position_type'] == 'long':
                unrealized_pnl += (current_price - pos['entry_price']) * pos['order_qty']
            else:  # short
                unrealized_pnl += (pos['entry_price'] - current_price) * pos['order_qty']

        total_pnl = realized_pnl + unrealized_pnl

        balance_history.append({
            'date': date,
            'balance': init_deposit + total_pnl,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl
        })

    # Create interactive Plotly chart
    if not positions_df.empty or balance_history:
        print("\nGenerating interactive chart...")
        create_plotly_chart(positions_df, balance_history, df, init_deposit, "SOLUSDT", trail_history=trail_history)
    else:
        print("\nNo data to visualize")


if __name__ == "__main__":
    import sys

    test_mode = len(sys.argv) > 1 and sys.argv[1] == '--test'
    main(test_mode=test_mode)