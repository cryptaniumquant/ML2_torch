import pandas as pd
import numpy as np
from strategies_backtest_trailing import load_signals_data, get_signal_for_datetime, ml_strategy_torch
from copy import deepcopy
from tqdm import tqdm

def export_pnl_data_only():
    """Export only PnL data from backtest to CSV"""
    
    # Load price data
    price_file = "data/BTCUSDT_1m_20210106_to_20250825.csv"
    signals_file = "data/btc_new.csv"

    print("Loading data...")
    df = pd.read_csv(price_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date").reset_index(drop=True)

    # Filter data to match signal timeframe
    signals_start_date = pd.to_datetime('2025-03-10')
    df = df[df['Date'] >= signals_start_date].reset_index(drop=True)

    print(f"Price data loaded: {len(df)} rows from {df['Date'].min()} to {df['Date'].max()}")

    # Load signals
    print("Loading signals data...")
    signals_dict = load_signals_data(signals_file)
    print(f"Signals loaded: {len(signals_dict)} signals")

    # Backtest parameters
    init_deposit = 100_000
    balance = init_deposit
    max_pos_count = 1000
    fixed_trade_size = 1000
    pctOfReinvest = 0.0

    # ATR parameters
    trail_mode = 'atr_band'
    atr_period_minutes = 140
    sl_atr_mult = 30
    trail_on_new_signal = True
    
    state = {"OCO": []}
    positions_df = pd.DataFrame(columns=[
        'entry_date', 'entry_price', 'symbol', 'order_qty', 'position_type',
        'exit_date', 'exit_price', 'pnl', 'stop_price', 'take_price',
        'initial_stop', 'initial_take', 'initial_risk'
    ])

    # Run backtest with PnL tracking
    start_idx = max(300, 0)
    previous_signal = None
    
    # Store balance history for each candle
    balance_history = []
    
    print(f"Running backtest from index {start_idx} to {len(df)}...")
    
    for i in tqdm(range(start_idx, len(df)), desc="Backtest Progress", unit="candle"):
        # Get signal from previous candle
        if i > 0:
            previous_datetime = pd.to_datetime(df.iloc[i - 1]['Date'])
            previous_signal = get_signal_for_datetime(previous_datetime, signals_dict)

        # Execute trades
        positions_df, state, balance = ml_strategy_torch(
            Bars_df=df.iloc[max(0, i - 1500):i + 1],
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
            override_signal=previous_signal,
            trail_mode=trail_mode,
            atr_period_minutes=atr_period_minutes,
            sl_atr_mult=sl_atr_mult,
            trail_on_new_signal=trail_on_new_signal
        )
        
        # Calculate unrealized PnL for current candle
        current_price = df.iloc[i]['Close']
        current_date = df.iloc[i]['Date']
        
        # Calculate realized PnL up to this date
        realized_trades = positions_df[(pd.notna(positions_df['exit_date'])) & (positions_df['exit_date'] <= current_date)]
        realized_pnl = realized_trades['pnl'].sum() if not realized_trades.empty else 0

        # Calculate unrealized PnL for open positions
        open_positions_at_date = positions_df[
            (pd.notna(positions_df['entry_date'])) &
            (positions_df['entry_date'] <= current_date) &
            ((pd.isna(positions_df['exit_date'])) | (positions_df['exit_date'] > current_date))
        ]

        unrealized_pnl = 0
        commission_rate = 0.00018
        for _, pos in open_positions_at_date.iterrows():
            exit_commission = current_price * pos['order_qty'] * commission_rate
            if pos['position_type'] == 'long':
                unrealized_pnl += (current_price - pos['entry_price']) * pos['order_qty'] - exit_commission
            else:  # short
                unrealized_pnl += (pos['entry_price'] - current_price) * pos['order_qty'] - exit_commission

        total_pnl = realized_pnl + unrealized_pnl
        
        # Store balance data
        balance_history.append({
            'date': current_date,
            'price': current_price,
            'balance': init_deposit + total_pnl,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / init_deposit) * 100
        })

    # Convert to DataFrame and export to CSV
    balance_df = pd.DataFrame(balance_history)
    
    # Export to CSV
    csv_filename = 'pnl_data_for_savgol.csv'
    balance_df.to_csv(csv_filename, index=False)
    
    print(f"\nPnL data exported to {csv_filename}")
    print(f"Generated {len(balance_df)} data points")
    print(f"Date range: {balance_df['date'].min()} to {balance_df['date'].max()}")
    print(f"Final Total PnL: ${balance_df['total_pnl'].iloc[-1]:.2f}")
    print(f"Total PnL Range: ${balance_df['total_pnl'].min():.2f} to ${balance_df['total_pnl'].max():.2f}")
    
    return csv_filename

if __name__ == "__main__":
    export_pnl_data_only()
