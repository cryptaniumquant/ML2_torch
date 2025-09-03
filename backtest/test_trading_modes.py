#!/usr/bin/env python3
"""
Test script to demonstrate flexible trading modes with SL/TP and signal-based logic.
This script shows how to use both trading modes with the enhanced backtest system.
"""

import pandas as pd
import numpy as np
from strategies_backtest_fifo import (
    load_signals_data, 
    ml_strategy_torch,
    create_pnl_visualization
)

def run_trading_mode_test():
    """Test both signal-based and SL/TP trading modes"""
    
    # Load sample data (using SOLUSDT as example)
    print("Loading market data...")
    data_file = "data/SOLUSDT_1m_20210101_to_20250820.csv"
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Use a smaller subset for testing (last 10000 rows)
    df = df.tail(10000).reset_index(drop=True)
    print(f"Using {len(df)} data points for testing")
    
    # Load signals data
    signals_file = "test_signals.csv"  # You'll need to create this or use existing signals
    try:
        signals_dict = load_signals_data(signals_file)
        print(f"Loaded {len(signals_dict)} signals")
    except FileNotFoundError:
        print(f"Signals file {signals_file} not found. Creating sample signals...")
        # Create sample signals for testing
        sample_signals = []
        for i in range(0, len(df), 100):  # Signal every 100 bars
            signal_class = np.random.choice([0, 1, 2])  # Random signals for testing
            sample_signals.append({
                'date': df.iloc[i]['Date'],
                'predicted': signal_class
            })
        
        signals_df = pd.DataFrame(sample_signals)
        signals_df.to_csv(signals_file, index=False)
        signals_dict = load_signals_data(signals_file)
        print(f"Created and loaded {len(signals_dict)} sample signals")
    
    # Test parameters
    symbol = "SOLUSDT"
    fixed_trade_size = 1000  # $1000 per trade
    max_pos_count = 10  # Max 10 positions
    init_balance = 100000  # $100k starting capital
    
    # Test 1: Signal-based trading mode
    print("\n" + "="*50)
    print("TESTING SIGNAL-BASED TRADING MODE")
    print("="*50)
    
    test_signal_based_mode(df, signals_dict, symbol, fixed_trade_size, max_pos_count, init_balance)
    
    # Test 2: SL/TP trading mode
    print("\n" + "="*50)
    print("TESTING SL/TP TRADING MODE")
    print("="*50)
    
    test_sl_tp_mode(df, signals_dict, symbol, fixed_trade_size, max_pos_count, init_balance)

def test_signal_based_mode(df, signals_dict, symbol, fixed_trade_size, max_pos_count, init_balance):
    """Test signal-based trading mode"""
    
    # Initialize positions DataFrame
    positions_df = pd.DataFrame(columns=[
        'entry_date', 'entry_price', 'symbol', 'order_qty', 'position_type',
        'exit_date', 'exit_price', 'pnl'
    ])
    
    current_balance = init_balance
    state = {}
    
    # Run backtest for a subset of data (first 1000 bars)
    test_data = df.head(1000)
    
    print(f"Running signal-based backtest on {len(test_data)} bars...")
    
    for i in range(100, len(test_data)):  # Start after 100 bars for indicators
        current_bars = test_data.iloc[:i+1]
        current_date = current_bars['Date'].iloc[-1]
        series_coin = current_bars['Close']
        
        positions_df, state, current_balance = ml_strategy_torch(
            Bars_df=current_bars,
            current_date=current_date,
            symbol=symbol,
            series_coin=series_coin,
            fixed_trade_size=fixed_trade_size,
            pctOfReinvest=0.0,
            signals_dict=signals_dict,
            max_pos_count=max_pos_count,
            strategy_name="signal_based_test",
            state=state,
            positions_df=positions_df,
            current_balance=current_balance,
            trading_mode="signal_based"
        )
    
    # Print results
    total_trades = len(positions_df[positions_df['pnl'].notna()])
    total_pnl = positions_df['pnl'].sum()
    open_positions = len(positions_df[positions_df['exit_date'].isna()])
    
    print(f"Signal-based mode results:")
    print(f"  Total completed trades: {total_trades}")
    print(f"  Total realized PnL: ${total_pnl:.2f}")
    print(f"  Open positions: {open_positions}")
    print(f"  Final balance: ${current_balance:.2f}")
    print(f"  Return: {((current_balance - init_balance) / init_balance * 100):.2f}%")

def test_sl_tp_mode(df, signals_dict, symbol, fixed_trade_size, max_pos_count, init_balance):
    """Test SL/TP trading mode"""
    
    # Initialize positions DataFrame
    positions_df = pd.DataFrame(columns=[
        'entry_date', 'entry_price', 'symbol', 'order_qty', 'position_type',
        'exit_date', 'exit_price', 'pnl'
    ])
    
    current_balance = init_balance
    state = {}
    
    # SL/TP parameters
    stop_loss_pct = 0.02  # 2% stop loss
    take_profit_pct = 0.04  # 4% take profit
    
    # Run backtest for a subset of data (first 1000 bars)
    test_data = df.head(1000)
    
    print(f"Running SL/TP backtest on {len(test_data)} bars...")
    print(f"Stop Loss: {stop_loss_pct*100}%, Take Profit: {take_profit_pct*100}%")
    
    for i in range(100, len(test_data)):  # Start after 100 bars for indicators
        current_bars = test_data.iloc[:i+1]
        current_date = current_bars['Date'].iloc[-1]
        series_coin = current_bars['Close']
        
        positions_df, state, current_balance = ml_strategy_torch(
            Bars_df=current_bars,
            current_date=current_date,
            symbol=symbol,
            series_coin=series_coin,
            fixed_trade_size=fixed_trade_size,
            pctOfReinvest=0.0,
            signals_dict=signals_dict,
            max_pos_count=max_pos_count,
            strategy_name="sl_tp_test",
            state=state,
            positions_df=positions_df,
            current_balance=current_balance,
            trading_mode="sl_tp",
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
    
    # Print results
    total_trades = len(positions_df[positions_df['pnl'].notna()])
    total_pnl = positions_df['pnl'].sum()
    open_positions = len(positions_df[positions_df['exit_date'].isna()])
    
    # Analyze exit reasons
    completed_trades = positions_df[positions_df['pnl'].notna()]
    if len(completed_trades) > 0:
        winning_trades = len(completed_trades[completed_trades['pnl'] > 0])
        losing_trades = len(completed_trades[completed_trades['pnl'] < 0])
        win_rate = winning_trades / len(completed_trades) * 100
    else:
        winning_trades = losing_trades = win_rate = 0
    
    print(f"SL/TP mode results:")
    print(f"  Total completed trades: {total_trades}")
    print(f"  Winning trades: {winning_trades}")
    print(f"  Losing trades: {losing_trades}")
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  Total realized PnL: ${total_pnl:.2f}")
    print(f"  Open positions: {open_positions}")
    print(f"  Final balance: ${current_balance:.2f}")
    print(f"  Return: {((current_balance - init_balance) / init_balance * 100):.2f}%")

if __name__ == "__main__":
    print("Flexible Trading Modes Test")
    print("Testing both signal-based and SL/TP trading modes...")
    
    try:
        run_trading_mode_test()
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
