#!/usr/bin/env python3

import pandas as pd
import numpy as np
from strategies_backtest_sltp import load_signals_data, get_signal_for_datetime

def test_signal_loading():
    """Test signal loading and matching functionality"""
    
    # Test signal loading
    signals_file = "data/sol_new.csv"
    try:
        signals_df = load_signals_data(signals_file)
        print(f"✓ Successfully loaded {len(signals_df)} signals")
        print(f"  Date range: {signals_df['date'].min()} to {signals_df['date'].max()}")
        print(f"  Sample signal: {signals_df.iloc[0][['date', 'predicted', 'prob_class_1']].to_dict()}")
    except Exception as e:
        print(f"✗ Error loading signals: {e}")
        return False
    
    # Test signal matching
    test_datetime = signals_df['date'].iloc[100]  # Pick a datetime that should exist
    try:
        predicted_class = get_signal_for_datetime(test_datetime, signals_df)
        print(f"✓ Signal matching works")
        print(f"  For {test_datetime}: class={predicted_class}")
    except Exception as e:
        print(f"✗ Error matching signals: {e}")
        return False
    
    # Test with non-existent datetime
    test_datetime_missing = pd.to_datetime('2020-01-01 12:00:00')
    predicted_class = get_signal_for_datetime(test_datetime_missing, signals_df)
    if predicted_class is None:
        print(f"✓ Correctly handles missing signals")
    else:
        print(f"✗ Should return None for missing signals")
        return False
    
    return True

def test_price_data():
    """Test price data loading"""
    
    price_file = "data/SOLUSDT_1m_20210101_to_20250820.csv"
    try:
        df = pd.read_csv(price_file)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"✓ Successfully loaded {len(df)} price records")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Filter to signal timeframe
        signals_start_date = pd.to_datetime('2025-03-04')
        filtered_df = df[df['Date'] >= signals_start_date]
        print(f"  Filtered to signal timeframe: {len(filtered_df)} records")
        
        return True
    except Exception as e:
        print(f"✗ Error loading price data: {e}")
        return True

def test_timestamp_alignment():
    """Test if signal timestamps align with price data timestamps"""
    
    # Load both datasets
    price_file = "data/SOLUSDT_1m_20210101_to_20250820.csv"
    signals_file = "data/sol_new.csv"
    
    try:
        # Load price data
        price_df = pd.read_csv(price_file)
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        
        # Load signals data
        signals_df = pd.read_csv(signals_file)
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        
        print(f"✓ Loaded price data: {len(price_df)} records")
        print(f"✓ Loaded signals data: {len(signals_df)} records")
        
        # Filter price data to signal timeframe
        signals_start = signals_df['date'].min()
        signals_end = signals_df['date'].max()
        filtered_price_df = price_df[(price_df['Date'] >= signals_start) & (price_df['Date'] <= signals_end)]
        
        print(f"  Signal timeframe: {signals_start} to {signals_end}")
        print(f"  Price data in signal timeframe: {len(filtered_price_df)} records")
        
        # Check timestamp alignment
        signal_times = set(signals_df['date'])
        price_times = set(filtered_price_df['Date'])
        
        # Find matches
        matching_times = signal_times.intersection(price_times)
        missing_in_price = signal_times - price_times
        
        print(f"  Matching timestamps: {len(matching_times)} out of {len(signal_times)} signals")
        print(f"  Coverage: {len(matching_times)/len(signal_times)*100:.1f}%")
        
        if missing_in_price:
            print(f"  ⚠️ Missing in price data: {len(missing_in_price)} timestamps")
            # Show first few missing timestamps
            missing_sample = sorted(list(missing_in_price))[:5]
            print(f"    Sample missing: {missing_sample}")
        
        # Test signal matching function with actual data
        test_signal_time = signals_df['date'].iloc[0]
        predicted_class = get_signal_for_datetime(test_signal_time, signals_df)
        
        if predicted_class is not None:
            print(f"✓ Signal matching function works with real data")
            print(f"  Test signal at {test_signal_time}: class={predicted_class}")
        else:
            print(f"✗ Signal matching function failed with real data")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing timestamp alignment: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Signal-Based Backtest Components ===\n")
    
    success = True
    
    print("1. Testing signal loading...")
    success &= test_signal_loading()
    print()
    
    print("2. Testing price data loading...")
    success &= test_price_data()
    print()
    
    print("3. Testing timestamp alignment...")
    success &= test_timestamp_alignment()
    print()
    
    if success:
        print("✓ All tests passed! The backtest should work correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
