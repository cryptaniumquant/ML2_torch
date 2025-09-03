import pandas as pd
import numpy as np
from strategies_backtest_fifo import determine_market_regime_by_stop_losses

def test_regime_switching():
    """Test the regime switching logic based on stop loss counts"""
    
    # Create test positions DataFrame
    positions_df = pd.DataFrame(columns=['entry_date', 'entry_price', 'symbol', 'order_qty', 'position_type', 'exit_date', 'exit_price', 'pnl', 'exit_reason'])
    
    # Test 1: Mean-reversion regime with 19 stop losses (should stay in mean-reversion)
    print("Test 1: Mean-reversion with 19 stop losses")
    for i in range(19):
        positions_df = pd.concat([positions_df, pd.DataFrame([{
            'entry_date': f'2023-01-{i+1:02d}',
            'entry_price': 100.0,
            'symbol': 'BTCUSDT',
            'order_qty': 1.0,
            'position_type': 'long',
            'exit_date': f'2023-01-{i+1:02d}',
            'exit_price': 97.0,
            'pnl': -3.0,
            'exit_reason': 'stop_loss'
        }])], ignore_index=True)
    
    result = determine_market_regime_by_stop_losses(positions_df, 'mean_reversion')
    print(f"Result: {result} (Expected: mean_reversion)")
    assert result == 'mean_reversion', f"Expected mean_reversion, got {result}"
    
    # Test 2: Mean-reversion regime with 20 stop losses (should switch to momentum)
    print("\nTest 2: Mean-reversion with 20 stop losses")
    positions_df = pd.concat([positions_df, pd.DataFrame([{
        'entry_date': '2023-01-20',
        'entry_price': 100.0,
        'symbol': 'BTCUSDT',
        'order_qty': 1.0,
        'position_type': 'long',
        'exit_date': '2023-01-20',
        'exit_price': 97.0,
        'pnl': -3.0,
        'exit_reason': 'stop_loss'
    }])], ignore_index=True)
    
    result = determine_market_regime_by_stop_losses(positions_df, 'mean_reversion')
    print(f"Result: {result} (Expected: momentum)")
    assert result == 'momentum', f"Expected momentum, got {result}"
    
    # Test 3: Add some signal exits to test momentum regime
    print("\nTest 3: Momentum with mixed exits")
    for i in range(10):
        positions_df = pd.concat([positions_df, pd.DataFrame([{
            'entry_date': f'2023-02-{i+1:02d}',
            'entry_price': 100.0,
            'symbol': 'BTCUSDT',
            'order_qty': 1.0,
            'position_type': 'long',
            'exit_date': f'2023-02-{i+1:02d}',
            'exit_price': 103.0,
            'pnl': 3.0,
            'exit_reason': 'signal'
        }])], ignore_index=True)
    
    # Add 20 more stop losses (should not switch yet, need 21 for momentum)
    for i in range(20):
        positions_df = pd.concat([positions_df, pd.DataFrame([{
            'entry_date': f'2023-02-{i+11:02d}',
            'entry_price': 100.0,
            'symbol': 'BTCUSDT',
            'order_qty': 1.0,
            'position_type': 'long',
            'exit_date': f'2023-02-{i+11:02d}',
            'exit_price': 96.0,
            'pnl': -4.0,
            'exit_reason': 'stop_loss'
        }])], ignore_index=True)
    
    result = determine_market_regime_by_stop_losses(positions_df, 'momentum')
    print(f"Result: {result} (Expected: momentum)")
    assert result == 'momentum', f"Expected momentum, got {result}"
    
    # Test 4: Add one more stop loss to trigger switch to mean-reversion
    print("\nTest 4: Momentum with 21 stop losses")
    positions_df = pd.concat([positions_df, pd.DataFrame([{
        'entry_date': '2023-03-01',
        'entry_price': 100.0,
        'symbol': 'BTCUSDT',
        'order_qty': 1.0,
        'position_type': 'long',
        'exit_date': '2023-03-01',
        'exit_price': 96.0,
        'pnl': -4.0,
        'exit_reason': 'stop_loss'
    }])], ignore_index=True)
    
    result = determine_market_regime_by_stop_losses(positions_df, 'momentum')
    print(f"Result: {result} (Expected: mean_reversion)")
    assert result == 'mean_reversion', f"Expected mean_reversion, got {result}"
    
    print("\n✅ All regime switching tests passed!")

if __name__ == "__main__":
    test_regime_switching()
