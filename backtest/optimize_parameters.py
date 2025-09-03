import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the main backtest function
from strategies_backtest_trailing_numba import run_backtest_numba_optimized, load_signals_data

def calculate_metrics(balance_history):
    """Calculate performance metrics from balance history"""
    if not balance_history or len(balance_history) < 2:
        return {
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
    
    # Convert to DataFrame
    df = pd.DataFrame(balance_history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate returns
    initial_balance = 100000  # INIT_DEPOSIT
    final_balance = df['balance'].iloc[-1]
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    # Calculate daily returns
    df['daily_return'] = df['balance'].pct_change().fillna(0)
    
    # Calculate maximum drawdown
    df['cumulative_max'] = df['balance'].expanding().max()
    df['drawdown'] = (df['balance'] - df['cumulative_max']) / df['cumulative_max'] * 100
    max_drawdown = abs(df['drawdown'].min())
    
    # Calculate Calmar ratio (annualized return / max drawdown)
    days_in_backtest = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    annualized_return = total_return * (365 / days_in_backtest) if days_in_backtest > 0 else 0
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = df['daily_return'].mean() / df['daily_return'].std() * np.sqrt(365) if df['daily_return'].std() > 0 else 0
    
    # Calculate win rate and profit factor from realized PnL
    positive_days = df[df['daily_return'] > 0]['daily_return'].sum()
    negative_days = abs(df[df['daily_return'] < 0]['daily_return'].sum())
    win_rate = len(df[df['daily_return'] > 0]) / len(df[df['daily_return'] != 0]) * 100 if len(df[df['daily_return'] != 0]) > 0 else 0
    profit_factor = positive_days / negative_days if negative_days > 0 else float('inf') if positive_days > 0 else 0
    
    return {
        'total_return': round(total_return, 2),
        'max_drawdown': round(max_drawdown, 2),
        'calmar_ratio': round(calmar_ratio, 3),
        'sharpe_ratio': round(sharpe_ratio, 3),
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 3),
        'final_balance': round(final_balance, 2),
        'days_in_backtest': days_in_backtest
    }

def run_single_optimization(params, price_file, signals_file):
    """Run a single backtest with given parameters"""
    try:
        # Temporarily modify the global constants
        import strategies_backtest_trailing_numba as bt_module
        
        # Store original values
        original_values = {
            'TOKEN_SENSITIVITY': bt_module.TOKEN_SENSITIVITY,
            'SL_ATR_MULTIPLIER': bt_module.SL_ATR_MULTIPLIER,
            'ATR_PERIOD_MINUTES': bt_module.ATR_PERIOD_MINUTES,
            'SMA_WINDOW_SIZE': bt_module.SMA_WINDOW_SIZE,
            'FIXED_TRADE_SIZE': bt_module.FIXED_TRADE_SIZE,
            'MAX_POSITION_COUNT': bt_module.MAX_POSITION_COUNT
        }
        
        # Set new parameters
        bt_module.TOKEN_SENSITIVITY = params['token_sensitivity']
        bt_module.SL_ATR_MULTIPLIER = params['sl_atr_multiplier']
        bt_module.ATR_PERIOD_MINUTES = params['atr_period']
        bt_module.SMA_WINDOW_SIZE = params['sma_window']
        bt_module.FIXED_TRADE_SIZE = params['trade_size']
        bt_module.MAX_POSITION_COUNT = params['max_position_count']
        
        # Recalculate thresholds with new sensitivity
        bt_module.TAKE_PROFIT_THRESHOLDS = {
            'level1': 2.0 * bt_module.TOKEN_SENSITIVITY,
            'level2': 4.0 * bt_module.TOKEN_SENSITIVITY,
            'level3': 6.0 * bt_module.TOKEN_SENSITIVITY,
            'level4': 10.0 * bt_module.TOKEN_SENSITIVITY
        }
        
        # Load data
        df = pd.read_csv(price_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values("Date").reset_index(drop=True)
        
        # Filter data to match signal timeframe
        signals_start_date = pd.to_datetime('2025-03-10')
        df = df[df['Date'] >= signals_start_date].reset_index(drop=True)
        
        # Load signals
        signals_dict = load_signals_data(signals_file)
        
        # Run backtest without visualization
        results = run_backtest_numba_optimized(
            df, 
            signals_dict,
            enable_take_profit=True, 
            test_mode=False,
            create_chart=False  
        )
        
        # Extract results from dictionary
        virtual_positions_df = results['virtual_positions']
        real_positions_df = results['real_positions']
        real_pnl_history = results['real_pnl_history']
        filtered_real_pnl_history = results['filtered_real_pnl_history']
        virtual_pnl_history = results['virtual_pnl_history']
        filtered_virtual_pnl_history = results['filtered_virtual_pnl_history']
        
        # Create balance history from positions
        def create_balance_history(positions_df, pnl_type):
            balance_history = []
            if positions_df.empty:
                return balance_history
            
            # Get all unique dates from positions
            all_dates = []
            for _, row in positions_df.iterrows():
                if pd.notna(row['entry_date']):
                    all_dates.append(row['entry_date'])
                if pd.notna(row['exit_date']):
                    all_dates.append(row['exit_date'])
            
            if not all_dates:
                return balance_history
            
            all_dates = sorted(set(all_dates))
            
            for date in all_dates:
                # Calculate realized PnL up to this date
                realized_pnl = positions_df[
                    (pd.notna(positions_df['exit_date'])) & 
                    (positions_df['exit_date'] <= date) &
                    (pd.notna(positions_df['pnl']))
                ]['pnl'].sum()
                
                # Calculate unrealized PnL for open positions
                open_positions = positions_df[
                    (positions_df['entry_date'] <= date) & 
                    ((pd.isna(positions_df['exit_date'])) | (positions_df['exit_date'] > date))
                ]
                
                unrealized_pnl = 0
                if not open_positions.empty:
                    # Use last known price for unrealized PnL calculation
                    last_price = df['Close'].iloc[-1] if not df.empty else 100000
                    for _, pos in open_positions.iterrows():
                        if pos['position_type'] == 'long':
                            unrealized_pnl += (last_price - pos['entry_price']) * pos['order_qty']
                        else:
                            unrealized_pnl += (pos['entry_price'] - last_price) * pos['order_qty']
                
                total_pnl = realized_pnl + unrealized_pnl
                
                balance_history.append({
                    'date': date,
                    'balance': bt_module.INIT_DEPOSIT + total_pnl,
                    'realized_pnl': realized_pnl,
                    'unrealized_pnl': unrealized_pnl,
                    'total_pnl': total_pnl
                })
            
            return balance_history
        
        virtual_balance_history = create_balance_history(virtual_positions_df, 'virtual')
        real_balance_history = create_balance_history(real_positions_df, 'real')
        
        # Calculate metrics
        metrics = calculate_metrics(real_balance_history)
        
        # Restore original values
        for key, value in original_values.items():
            setattr(bt_module, key, value)
        
        # Restore original thresholds
        bt_module.TAKE_PROFIT_THRESHOLDS = {
            'level1': 2.0 * bt_module.TOKEN_SENSITIVITY,
            'level2': 4.0 * bt_module.TOKEN_SENSITIVITY,
            'level3': 6.0 * bt_module.TOKEN_SENSITIVITY,
            'level4': 10.0 * bt_module.TOKEN_SENSITIVITY
        }
        
        return {
            'params': params,
            'metrics': metrics,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'params': params,
            'metrics': None,
            'success': False,
            'error': str(e)
        }

def optimize_parameters():
    """Run parameter optimization"""
    
    # Define parameter ranges for optimization
    param_ranges = {
        'token_sensitivity': [0.5, 1.0, 1.5],  # Sensitivity multiplier
        'sl_atr_multiplier': [30, 60],   # Stop-loss ATR multiplier
        'atr_period': [120],                # ATR period
        'sma_window': [120],                 # SMA window size
        'trade_size': [1000],               # Trade size
        'max_position_count': [200, 300, 400]  # Maximum concurrent positions
    }
    
    # Data files
    price_file = "data/SOLUSDT_1m_20210101_to_20250820.csv"
    signals_file = "data/sol_new.csv"
    
    print("=== PARAMETER OPTIMIZATION START ===")
    print(f"Price data: {price_file}")
    print(f"Signals data: {signals_file}")
    print("\nParameter ranges:")
    for param, values in param_ranges.items():
        print(f"  {param}: {values}")
    
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    all_combinations = list(itertools.product(*param_values))
    
    total_combinations = len(all_combinations)
    print(f"\nTotal combinations to test: {total_combinations}")
    print("=" * 50)
    
    results = []
    start_time = time.time()
    
    # Run optimization with progress bar
    for i, combination in enumerate(tqdm(all_combinations, desc="Optimizing")):
        params = dict(zip(param_names, combination))
        
        result = run_single_optimization(params, price_file, signals_file)
        results.append(result)
        
        # Print progress every 10%
        if (i + 1) % max(1, total_combinations // 10) == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / total_combinations * 100
            eta = elapsed / (i + 1) * (total_combinations - i - 1)
            print(f"\nProgress: {progress:.1f}% | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\n=== OPTIMIZATION COMPLETE ===")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Successful runs: {len(successful_results)}/{total_combinations}")
    print(f"Failed runs: {len(failed_results)}")
    
    if failed_results:
        print("\nFailed combinations:")
        for result in failed_results[:5]:  # Show first 5 failures
            print(f"  {result['params']} -> Error: {result['error']}")
    
    if not successful_results:
        print("No successful optimization runs!")
        return
    
    # Sort results by different metrics
    metrics_to_sort = ['calmar_ratio', 'total_return', 'sharpe_ratio']
    
    print(f"\n=== TOP 10 RESULTS BY DIFFERENT METRICS ===")
    
    for metric in metrics_to_sort:
        print(f"\n--- TOP 10 BY {metric.upper().replace('_', ' ')} ---")
        sorted_results = sorted(successful_results, 
                              key=lambda x: x['metrics'][metric], 
                              reverse=True)
        
        print(f"{'Rank':<4} {'Token_Sens':<10} {'ATR_Mult':<8} {'ATR_Per':<7} {'SMA_Win':<7} {'Trade_Size':<10} {'Max_Pos':<7} {metric.replace('_', ' ').title():<12} {'Total_Ret%':<10} {'Max_DD%':<8} {'Calmar':<8}")
        print("-" * 110)
        
        for i, result in enumerate(sorted_results[:10]):
            params = result['params']
            metrics = result['metrics']
            print(f"{i+1:<4} {params['token_sensitivity']:<10} {params['sl_atr_multiplier']:<8} {params['atr_period']:<7} {params['sma_window']:<7} {params['trade_size']:<10} {params['max_position_count']:<7} {metrics[metric]:<12} {metrics['total_return']:<10} {metrics['max_drawdown']:<8} {metrics['calmar_ratio']:<8}")
    
    # Save detailed results to CSV
    results_data = []
    for result in successful_results:
        row = {}
        row.update(result['params'])
        row.update(result['metrics'])
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_results_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    
    print(f"\n=== RESULTS SAVED ===")
    print(f"Detailed results saved to: {filename}")
    print(f"Total rows: {len(results_df)}")
    
    # Show best overall result
    best_result = max(successful_results, key=lambda x: x['metrics']['calmar_ratio'])
    print(f"\n=== BEST OVERALL RESULT (by Calmar Ratio) ===")
    print("Parameters:")
    for param, value in best_result['params'].items():
        print(f"  {param}: {value}")
    print("\nMetrics:")
    for metric, value in best_result['metrics'].items():
        print(f"  {metric}: {value}")

if __name__ == "__main__":
    optimize_parameters()
