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
from numba import jit, njit, types
from numba.typed import Dict, List
import numba
from scipy.signal import savgol_filter
from scipy.linalg import lstsq
import time
import cProfile
import pstats
from functools import wraps

log = logging.getLogger("backtest")

# Performance profiling decorator
def profile_time(func_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Timer {func_name}: {end_time - start_time:.4f} seconds")
            return result
        return wrapper
    return decorator

# Global performance tracking
performance_stats = {}

# Performance optimization flags
USE_FAST_EMA_FILTER = False  # Use proper causal Savgol filter instead of EMA
REDUCE_FILTER_FREQUENCY = 100  # Only filter every N candles

# ===== TRADING SYSTEM CONSTANTS =====
# Initial capital and position sizing
INIT_DEPOSIT = 100_000  # Initial capital
FIXED_TRADE_SIZE = 1000  # Fixed dollar amount per trade
MAX_POSITION_COUNT = 400  # Maximum concurrent positions

# ATR-based trailing stop parameters
ATR_PERIOD_MINUTES = 120  # ATR calculation period
SL_ATR_MULTIPLIER = 30.0  # Stop-loss ATR multiplier
TRAIL_ON_NEW_SIGNAL = True  # Update trailing levels on matching signals

# Simple moving average parameters for virtual PnL smoothing
SMA_WINDOW_SIZE = 120

# Token sensitivity multiplier for take-profit thresholds
TOKEN_SENSITIVITY = 1  # Multiplier for take-profit thresholds (1.0 = default, >1.0 = less sensitive, <1.0 = more sensitive)

# Take-profit system thresholds (percentage of capital)
TAKE_PROFIT_THRESHOLDS = {
    'level1': 4.0 * TOKEN_SENSITIVITY,  # 3% of capital - close 25% of positions
    'level2': 6.0 * TOKEN_SENSITIVITY,  # 4% of capital - close 25% of positions
    'level3': 8.0 * TOKEN_SENSITIVITY,  # 5% of capital - close 25% of positions
    'level4': 10.0 * TOKEN_SENSITIVITY   # 6% of capital - close 25% of positions
}

# Position sizing for take-profit levels (percentage of open positions to close)
TAKE_PROFIT_POSITION_SIZES = {
    'level1': 0.1,  # Close 25% of profitable positions
    'level2': 0.2,  # Close 25% of profitable positions
    'level3': 0.3,  # Close 25% of profitable positions
    'level4': 0.5   # Close 25% of profitable positions
}

# Commission rate
COMMISSION_RATE = 0.0002  # 0.02%

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


@njit
def simple_moving_average(data, window_size):
    """
    Calculate simple moving average for the last point only
    Uses only past data - causal implementation
    """
    n = len(data)
    if n == 0:
        return 0.0
    
    # Use only past data up to current point
    actual_window = min(window_size, n)
    window_data = data[-actual_window:]
    
    return np.mean(window_data)




@njit
def calculate_atr_numba(high_prices, low_prices, close_prices, period):
    """
    Calculate ATR using Numba for speed
    Returns ATR as percentage of current price
    """
    n = len(high_prices)
    if n < period + 1:
        return np.nan
    
    # Calculate True Range
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr1 = high_prices[i] - low_prices[i]
        tr2 = abs(high_prices[i] - close_prices[i-1])
        tr3 = abs(low_prices[i] - close_prices[i-1])
        tr[i] = max(tr1, max(tr2, tr3))
    
    # Calculate ATR (average of last 'period' true ranges)
    atr_absolute = np.mean(tr[n-period:n])
    current_price = close_prices[n-1]
    
    # Return as percentage
    return atr_absolute / current_price


@njit
def update_trailing_levels_numba(
    position_types,  # array of position types (0=long, 1=short)
    entry_prices,
    current_stop_prices,
    exit_dates_valid,  # boolean array indicating if position is closed
    current_price,
    atr_percentage,
    sl_atr_mult,
    current_signal,
    trail_on_new_signal
):
    """
    Numba-optimized trailing levels update
    Returns updated stop prices
    """
    n_positions = len(position_types)
    new_stop_prices = current_stop_prices.copy()
    atr_absolute = atr_percentage * current_price
    
    for i in range(n_positions):
        # Skip closed positions
        if exit_dates_valid[i]:
            continue
            
        side = position_types[i]  # 0=long, 1=short
        entry = entry_prices[i]
        cur_sl = current_stop_prices[i]
        
        # Initialize stop loss if NaN
        if np.isnan(cur_sl):
            if side == 0:  # long
                cur_sl = entry - (atr_absolute * sl_atr_mult)
            else:  # short
                cur_sl = entry + (atr_absolute * sl_atr_mult)
            new_stop_prices[i] = cur_sl
            continue
        
        # Check if we should update levels based on new signal
        should_update_levels = True
        if trail_on_new_signal and current_signal != -1:  # -1 means no signal
            # Only update if signal matches position direction
            signal_matches_position = (
                (side == 0 and current_signal == 2) or  # Buy signal for long
                (side == 1 and current_signal == 0)     # Sell signal for short
            )
            if not signal_matches_position:
                should_update_levels = False
        
        if should_update_levels:
            # Calculate new stop level
            if side == 0:  # long
                sl_candidate = current_price - sl_atr_mult * atr_absolute
                # Trailing ratchet: only move up
                new_sl = max(cur_sl, sl_candidate)
            else:  # short
                sl_candidate = current_price + sl_atr_mult * atr_absolute
                # Trailing ratchet: only move down
                new_sl = min(cur_sl, sl_candidate)
            
            new_stop_prices[i] = new_sl
    
    return new_stop_prices


@njit
def check_exits_numba(
    position_types,  # 0=long, 1=short
    entry_prices,
    order_qtys,
    stop_prices,
    exit_dates_valid,  # boolean array indicating if position is closed
    high_price,
    low_price,
    commission_rate
):
    """
    Numba-optimized exit checking for virtual PnL (trailing stops)
    Returns arrays of exit flags, exit prices, and PnLs
    """
    n_positions = len(position_types)
    exit_triggered = np.zeros(n_positions, dtype=numba.boolean)
    exit_prices = np.full(n_positions, np.nan)
    pnls = np.full(n_positions, np.nan)
    
    for i in range(n_positions):
        # Skip closed positions
        if exit_dates_valid[i]:
            continue
            
        position_type = position_types[i]
        entry_price = entry_prices[i]
        order_qty = order_qtys[i]
        stop_price = stop_prices[i]
        
        if np.isnan(stop_price):
            continue
            
        exit_hit = False
        exit_price = np.nan
        
        if position_type == 0:  # long
            if low_price <= stop_price:
                exit_hit = True
                exit_price = stop_price
        else:  # short
            if high_price >= stop_price:
                exit_hit = True
                exit_price = stop_price
        
        if exit_hit:
            # Calculate PnL with commission
            entry_commission = entry_price * order_qty * commission_rate
            exit_commission = exit_price * order_qty * commission_rate
            total_commission = entry_commission + exit_commission
            
            if position_type == 0:  # long
                pnl = (exit_price - entry_price) * order_qty - total_commission
            else:  # short
                pnl = (entry_price - exit_price) * order_qty - total_commission
            
            exit_triggered[i] = True
            exit_prices[i] = exit_price
            pnls[i] = pnl
    
    return exit_triggered, exit_prices, pnls


@njit
def calculate_real_pnl_numba(
    position_types,
    entry_prices, 
    order_qtys,
    position_pnls,
    exit_dates_valid,
    current_price
):
    """
    Numba-optimized calculation of current real PnL
    """
    total_pnl = 0.0
    
    for j in range(len(position_types)):
        if not exit_dates_valid[j]:  # Open position
            pos_type = position_types[j]
            entry_price = entry_prices[j]
            order_qty = order_qtys[j]
            
            # Calculate unrealized PnL
            if pos_type == 0:  # long
                unrealized_pnl = (current_price - entry_price) * order_qty
            else:  # short
                unrealized_pnl = (entry_price - current_price) * order_qty
            total_pnl += unrealized_pnl
        else:  # Closed position
            if not np.isnan(position_pnls[j]):
                total_pnl += position_pnls[j]
    
    return total_pnl


@njit
def find_profitable_positions_numba(
    position_types,
    entry_prices,
    order_qtys, 
    exit_dates_valid,
    current_price
):
    """
    Numba-optimized function to find and sort profitable positions
    Returns indices and PnLs sorted by profitability (descending)
    """
    # Pre-allocate maximum possible arrays
    max_positions = len(position_types)
    temp_indices = np.empty(max_positions, dtype=np.int32)
    temp_pnls = np.empty(max_positions, dtype=np.float64)
    
    # Single pass: collect profitable positions
    profitable_count = 0
    for j in range(max_positions):
        if not exit_dates_valid[j]:
            pos_type = position_types[j]
            entry_price = entry_prices[j]
            order_qty = order_qtys[j]
            
            if pos_type == 0:  # long
                position_pnl = (current_price - entry_price) * order_qty
            else:  # short
                position_pnl = (entry_price - current_price) * order_qty
            
            if position_pnl > 0:
                temp_indices[profitable_count] = j
                temp_pnls[profitable_count] = position_pnl
                profitable_count += 1
    
    # Trim arrays to actual size
    profitable_indices = temp_indices[:profitable_count]
    profitable_pnls = temp_pnls[:profitable_count]
    
    # Quick sort implementation for better performance than bubble sort
    if profitable_count > 1:
        # Simple insertion sort (faster for small arrays)
        for i in range(1, profitable_count):
            key_pnl = profitable_pnls[i]
            key_idx = profitable_indices[i]
            j = i - 1
            
            # Move elements that are smaller than key to one position ahead
            while j >= 0 and profitable_pnls[j] < key_pnl:
                profitable_pnls[j + 1] = profitable_pnls[j]
                profitable_indices[j + 1] = profitable_indices[j]
                j -= 1
            
            profitable_pnls[j + 1] = key_pnl
            profitable_indices[j + 1] = key_idx
    
    return profitable_indices, profitable_pnls


def check_take_profit_signals(virtual_pnl_history, filtered_virtual_pnl_history, current_balance):
    """
    Check for take-profit signals based on distance between virtual and SMA virtual PnL
    Returns dictionary with take-profit levels triggered
    """
    if len(virtual_pnl_history) < SMA_WINDOW_SIZE:
        return {}
    
    # Convert to percentages of capital
    virtual_pnl_pct = (virtual_pnl_history[-1] / INIT_DEPOSIT) * 100
    sma_virtual_pnl_pct = (filtered_virtual_pnl_history[-1] / INIT_DEPOSIT) * 100
    
    # Calculate distance (only when virtual > SMA virtual)
    distance_pct = max(0, virtual_pnl_pct - sma_virtual_pnl_pct)
    
    # Check which take-profit levels are triggered
    triggered_levels = {}
    for level, threshold in TAKE_PROFIT_THRESHOLDS.items():
        if distance_pct > threshold:
            triggered_levels[level] = {
                'distance': distance_pct,
                'threshold': threshold,
                'virtual_pnl_pct': virtual_pnl_pct,
                'sma_virtual_pnl_pct': sma_virtual_pnl_pct,
                'position_size_to_close': TAKE_PROFIT_POSITION_SIZES[level]
            }
    
    return triggered_levels


@njit
def should_open_position_numba(
    predicted_class,
    position_types,
    exit_dates_valid,
    max_pos_count
):
    """
    Check if we should open a new position based on signal and current position count
    """
    if predicted_class not in (0, 2):  # Only sell (0) and buy (2) signals
        return False
    
    # Count open positions
    open_positions = 0
    for i in range(len(position_types)):
        if not exit_dates_valid[i]:
            open_positions += 1
    
    return open_positions < max_pos_count


@njit
def calculate_position_levels_numba(predicted_class, current_price, atr_absolute, sl_atr_mult):
    """
    Calculate entry levels for new position
    Returns: position_type (0=long, 1=short), stop_price, initial_risk
    """
    if predicted_class == 2:  # Buy signal - long position
        position_type = 0  # long
        stop_price = current_price - (atr_absolute * sl_atr_mult)
        initial_risk = current_price - stop_price
    elif predicted_class == 0:  # Sell signal - short position
        position_type = 1  # short
        stop_price = current_price + (atr_absolute * sl_atr_mult)
        initial_risk = stop_price - current_price
    else:
        return -1, np.nan, np.nan  # Invalid signal
    
    return position_type, stop_price, initial_risk


@profile_time("Main Backtest Loop")
def run_backtest_numba_optimized(df, signals_dict, start_idx=300, enable_take_profit=True, test_mode=False, create_chart=True):
    """
    Main Numba-optimized backtest function with dual PnL system
    Virtual PnL: Uses trailing stops only
    Real PnL: Uses take-profit system based on Savitzky-Golay filtering
    
    Uses global constants for all parameters
    """
    n_bars = len(df)
    
    # Convert DataFrame to numpy arrays for speed
    dates = df['Date'].values
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    # Pre-allocate position arrays (estimate max positions needed)
    max_total_positions = min(30000, n_bars // 10)
    
    # Virtual PnL positions (trailing stops)
    virtual_position_entry_dates = [pd.NaT] * max_total_positions
    virtual_position_entry_prices = np.full(max_total_positions, np.nan)
    virtual_position_order_qtys = np.full(max_total_positions, np.nan)
    virtual_position_types = np.full(max_total_positions, -1, dtype=np.int32)  # 0=long, 1=short, -1=empty
    virtual_position_exit_dates = [pd.NaT] * max_total_positions
    virtual_position_exit_prices = np.full(max_total_positions, np.nan)
    virtual_position_pnls = np.full(max_total_positions, np.nan)
    virtual_position_stop_prices = np.full(max_total_positions, np.nan)
    virtual_position_initial_stops = np.full(max_total_positions, np.nan)
    virtual_position_initial_risks = np.full(max_total_positions, np.nan)
    virtual_position_exit_dates_valid = np.full(max_total_positions, True)  # True = closed, False = open
    
    # Real PnL positions (take-profit system)
    real_position_entry_dates = [pd.NaT] * max_total_positions
    real_position_entry_prices = np.full(max_total_positions, np.nan)
    real_position_order_qtys = np.full(max_total_positions, np.nan)
    real_position_types = np.full(max_total_positions, -1, dtype=np.int32)
    real_position_exit_dates = [pd.NaT] * max_total_positions
    real_position_exit_prices = np.full(max_total_positions, np.nan)
    real_position_pnls = np.full(max_total_positions, np.nan)
    real_position_stop_prices = np.full(max_total_positions, np.nan)
    real_position_take_profit_levels = [None] * max_total_positions  # Track which TP level triggered
    real_position_exit_reasons = [None] * max_total_positions  # Track exit reason: 'take_profit' or 'trailing_stop'
    real_position_exit_dates_valid = np.full(max_total_positions, True)
    
    # Balance tracking
    virtual_balance = INIT_DEPOSIT
    real_balance = INIT_DEPOSIT
    virtual_position_count = 0
    real_position_count = 0
    
    # PnL history for Savitzky-Golay filtering
    real_pnl_history = []
    filtered_real_pnl_history = []
    virtual_pnl_history = []
    filtered_virtual_pnl_history = []
    
    # Take-profit state tracking
    tp_state = {
        'last_signal_indices': {'level1': -1, 'level2': -1, 'level3': -1, 'level4': -1},
        'waiting_for_derivative_change': {'level1': False, 'level2': False, 'level3': False, 'level4': False}
    }
    
    # Define take-profit colors
    take_profit_colors = {
        'level1': 'green',
        'level2': 'orange', 
        'level3': 'red',
        'level4': 'purple'
    }
    
    # Convert signals dict to timestamp-based lookup for speed
    date_to_signal = {}
    for dt, signal in signals_dict.items():
        timestamp = pd.Timestamp(dt).value
        date_to_signal[timestamp] = signal
    
    print(f"Starting dual PnL backtest from index {start_idx} to {n_bars}")
    print(f"Virtual PnL: Trailing stops only")
    print(f"Real PnL: Take-profit system with Savitzky-Golay filtering")
    
    # Main backtest loop
    loop_start_time = time.time()
    atr_total_time = 0
    signal_total_time = 0
    position_mgmt_time = 0
    
    for i in tqdm(range(start_idx, len(df)), desc="Running backtest", unit="candle"):
        current_date = dates[i]
        current_price = closes[i]
        high_price = highs[i]
        low_price = lows[i]
        
        # Get signal from previous candle (realistic timing)
        previous_signal = -1  # Default no signal
        if i > 0:
            prev_timestamp = pd.Timestamp(dates[i-1]).value
            previous_signal = date_to_signal.get(prev_timestamp, -1)
        
        # Calculate ATR for current window
        atr_pct = np.nan
        if i >= ATR_PERIOD_MINUTES:
            window_start = max(0, i - ATR_PERIOD_MINUTES)
            atr_start = time.time()
            atr_pct = calculate_atr_numba(
                df['High'].values[window_start:i+1],
                df['Low'].values[window_start:i+1], 
                df['Close'].values[window_start:i+1],
                ATR_PERIOD_MINUTES
            )
            atr_total_time += time.time() - atr_start
        
        if np.isnan(atr_pct):
            continue
        
        # === VIRTUAL PnL SYSTEM (Trailing Stops) ===
        # Update trailing levels for virtual positions
        pos_mgmt_start = time.time()
        if virtual_position_count > 0:
            new_stop_prices = update_trailing_levels_numba(
                virtual_position_types[:virtual_position_count],
                virtual_position_entry_prices[:virtual_position_count],
                virtual_position_stop_prices[:virtual_position_count],
                virtual_position_exit_dates_valid[:virtual_position_count],
                current_price,
                atr_pct,
                SL_ATR_MULTIPLIER,
                previous_signal,
                TRAIL_ON_NEW_SIGNAL
            )
            virtual_position_stop_prices[:virtual_position_count] = new_stop_prices
        position_mgmt_time += time.time() - pos_mgmt_start
        
        # Calculate current virtual PnL
        current_virtual_pnl = calculate_real_pnl_numba(
            virtual_position_types[:virtual_position_count],
            virtual_position_entry_prices[:virtual_position_count],
            virtual_position_order_qtys[:virtual_position_count],
            virtual_position_pnls[:virtual_position_count],
            virtual_position_exit_dates_valid[:virtual_position_count],
            current_price
        )
        
        # Update virtual PnL history
        virtual_pnl_history.append(current_virtual_pnl)
        
        # Apply simple moving average to virtual PnL
        if len(virtual_pnl_history) >= SMA_WINDOW_SIZE:
            # Calculate SMA using only past data
            recent_virtual_pnl = np.array(virtual_pnl_history[-SMA_WINDOW_SIZE:])
            filtered_virtual_value = simple_moving_average(recent_virtual_pnl, SMA_WINDOW_SIZE)
        else:
            # Not enough history, use current value
            filtered_virtual_value = current_virtual_pnl
                
        filtered_virtual_pnl_history.append(filtered_virtual_value)
        
        # Check for virtual exits (trailing stops)
        if virtual_position_count > 0:
            exit_flags, exit_prices_arr, pnls_arr = check_exits_numba(
                virtual_position_types[:virtual_position_count],
                virtual_position_entry_prices[:virtual_position_count],
                virtual_position_order_qtys[:virtual_position_count],
                virtual_position_stop_prices[:virtual_position_count],
                virtual_position_exit_dates_valid[:virtual_position_count],
                high_price,
                low_price,
                COMMISSION_RATE
            )
            
            # Process virtual exits
            for j in range(virtual_position_count):
                if exit_flags[j]:
                    virtual_position_exit_dates[j] = pd.Timestamp(current_date)
                    virtual_position_exit_prices[j] = exit_prices_arr[j]
                    virtual_position_pnls[j] = pnls_arr[j]
                    virtual_position_exit_dates_valid[j] = True
                    virtual_balance += pnls_arr[j]
        
        # === REAL PnL SYSTEM (Trailing Stops + Take-Profit) ===
        # Update trailing levels for real positions
        if real_position_count > 0:
            new_real_stop_prices = update_trailing_levels_numba(
                real_position_types[:real_position_count],
                real_position_entry_prices[:real_position_count],
                real_position_stop_prices[:real_position_count],
                real_position_exit_dates_valid[:real_position_count],
                current_price,
                atr_pct,
                SL_ATR_MULTIPLIER,
                previous_signal,
                TRAIL_ON_NEW_SIGNAL
            )
            real_position_stop_prices[:real_position_count] = new_real_stop_prices
        
        # Check for real exits (trailing stops)
        if real_position_count > 0:
            exit_flags, exit_prices_arr, pnls_arr = check_exits_numba(
                real_position_types[:real_position_count],
                real_position_entry_prices[:real_position_count],
                real_position_order_qtys[:real_position_count],
                real_position_stop_prices[:real_position_count],
                real_position_exit_dates_valid[:real_position_count],
                high_price,
                low_price,
                COMMISSION_RATE
            )
            
            # Process real trailing stop exits
            for j in range(real_position_count):
                if exit_flags[j]:
                    real_position_exit_dates[j] = pd.Timestamp(current_date)
                    real_position_exit_prices[j] = exit_prices_arr[j]
                    real_position_pnls[j] = pnls_arr[j]
                    real_position_exit_reasons[j] = 'trailing_stop'
                    real_position_exit_dates_valid[j] = True
                    real_balance += pnls_arr[j]
        
        # Calculate current real PnL using Numba-optimized function
        current_real_pnl = calculate_real_pnl_numba(
            real_position_types[:real_position_count],
            real_position_entry_prices[:real_position_count],
            real_position_order_qtys[:real_position_count],
            real_position_pnls[:real_position_count],
            real_position_exit_dates_valid[:real_position_count],
            current_price
        )
        
        # Update real PnL history
        real_pnl_history.append(current_real_pnl)
        
        # Store real PnL without filtering
        filtered_real_pnl_history.append(current_real_pnl)
        
        # Check for take-profit signals
        if enable_take_profit and len(virtual_pnl_history) >= SMA_WINDOW_SIZE:
            tp_start = time.time()
            tp_signals = check_take_profit_signals(
                virtual_pnl_history, 
                filtered_virtual_pnl_history, 
                virtual_balance
            )
            tp_end = time.time()
            performance_stats.setdefault('take_profit_signals', []).append(tp_end - tp_start)
            
            # Calculate derivative once for both state reset and signal validation
            current_derivative = 0.0
            prev_derivative = 0.0
            
            if len(filtered_virtual_pnl_history) > 1:
                # Use only last few points for derivative calculation to avoid O(n) growth
                derivative_window = min(10, len(filtered_virtual_pnl_history))
                recent_pnl = np.array(filtered_virtual_pnl_history[-derivative_window:])
                recent_derivative = np.gradient(recent_pnl)
                current_derivative = recent_derivative[-1]
                prev_derivative = recent_derivative[-2] if len(recent_derivative) > 1 else current_derivative
                
                # Check for derivative sign changes to reset states
                for level in TAKE_PROFIT_THRESHOLDS.keys():
                    if tp_state['waiting_for_derivative_change'][level]:
                        if prev_derivative > 0 and current_derivative < 0:
                            # Derivative changed from positive to negative - reset state
                            tp_state['waiting_for_derivative_change'][level] = False
                            print(f"  -> {level.upper()}: State reset due to derivative sign change (+->)")
            
            # Process take-profit signals
            if tp_signals:
                # Use already calculated derivatives
                    
                    # Check each take-profit level
                    for level, signal_data in tp_signals.items():
                        state = tp_state['waiting_for_derivative_change'][level]
                        last_signal_idx = tp_state['last_signal_indices'][level]
                        
                        # Trigger take-profit if conditions are met
                        should_trigger = False
                        if last_signal_idx == -1 or not state:
                            should_trigger = True
                        
                        if should_trigger:
                            # Close specified percentage of profitable real positions at this level
                            position_size_to_close = signal_data['position_size_to_close']
                            
                            # Find profitable positions using Numba-optimized function
                            profitable_indices, profitable_pnls = find_profitable_positions_numba(
                                real_position_types[:real_position_count],
                                real_position_entry_prices[:real_position_count],
                                real_position_order_qtys[:real_position_count],
                                real_position_exit_dates_valid[:real_position_count],
                                current_price
                            )
                            
                            # Calculate how many positions to close
                            positions_to_close = max(1, int(len(profitable_indices) * position_size_to_close))
                            positions_closed = 0
                            total_pnl_closed = 0
                            
                            for idx in range(min(positions_to_close, len(profitable_indices))):
                                if positions_closed < positions_to_close:
                                    j = profitable_indices[idx]
                                    position_pnl = profitable_pnls[idx]
                                    pos_type = real_position_types[j]
                                    entry_price = real_position_entry_prices[j]
                                    order_qty = real_position_order_qtys[j]
                                    
                                    # Close position
                                    real_position_exit_dates[j] = pd.Timestamp(current_date)
                                    real_position_exit_prices[j] = current_price
                                    
                                    # Calculate final PnL with commission
                                    entry_commission = entry_price * order_qty * COMMISSION_RATE
                                    exit_commission = current_price * order_qty * COMMISSION_RATE
                                    final_pnl = position_pnl - entry_commission - exit_commission
                                    
                                    real_position_pnls[j] = final_pnl
                                    real_position_take_profit_levels[j] = level
                                    real_position_exit_reasons[j] = 'take_profit'
                                    real_position_exit_dates_valid[j] = True
                                    real_balance += final_pnl
                                    total_pnl_closed += final_pnl
                                    positions_closed += 1
                            
                            if positions_closed > 0:
                                print(f"  -> {level.upper()} Take-profit: Closed {positions_closed} positions (${total_pnl_closed:.2f}) at distance {signal_data['distance']:.2f}%")
                            
                            # Update state
                            tp_state['last_signal_indices'][level] = i
                            tp_state['waiting_for_derivative_change'][level] = True
        
        # === POSITION OPENING (Both Virtual and Real) ===
        # Check for new position opening
        if previous_signal in (0, 2):
            # Check virtual positions
            can_open_virtual = should_open_position_numba(
                previous_signal,
                virtual_position_types[:virtual_position_count],
                virtual_position_exit_dates_valid[:virtual_position_count],
                MAX_POSITION_COUNT
            )
            
            # Check real positions
            can_open_real = should_open_position_numba(
                previous_signal,
                real_position_types[:real_position_count],
                real_position_exit_dates_valid[:real_position_count],
                MAX_POSITION_COUNT
            )
            
            # Debug: Log signal activity for July-August
            current_month = pd.Timestamp(current_date).month
            if current_month >= 7:  # July and August
                virtual_open_count = sum(1 for j in range(virtual_position_count) if not virtual_position_exit_dates_valid[j])
                real_open_count = sum(1 for j in range(real_position_count) if not real_position_exit_dates_valid[j])
                
                if i % 1440 == 0:  # Log once per day (1440 minutes)
                    print(f"Debug {pd.Timestamp(current_date).strftime('%Y-%m-%d')}: Signal={previous_signal}")
                    print(f"  Virtual: {virtual_open_count}/{MAX_POSITION_COUNT}, Real: {real_open_count}/{MAX_POSITION_COUNT}")
            
            # Open virtual position
            if can_open_virtual and virtual_position_count < max_total_positions:
                order_qty = FIXED_TRADE_SIZE / current_price
                atr_absolute = atr_pct * current_price
                
                pos_type, stop_price, initial_risk = calculate_position_levels_numba(
                    previous_signal, current_price, atr_absolute, SL_ATR_MULTIPLIER
                )
                
                if pos_type != -1:  # Valid position
                    virtual_position_entry_dates[virtual_position_count] = pd.Timestamp(current_date)
                    virtual_position_entry_prices[virtual_position_count] = current_price
                    virtual_position_order_qtys[virtual_position_count] = order_qty
                    virtual_position_types[virtual_position_count] = pos_type
                    virtual_position_stop_prices[virtual_position_count] = stop_price
                    virtual_position_initial_stops[virtual_position_count] = stop_price
                    virtual_position_initial_risks[virtual_position_count] = initial_risk
                    virtual_position_exit_dates_valid[virtual_position_count] = False
                    
                    virtual_position_count += 1
            
            # Open real position (same logic but separate tracking)
            if can_open_real and real_position_count < max_total_positions:
                order_qty = FIXED_TRADE_SIZE / current_price
                atr_absolute = atr_pct * current_price
                
                pos_type, stop_price, initial_risk = calculate_position_levels_numba(
                    previous_signal, current_price, atr_absolute, SL_ATR_MULTIPLIER
                )
                
                if pos_type != -1:  # Valid position
                    real_position_entry_dates[real_position_count] = pd.Timestamp(current_date)
                    real_position_entry_prices[real_position_count] = current_price
                    real_position_order_qtys[real_position_count] = order_qty
                    real_position_types[real_position_count] = pos_type
                    real_position_stop_prices[real_position_count] = stop_price
                    real_position_exit_dates_valid[real_position_count] = False
                    
                    real_position_count += 1
                    
                    # Debug: Log new position opening
                    if current_month >= 7:
                        pos_name = "LONG" if pos_type == 0 else "SHORT"
                        print(f"  -> Opened {pos_name} (Virtual & Real) at ${current_price:.2f}, SL=${stop_price:.2f}")
    
    # Create results DataFrames for both virtual and real positions
    virtual_results_data = {
        'entry_date': virtual_position_entry_dates[:virtual_position_count],
        'entry_price': virtual_position_entry_prices[:virtual_position_count],
        'symbol': ['SOLUSDT'] * virtual_position_count,
        'order_qty': virtual_position_order_qtys[:virtual_position_count],
        'position_type': ['long' if pt == 0 else 'short' for pt in virtual_position_types[:virtual_position_count]],
        'exit_date': virtual_position_exit_dates[:virtual_position_count],
        'exit_price': virtual_position_exit_prices[:virtual_position_count],
        'pnl': virtual_position_pnls[:virtual_position_count],
        'stop_price': virtual_position_stop_prices[:virtual_position_count],
        'take_price': [np.nan] * virtual_position_count,
        'initial_stop': virtual_position_initial_stops[:virtual_position_count],
        'initial_take': [np.nan] * virtual_position_count,
        'initial_risk': virtual_position_initial_risks[:virtual_position_count],
        'pnl_type': ['virtual'] * virtual_position_count
    }
    
    real_results_data = {
        'entry_date': real_position_entry_dates[:real_position_count],
        'entry_price': real_position_entry_prices[:real_position_count],
        'symbol': ['SOLUSDT'] * real_position_count,
        'order_qty': real_position_order_qtys[:real_position_count],
        'position_type': ['long' if pt == 0 else 'short' for pt in real_position_types[:real_position_count]],
        'exit_date': real_position_exit_dates[:real_position_count],
        'exit_price': real_position_exit_prices[:real_position_count],
        'pnl': real_position_pnls[:real_position_count],
        'stop_price': real_position_stop_prices[:real_position_count],
        'take_price': [np.nan] * real_position_count,
        'initial_stop': [np.nan] * real_position_count,  # Real positions don't use initial stops
        'initial_take': [np.nan] * real_position_count,
        'initial_risk': [np.nan] * real_position_count,
        'pnl_type': ['real'] * real_position_count,
        'take_profit_level': real_position_take_profit_levels[:real_position_count],
        'exit_reason': real_position_exit_reasons[:real_position_count]
    }
    
    virtual_positions_df = pd.DataFrame(virtual_results_data)
    real_positions_df = pd.DataFrame(real_results_data)
    
    # Convert NaT values properly
    for df in [virtual_positions_df, real_positions_df]:
        for col in ['entry_date', 'exit_date']:
            if col in df.columns:
                mask = pd.isna(df[col])
                df.loc[mask, col] = pd.NaT
    
    # Record loop timing statistics
    loop_end_time = time.time()
    total_loop_time = loop_end_time - loop_start_time
    
    performance_stats['total_loop_time'] = [total_loop_time]
    performance_stats['atr_calculation'] = [atr_total_time]
    performance_stats['signal_lookup'] = [signal_total_time]
    performance_stats['position_management'] = [position_mgmt_time]
    
    return {
        'virtual_positions': virtual_positions_df,
        'real_positions': real_positions_df,
        'real_pnl_history': real_pnl_history,
        'filtered_real_pnl_history': filtered_real_pnl_history,
        'virtual_pnl_history': virtual_pnl_history,
        'filtered_virtual_pnl_history': filtered_virtual_pnl_history
    }


def print_trading_statistics(positions_df, pnl_type="Virtual"):
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

    return_pct = (total_pnl / INIT_DEPOSIT) * 100

    print("\n" + "=" * 60)
    print(f"              {pnl_type.upper()} TRADING STATISTICS")
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
    print(f"Initial Deposit:     ${INIT_DEPOSIT:.2f}")
    print(f"Final Balance:       ${INIT_DEPOSIT + total_pnl:.2f}")
    print(f"Return:              {return_pct:.2f}%")
    print("=" * 60)


def create_plotly_chart_dual_pnl(virtual_positions_df, real_positions_df, virtual_balance_history, real_balance_history, price_data, symbol, real_pnl_history=None, filtered_real_pnl_history=None, virtual_pnl_history=None, filtered_virtual_pnl_history=None):
    """
    Create interactive Plotly chart with dual PnL system: Virtual (trailing) and Real (take-profit)
    """
    if virtual_positions_df.empty and real_positions_df.empty:
        print("No data to visualize")
        return

    # Create subplots with dual PnL layout + take-profit stats
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            f'{symbol} Price & Take-Profit Points', 
            'Virtual PnL (%) - Trailing Stops',
            'Real PnL (%) - Take-Profit System',
            'Take-Profit Levels Distribution'
        ],
        row_heights=[0.4, 0.2, 0.2, 0.2],
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )

    # Define take-profit colors for chart
    take_profit_colors = {
        'level1': 'green',
        'level2': 'orange', 
        'level3': 'red',
        'level4': 'purple'
    }
    
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

    # Calculate position timelines for both virtual and real
    def create_position_timeline(positions_df, name_prefix):
        timeline = []
        for _, row in positions_df.iterrows():
            if pd.notna(row['entry_date']):
                position_change = 1 if row['position_type'] == 'long' else -1
                timeline.append({'date': row['entry_date'], 'position_change': position_change})
            if pd.notna(row['exit_date']) and pd.notna(row['pnl']):
                position_change = -1 if row['position_type'] == 'long' else 1
                timeline.append({'date': row['exit_date'], 'position_change': position_change})
        
        if timeline:
            timeline_df = pd.DataFrame(timeline)
            timeline_df = timeline_df.sort_values('date')
            timeline_df['cumulative_positions'] = timeline_df['position_change'].cumsum()
            return timeline_df
        return None
    
    virtual_timeline = create_position_timeline(virtual_positions_df, 'Virtual')
    real_timeline = create_position_timeline(real_positions_df, 'Real')
    
    if virtual_timeline is not None:
        fig.add_trace(
            go.Scatter(
                x=virtual_timeline['date'],
                y=virtual_timeline['cumulative_positions'],
                name='Virtual Positions',
                line=dict(color='orange', width=2),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
    
    if real_timeline is not None:
        fig.add_trace(
            go.Scatter(
                x=real_timeline['date'],
                y=real_timeline['cumulative_positions'],
                name='Real Positions',
                line=dict(color='purple', width=2, dash='dot'),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )


    # Virtual PnL chart with smoothed version
    if virtual_balance_history:
        virtual_df = pd.DataFrame(virtual_balance_history)
        virtual_df['total_pnl_pct'] = (virtual_df['total_pnl'] / INIT_DEPOSIT) * 100
        
        # Raw virtual PnL
        fig.add_trace(
            go.Scatter(
                x=virtual_df['date'],
                y=virtual_df['total_pnl_pct'],
                name='Virtual PnL %',
                line=dict(color='blue', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # Add smoothed virtual PnL if available
        if virtual_pnl_history and filtered_virtual_pnl_history and price_data is not None:
            # Use price data dates for proper alignment
            start_idx = len(price_data) - len(virtual_pnl_history)
            if start_idx >= 0:
                pnl_dates = price_data['Date'].iloc[start_idx:].reset_index(drop=True)
                virtual_pnl_pct = [(pnl / INIT_DEPOSIT) * 100 for pnl in virtual_pnl_history]
                filtered_virtual_pnl_pct = [(pnl / INIT_DEPOSIT) * 100 for pnl in filtered_virtual_pnl_history]
                
                # Smoothed virtual PnL area
                fig.add_trace(
                    go.Scatter(
                        x=pnl_dates,
                        y=filtered_virtual_pnl_pct,
                        name='SMA Virtual PnL %',
                        line=dict(color='darkblue', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(0,0,139,0.3)',
                        opacity=0.8
                    ),
                    row=2, col=1
                )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    
    # Real PnL chart with Savitzky-Golay filtering
    if real_balance_history:
        real_df = pd.DataFrame(real_balance_history)
        real_df['total_pnl_pct'] = (real_df['total_pnl'] / INIT_DEPOSIT) * 100
        
        fig.add_trace(
            go.Scatter(
                x=real_df['date'],
                y=real_df['total_pnl_pct'],
                name='Real PnL %',
                line=dict(color='green', width=2),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7, row=3, col=1)

    # Plot take-profit exit points on price chart
    for level, color in take_profit_colors.items():
        tp_exits = real_positions_df[
            (real_positions_df['take_profit_level'] == level) & 
            (pd.notna(real_positions_df['exit_date'])) &
            (real_positions_df['exit_reason'] == 'take_profit')
        ]
        
        if not tp_exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=tp_exits['exit_date'],
                    y=tp_exits['exit_price'],
                    mode='markers',
                    name=f'Take-Profit {level.upper()}',
                    marker=dict(
                        color=color,
                        size=10,
                        symbol='diamond',
                        line=dict(width=2, color='black')
                    ),
                    text=[f"TP {level}: ${pnl:.2f}" for pnl in tp_exits['pnl']],
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Date: %{x}<br>' +
                                  'Price: $%{y:.2f}<br>' +
                                  '%{text}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Plot trailing stop exits
    trailing_exits = real_positions_df[
        (pd.notna(real_positions_df['exit_date'])) &
        (real_positions_df['exit_reason'] == 'trailing_stop')
    ]
    
    if not trailing_exits.empty:
        fig.add_trace(
            go.Scatter(
                x=trailing_exits['exit_date'],
                y=trailing_exits['exit_price'],
                mode='markers',
                name='Trailing Stop Exits',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='x',
                    line=dict(width=2, color='darkred')
                ),
                text=[f"SL: ${pnl:.2f}" for pnl in trailing_exits['pnl']],
                hovertemplate='<b>Trailing Stop Exit</b><br>' +
                              'Date: %{x}<br>' +
                              'Price: $%{y:.2f}<br>' +
                              '%{text}<extra></extra>'
            ),
            row=1, col=1
        )
            
    # Add take-profit statistics bar chart
    tp_stats = real_positions_df[real_positions_df['take_profit_level'].notna()]
    if not tp_stats.empty:
        tp_counts = tp_stats['take_profit_level'].value_counts().sort_index()
        tp_pnls = tp_stats.groupby('take_profit_level')['pnl'].sum()
        
        # Define colors for bar chart
        bar_colors = [take_profit_colors.get(level, 'gray') for level in tp_counts.index]
        
        # Bar chart for take-profit counts
        fig.add_trace(
            go.Bar(
                x=[f"TP {level.upper()}" for level in tp_counts.index],
                y=tp_counts.values,
                name='TP Exits Count',
                marker_color=bar_colors,
                text=[f"${pnl:.0f}" for pnl in tp_pnls[tp_counts.index]],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                              'Exits: %{y}<br>' +
                              'Total PnL: %{text}<extra></extra>'
            ),
            row=4, col=1
        )

    # Update layout
    fig.update_layout(
        title=f'{symbol} Dual PnL Trading Analysis with Take-Profit Visualization',
        xaxis_title='Date',
        height=1400,
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)')
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Position Count", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Virtual PnL (%)", row=2, col=1)
    fig.update_yaxes(title_text="Real PnL (%)", row=3, col=1)
    fig.update_yaxes(title_text="Exit Count", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)

    # Save and show
    filename = 'dual_pnl_trading_analysis.html'
    fig.write_html(filename)
    print(f"\nDual PnL interactive chart saved as {filename}")
    fig.show()


def main_numba(test_mode=False, create_chart=True):
    # Reset performance stats
    global performance_stats
    performance_stats = {}
    # Load price data
    price_file = "data/SOLUSDT_1m_20210101_to_20250820.csv"
    signals_file = "data/sol_new.csv"

    print("Loading data...")
    df = pd.read_csv(price_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date").reset_index(drop=True)

    # Filter data to match signal timeframe (signals start from 2025-03-04)
    signals_start_date = pd.to_datetime('2025-03-10')
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
        print("\n=== TEST MODE: Running on first 50,000 candles ===\n")
        df = df.head(50_000)
        
        # Check signals in test range
        test_signals = []
        for i in range(len(df)):
            dt = pd.to_datetime(df.iloc[i]['Date'])
            signal_start = time.time()
            current_signal = get_signal_for_datetime(dt, signals_dict)
            signal_total_time += time.time() - signal_start
            if current_signal is not None:
                test_signals.append(current_signal)
        
        test_signal_counts = pd.Series(test_signals).value_counts().to_dict()
        print(f"Test range signal distribution: {test_signal_counts}")
        
        if test_signal_counts.get(0, 0) == 0 and test_signal_counts.get(2, 0) == 0:
            print("WARNING: No trading signals (0 or 2) in test range!")
        else:
            print(f"Trading signals in test range: {test_signal_counts.get(0, 0)} sell + {test_signal_counts.get(2, 0)} buy")

    # Start backtest from a reasonable point (need some history for ATR calculation)
    start_idx = max(300, 0)
    
    print(f"Starting dual PnL backtest with global constants...")
    print(f"Parameters: init_deposit=${INIT_DEPOSIT:,}, max_pos={MAX_POSITION_COUNT}, trade_size=${FIXED_TRADE_SIZE}")
    print(f"ATR period: {ATR_PERIOD_MINUTES}m, SL multiplier: {SL_ATR_MULTIPLIER}")
    print(f"Take-profit thresholds: {list(TAKE_PROFIT_THRESHOLDS.values())}%")
    print(f"Position sizes to close: {list(TAKE_PROFIT_POSITION_SIZES.values())}")
    
    # Run the dual PnL backtest
    results = run_backtest_numba_optimized(
        df, 
        signals_dict,
        start_idx=start_idx,
        enable_take_profit=True, 
        test_mode=test_mode,
        create_chart=False
    )
    
    virtual_positions_df = results['virtual_positions']
    real_positions_df = results['real_positions']
    real_pnl_history = results['real_pnl_history']
    filtered_real_pnl_history = results['filtered_real_pnl_history']
    virtual_pnl_history = results['virtual_pnl_history']
    filtered_virtual_pnl_history = results['filtered_virtual_pnl_history']
    
    # Calculate final balances from positions
    virtual_balance = INIT_DEPOSIT + virtual_positions_df[virtual_positions_df['pnl'].notna()]['pnl'].sum()
    real_balance = INIT_DEPOSIT + real_positions_df[real_positions_df['pnl'].notna()]['pnl'].sum()
    
    print("==== Dual PnL Backtest Results ====")
    print(f"Start deposit: ${INIT_DEPOSIT:,.2f}")
    print(f"Virtual final balance: ${virtual_balance:,.2f}")
    print(f"Real final balance: ${real_balance:,.2f}")
    print(f"Virtual return: {((virtual_balance - INIT_DEPOSIT) / INIT_DEPOSIT * 100):.2f}%")
    print(f"Real return: {((real_balance - INIT_DEPOSIT) / INIT_DEPOSIT * 100):.2f}%")
    
    # Virtual trades statistics
    virtual_closed_trades = virtual_positions_df[virtual_positions_df['pnl'].notna()]['pnl'].tolist()
    real_closed_trades = real_positions_df[real_positions_df['pnl'].notna()]['pnl'].tolist()
    
    print(f"\nVirtual trades: {len(virtual_closed_trades)}")
    print(f"Real trades: {len(real_closed_trades)}")
    
    if virtual_closed_trades:
        avg_pnl = np.mean(virtual_closed_trades)
        win_rate = np.mean([1 if t > 0 else 0 for t in virtual_closed_trades])
        print(f"Virtual avg PnL: ${avg_pnl:.2f}, win rate: {win_rate:.2%}")
    
    if real_closed_trades:
        avg_pnl = np.mean(real_closed_trades)
        win_rate = np.mean([1 if t > 0 else 0 for t in real_closed_trades])
        print(f"Real avg PnL: ${avg_pnl:.2f}, win rate: {win_rate:.2%}")
        
        # Show take-profit level distribution
        tp_levels = real_positions_df[real_positions_df['take_profit_level'].notna()]['take_profit_level'].value_counts()
        if not tp_levels.empty:
            print(f"\nTake-profit level distribution:")
            for level, count in tp_levels.items():
                print(f"  {level}: {count} trades")
    
    # Print detailed statistics for both systems
    print_trading_statistics(virtual_positions_df, "Virtual")
    print_trading_statistics(real_positions_df, "Real")
    
    # Print performance statistics
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    for operation, times in performance_stats.items():
        if times:
            total_time = sum(times)
            avg_time = total_time / len(times)
            max_time = max(times)
            print(f"{operation:25s}: Total={total_time:8.4f}s, Avg={avg_time:8.6f}s, Max={max_time:8.6f}s, Calls={len(times)}")
    
    print("="*60)
    
    # Create balance histories for both virtual and real systems
    def create_balance_history(positions_df, pnl_type):
        balance_history = []
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
                exit_commission = current_price * pos['order_qty'] * COMMISSION_RATE
                if pos['position_type'] == 'long':
                    unrealized_pnl += (current_price - pos['entry_price']) * pos['order_qty'] - exit_commission
                else:  # short
                    unrealized_pnl += (pos['entry_price'] - current_price) * pos['order_qty'] - exit_commission
            
            total_pnl = realized_pnl + unrealized_pnl
            
            balance_history.append({
                'date': date,
                'balance': INIT_DEPOSIT + total_pnl,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': total_pnl
            })
        
        return balance_history
    
    virtual_balance_history = create_balance_history(virtual_positions_df, 'virtual')
    real_balance_history = create_balance_history(real_positions_df, 'real')
    
    # Create interactive dual PnL chart
    if create_chart and (not virtual_positions_df.empty or not real_positions_df.empty):
        print("\nGenerating dual PnL interactive chart...")
        create_plotly_chart_dual_pnl(
            virtual_positions_df, real_positions_df, 
            virtual_balance_history, real_balance_history, 
            df, "BTCUSDT", 
            real_pnl_history, filtered_real_pnl_history,
            virtual_pnl_history, filtered_virtual_pnl_history
        )
    else:
        print("\nNo data to visualize")
    
    return virtual_positions_df, real_positions_df, virtual_balance_history, real_balance_history, real_pnl_history, filtered_real_pnl_history, virtual_pnl_history, filtered_virtual_pnl_history


if __name__ == "__main__":
    import sys
    
    test_mode = len(sys.argv) > 1 and sys.argv[1] == '--test'
    main_numba(test_mode=test_mode)
