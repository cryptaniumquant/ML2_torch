"""
Momentum Trading Strategy with ADX Filter and ATR-based Trailing Stop Loss

Strategy Logic:
1. ADX Filter: Trade only when ADX > threshold (hourly timeframe)
2. Separate Long/Short Portfolios: Independent trailing stops for each direction
3. Volatility-based Trailing Stop: Dynamic adjustment based on volatility
4. Position Management: Add to positions on new signals in same direction
5. Exit: Close entire position when trailing stop is hit or ADX drops below threshold
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STRATEGY CONFIGURATION - All parameters in one place
# =============================================================================

class StrategyConfig:
    # Data files
    PRICE_FILE = "data/SOLUSDT_1m_20210101_to_20250820.csv"
    SIGNALS_FILE = "data/sol_new.csv"
    SIGNALS_START_DATE = '2025-03-04'
    
    # Strategy parameters
    INIT_DEPOSIT = 100000
    FIXED_TRADE_SIZE = 1000
    MAX_POSITIONS = 200  # Maximum number of open positions (leverage control)
    
    # ADX parameters
    ADX_PERIOD = 12  # ADX calculation period on hourly data
    ADX_THRESHOLD = 35  # Trade only when ADX > this value
    
    # Volatility parameters
    VOLATILITY_METRIC = 'atr'  # 'atr' or 'std'
    ATR_PERIOD = 12  # ATR calculation period on hourly data
    STD_PERIOD = 12  # Standard deviation calculation period on hourly data
    VOLATILITY_MIN = 0.7  # 0.5% - minimum observed volatility
    VOLATILITY_MAX = 2  # 2.5% - maximum observed volatility
    
    # Trailing stop parameters
    BASE_TRAILING_PCT = 0.05  # 5% - maximum trailing distance (low volatility)
    MIN_TRAILING_PCT = 0.01   # 1% - minimum trailing distance (high volatility)
    
    # Visualization parameters
    CHART_HEIGHT = 1000
    CHART_FILENAME = 'momentum_trailing_strategy.html'

# =============================================================================

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    def calculate_tr(high, low, close):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        return tr
    
    def calculate_dm(high, low):
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        return plus_dm, minus_dm
    
    tr = calculate_tr(high, low, close)
    plus_dm, minus_dm = calculate_dm(high, low)
    
    # Smooth TR and DM
    tr_smooth = pd.Series(tr).ewm(span=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period).mean().values
    
    # Calculate DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.nan_to_num(dx)
    
    adx = pd.Series(dx).ewm(span=period).mean().values
    
    return adx

def calculate_atr_normalized(high, low, close, period=14):
    """Calculate normalized ATR as percentage of price"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr = pd.Series(tr).rolling(window=period).mean().values
    # Normalize ATR as percentage of close price
    atr_normalized = (atr / close) * 100
    return atr_normalized

def calculate_volatility_std(close, period=14):
    """Calculate volatility using standard deviation of returns"""
    returns = pd.Series(close).pct_change().dropna()
    volatility = returns.rolling(window=period).std().values * 100  # As percentage
    # Pad with NaN to match original length
    volatility_padded = np.full(len(close), np.nan)
    volatility_padded[period:] = volatility[period-1:]
    return volatility_padded

def load_signals_data(signals_file):
    """Load and parse signals data"""
    signals_dict = {}
    
    try:
        signals_df = pd.read_csv(signals_file)
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        
        for _, row in signals_df.iterrows():
            date_key = row['date']
            # Convert predicted signal: 0->-1, 1->0, 2->1
            predicted_signal = row['predicted']
            if predicted_signal == 0:
                signal = -1
            elif predicted_signal == 1:
                signal = 0
            elif predicted_signal == 2:
                signal = 1
            else:
                signal = 0
            
            signals_dict[date_key] = signal
            
    except Exception as e:
        print(f"Error loading signals: {e}")
        return {}
    
    return signals_dict

def calculate_trailing_stop(current_price, position_type, volatility_value, base_pct=StrategyConfig.BASE_TRAILING_PCT, min_pct=StrategyConfig.MIN_TRAILING_PCT, volatility_min=StrategyConfig.VOLATILITY_MIN, volatility_max=StrategyConfig.VOLATILITY_MAX):
    """
    Calculate trailing stop distance based on volatility using linear interpolation
    
    Args:
        current_price: Current market price
        position_type: 'long' or 'short'
        volatility_value: Current volatility value (normalized ATR or std dev %)
        base_pct: Base trailing percentage (5%) - maximum trailing distance at low volatility
        min_pct: Minimum trailing percentage (1%) - minimum trailing distance at high volatility
        volatility_min: Minimum volatility level (0.5%)
        volatility_max: Maximum volatility level (2.5%)
    
    Returns:
        Trailing stop price
    
    Formula: trailing_pct = base_pct - (base_pct - min_pct) * (volatility - volatility_min) / (volatility_max - volatility_min)
    """
    # Clamp volatility to observed range
    volatility_clamped = max(volatility_min, min(volatility_value, volatility_max))
    
    # Linear interpolation: higher volatility = tighter trailing stop (lower percentage)
    volatility_ratio = (volatility_clamped - volatility_min) / (volatility_max - volatility_min)
    trailing_pct = base_pct - (base_pct - min_pct) * volatility_ratio
    
    # Calculate trailing stop distance
    trailing_distance = current_price * trailing_pct
    
    if position_type == 'long':
        return current_price - trailing_distance
    else:  # short
        return current_price + trailing_distance

class MomentumTrailingStrategy:
    def __init__(self, config=StrategyConfig):
        self.init_deposit = config.INIT_DEPOSIT
        self.fixed_trade_size = config.FIXED_TRADE_SIZE
        self.max_positions = config.MAX_POSITIONS
        self.adx_threshold = config.ADX_THRESHOLD
        self.volatility_metric = config.VOLATILITY_METRIC
        self.base_trailing_pct = config.BASE_TRAILING_PCT
        
        # Separate portfolios for long and short positions
        self.long_positions = []
        self.short_positions = []
        
        # Trailing stops
        self.long_trailing_stop = None
        self.short_trailing_stop = None
        
        # Balance tracking
        self.balance = self.init_deposit
        self.balance_history = []
        
        # Trade history
        self.trades_history = []
        
    def update_trailing_stops(self, current_price, volatility_value, current_date):
        """Update trailing stops for both long and short positions"""
        
        # Update long trailing stop
        if self.long_positions:
            avg_long_price = np.mean([pos['entry_price'] for pos in self.long_positions])
            
            # Only update if price moved in favorable direction
            if current_price > avg_long_price:
                new_long_stop = calculate_trailing_stop(current_price, 'long', volatility_value)
                
                # Only move trailing stop up (never down for longs)
                if self.long_trailing_stop is None or new_long_stop > self.long_trailing_stop:
                    self.long_trailing_stop = new_long_stop
                    print(f"  Updated LONG trailing stop: ${self.long_trailing_stop:.2f}")
        
        # Update short trailing stop
        if self.short_positions:
            avg_short_price = np.mean([pos['entry_price'] for pos in self.short_positions])
            
            # Only update if price moved in favorable direction
            if current_price < avg_short_price:
                new_short_stop = calculate_trailing_stop(current_price, 'short', volatility_value)
                
                # Only move trailing stop down (never up for shorts)
                if self.short_trailing_stop is None or new_short_stop < self.short_trailing_stop:
                    self.short_trailing_stop = new_short_stop
                    print(f"  Updated SHORT trailing stop: ${self.short_trailing_stop:.2f}")
    
    def check_trailing_stops(self, high_price, low_price, close_price, current_date):
        """Check if trailing stops are hit using OHLC data for accurate execution"""
        
        # Check long trailing stop - use low price for more accurate stop execution
        if self.long_positions and self.long_trailing_stop and low_price <= self.long_trailing_stop:
            # Exit at trailing stop price or low price, whichever is higher
            exit_price = max(self.long_trailing_stop, low_price)
            self.close_all_positions('long', exit_price, current_date, 'trailing_stop')
            
        # Check short trailing stop - use high price for more accurate stop execution
        if self.short_positions and self.short_trailing_stop and high_price >= self.short_trailing_stop:
            # Exit at trailing stop price or high price, whichever is lower
            exit_price = min(self.short_trailing_stop, high_price)
            self.close_all_positions('short', exit_price, current_date, 'trailing_stop')
    
    def add_position(self, signal, current_price, current_date, volatility_value):
        """Add a new position based on signal"""
        # Check position limit
        total_positions = len(self.long_positions) + len(self.short_positions)
        if total_positions >= self.max_positions:
            print(f"  Position limit reached ({total_positions}/{self.max_positions}) - skipping signal")
            return
            
        if signal == 1:  # Long signal
            position = {
                'entry_price': current_price,
                'entry_date': current_date,
                'size': self.fixed_trade_size / current_price,
                'value': self.fixed_trade_size
            }
            self.long_positions.append(position)
            print(f"  Added LONG position: ${current_price:.2f}, Size: {position['size']:.4f} ({len(self.long_positions)+len(self.short_positions)}/{self.max_positions})")
            
        elif signal == -1:  # Short signal
            position = {
                'entry_price': current_price,
                'entry_date': current_date,
                'size': self.fixed_trade_size / current_price,
                'value': self.fixed_trade_size
            }
            self.short_positions.append(position)
            print(f"  Added SHORT position: ${current_price:.2f}, Size: {position['size']:.4f} ({len(self.long_positions)+len(self.short_positions)}/{self.max_positions})")
            
        # Initialize or update trailing stop
        if signal == 1:  # Long signal
            if not self.long_trailing_stop:
                self.long_trailing_stop = calculate_trailing_stop(current_price, 'long', volatility_value)
                
        elif signal == -1:  # Short signal
            if not self.short_trailing_stop:
                self.short_trailing_stop = calculate_trailing_stop(current_price, 'short', volatility_value)
    
    def close_all_positions(self, position_type, exit_price, exit_date, exit_reason):
        """Close all positions of given type"""
        
        if position_type == 'long' and self.long_positions:
            total_pnl = 0
            total_value = 0
            
            for pos in self.long_positions:
                pnl = (exit_price - pos['entry_price']) * pos['size']
                total_pnl += pnl
                total_value += pos['value']
                
                # Record trade
                self.trades_history.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': exit_date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'size': pos['size'],
                    'position_type': 'long',
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })
            
            self.balance += total_pnl
            self.long_positions = []
            self.long_trailing_stop = None
            
            print(f"  Closed ALL LONG positions | Exit: ${exit_price:.2f} | Total PnL: ${total_pnl:.2f} | Reason: {exit_reason}")
            
        elif position_type == 'short' and self.short_positions:
            total_pnl = 0
            total_value = 0
            
            for pos in self.short_positions:
                pnl = (pos['entry_price'] - exit_price) * pos['size']
                total_pnl += pnl
                total_value += pos['value']
                
                # Record trade
                self.trades_history.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': exit_date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'size': pos['size'],
                    'position_type': 'short',
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })
            
            self.balance += total_pnl
            self.short_positions = []
            self.short_trailing_stop = None
            
            print(f"  Closed ALL SHORT positions | Exit: ${exit_price:.2f} | Total PnL: ${total_pnl:.2f} | Reason: {exit_reason}")
    
    def get_current_exposure(self):
        """Get current exposure for both long and short positions"""
        long_exposure = sum([pos['value'] for pos in self.long_positions])
        short_exposure = sum([pos['value'] for pos in self.short_positions])
        return long_exposure, short_exposure
    
    def update_balance_history(self, current_date, current_price):
        """Update balance history with current PnL"""
        # Calculate current unrealized PnL for long positions
        long_unrealized_pnl = sum([
            (current_price - pos['entry_price']) * pos['size'] 
            for pos in self.long_positions
        ])
        
        # Calculate current unrealized PnL for short positions
        short_unrealized_pnl = sum([
            (pos['entry_price'] - current_price) * pos['size'] 
            for pos in self.short_positions
        ])
        
        total_unrealized_pnl = long_unrealized_pnl + short_unrealized_pnl
        
        # Calculate realized PnL
        realized_pnl = sum([trade['pnl'] for trade in self.trades_history])
        
        # Calculate net position (positive = net long, negative = net short)
        long_exposure = sum([pos['size'] for pos in self.long_positions])
        short_exposure = sum([pos['size'] for pos in self.short_positions])
        net_position = long_exposure - short_exposure
        
        self.balance_history.append({
            'date': current_date,
            'total_balance': self.balance + total_unrealized_pnl,
            'cash_balance': self.balance,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'long_unrealized_pnl': long_unrealized_pnl,
            'short_unrealized_pnl': short_unrealized_pnl,
            'long_positions': len(self.long_positions),
            'short_positions': len(self.short_positions),
            'net_position': net_position,
            'long_trailing_stop': self.long_trailing_stop,
            'short_trailing_stop': self.short_trailing_stop,
            'price': current_price
        })

def create_momentum_trailing_chart(df, strategy, symbol, config):
    """Create comprehensive chart for momentum trailing strategy"""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f'{symbol} Price & Trailing Stops',
            'ADX & Volatility Indicators',
            'Portfolio Performance (%)',
            'Net Position & Price'
        ),
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": True}]
        ]
    )
    
    # 1. Price chart with trailing stops
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            name='Price',
            line=dict(color='black', width=1)
        ),
        row=1, col=1, secondary_y=False
    )
    
    # Add trailing stops from balance history
    balance_df = pd.DataFrame(strategy.balance_history)
    if not balance_df.empty:
        # Long trailing stops
        long_stops = balance_df['long_trailing_stop'].dropna()
        if not long_stops.empty:
            fig.add_trace(
                go.Scatter(
                    x=balance_df[balance_df['long_trailing_stop'].notna()]['date'],
                    y=long_stops,
                    name='Long Trailing Stop',
                    line=dict(color='green', width=2, dash='dash'),
                    mode='lines'
                ),
                row=1, col=1, secondary_y=False
            )
        
        # Short trailing stops
        short_stops = balance_df['short_trailing_stop'].dropna()
        if not short_stops.empty:
            fig.add_trace(
                go.Scatter(
                    x=balance_df[balance_df['short_trailing_stop'].notna()]['date'],
                    y=short_stops,
                    name='Short Trailing Stop',
                    line=dict(color='red', width=2, dash='dash'),
                    mode='lines'
                ),
                row=1, col=1, secondary_y=False
            )
    
    # Add trade markers
    if strategy.trades_history:
        trades_df = pd.DataFrame(strategy.trades_history)
        
        # Entry markers
        for _, trade in trades_df.iterrows():
            color = 'green' if trade['position_type'] == 'long' else 'red'
            symbol_marker = 'triangle-up' if trade['position_type'] == 'long' else 'triangle-down'
            
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_date']],
                    y=[trade['entry_price']],
                    mode='markers',
                    marker=dict(
                        symbol=symbol_marker,
                        size=10,
                        color=color,
                        line=dict(width=1, color='white')
                    ),
                    name=f"{trade['position_type'].title()} Entry",
                    showlegend=False,
                    hovertemplate=f"<b>Entry</b><br>Date: %{{x}}<br>Price: $%{{y:.2f}}<br>Type: {trade['position_type'].title()}<extra></extra>"
                ),
                row=1, col=1, secondary_y=False
            )
            
            # Exit markers
            exit_color = 'darkgreen' if trade['pnl'] > 0 else 'darkred'
            fig.add_trace(
                go.Scatter(
                    x=[trade['exit_date']],
                    y=[trade['exit_price']],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=8,
                        color=exit_color,
                        line=dict(width=2, color='white')
                    ),
                    name=f"{trade['position_type'].title()} Exit",
                    showlegend=False,
                    hovertemplate=f"<b>Exit</b><br>Date: %{{x}}<br>Price: $%{{y:.2f}}<br>PnL: ${trade['pnl']:.2f}<br>Reason: {trade['exit_reason']}<extra></extra>"
                ),
                row=1, col=1, secondary_y=False
            )
    
    # 2. ADX and ATR indicators
    if 'ADX' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['ADX'],
                name='ADX',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1, secondary_y=False
        )
        
        # ADX threshold line
        fig.add_hline(y=config.ADX_THRESHOLD, line_dash="dash", line_color="purple", opacity=0.7, 
                      row=2, col=1, annotation_text=f"ADX Threshold ({config.ADX_THRESHOLD})")
    
    if 'Volatility' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Volatility'],
                name='Volatility (%)',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1, secondary_y=True
        )
    
    # 3. Portfolio performance in percentage
    if strategy.balance_history:
        balance_df = pd.DataFrame(strategy.balance_history)
        
        # Portfolio performance traces (in percentage)
        total_pnl_pct = ((balance_df['total_balance'] - strategy.init_deposit) / strategy.init_deposit) * 100
        realized_pnl_pct = (balance_df['realized_pnl'] / strategy.init_deposit) * 100
        long_unrealized_pnl_pct = (balance_df['long_unrealized_pnl'] / strategy.init_deposit) * 100
        short_unrealized_pnl_pct = (balance_df['short_unrealized_pnl'] / strategy.init_deposit) * 100
        
        # Total PnL (realized + unrealized) in %
        fig.add_trace(
            go.Scatter(
                x=balance_df['date'],
                y=total_pnl_pct,
                name='Total PnL (%)',
                line=dict(color='green', width=3)
            ),
            row=3, col=1
        )
        
        # Realized PnL in %
        fig.add_trace(
            go.Scatter(
                x=balance_df['date'],
                y=realized_pnl_pct,
                name='Realized PnL (%)',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        
        # Unrealized PnL components in %
        fig.add_trace(
            go.Scatter(
                x=balance_df['date'],
                y=long_unrealized_pnl_pct,
                name='Long Unrealized PnL (%)',
                line=dict(color='lightgreen', width=1)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=balance_df['date'],
                y=short_unrealized_pnl_pct,
                name='Short Unrealized PnL (%)',
                line=dict(color='lightcoral', width=1)
            ),
            row=3, col=1
        )
        
        # Net Position and Price in new row
        fig.add_trace(
            go.Scatter(
                x=balance_df['date'],
                y=balance_df['net_position'],
                name='Net Position',
                line=dict(color='purple', width=2),
                mode='lines'
            ),
            row=4, col=1, secondary_y=False
        )
        
        # Price on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=balance_df['date'],
                y=balance_df['price'],
                name='Price',
                line=dict(color='black', width=1, dash='dot'),
                opacity=0.7
            ),
            row=4, col=1, secondary_y=True
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Momentum Trailing Stop Strategy (ADX>{config.ADX_THRESHOLD}, {config.VOLATILITY_METRIC.upper()})',
        height=config.CHART_HEIGHT,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="ADX", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="PnL (%)", row=3, col=1)
    fig.update_yaxes(title_text="Net Position", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Price ($)", row=4, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # Save and show
    fig.write_html(config.CHART_FILENAME)
    print(f"\nMomentum trailing strategy chart saved as {config.CHART_FILENAME}")
    fig.show()
    
    return fig

def main():
    # Load configuration
    config = StrategyConfig()
    
    print("Loading data...")
    df = pd.read_csv(config.PRICE_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Filter data to match signal timeframe
    signals_start_date = pd.to_datetime(config.SIGNALS_START_DATE)
    df = df[df['Date'] >= signals_start_date].reset_index(drop=True)
    
    print(f"Price data loaded: {len(df)} rows from {df['Date'].min()} to {df['Date'].max()}")
    
    # Load signals
    print("Loading signals data...")
    signals_dict = load_signals_data(config.SIGNALS_FILE)
    print(f"Signals loaded: {len(signals_dict)} signals")
    
    # Resample to hourly data for ADX and ATR calculation
    print("Calculating indicators...")
    df_hourly = df.set_index('Date').resample('1H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Calculate ADX
    adx_values = calculate_adx(df_hourly['High'].values, df_hourly['Low'].values, 
                              df_hourly['Close'].values, period=config.ADX_PERIOD)
    
    # Calculate normalized ATR and volatility
    atr_normalized = calculate_atr_normalized(df_hourly['High'].values, df_hourly['Low'].values, 
                                            df_hourly['Close'].values, period=config.ATR_PERIOD)
    volatility_std = calculate_volatility_std(df_hourly['Close'].values, period=config.STD_PERIOD)
    
    # Create indicator DataFrames
    adx_df = pd.DataFrame({
        'Date': df_hourly.index,
        'ADX': adx_values
    }).dropna()
    
    volatility_df = pd.DataFrame({
        'Date': df_hourly.index,
        'ATR_normalized': atr_normalized,
        'Volatility_std': volatility_std
    }).dropna()
    
    # Merge indicators back to minute data
    df = df.merge(adx_df, on='Date', how='left')
    df = df.merge(volatility_df, on='Date', how='left')
    df['ADX'] = df['ADX'].fillna(method='ffill')
    df['ATR_normalized'] = df['ATR_normalized'].fillna(method='ffill')
    df['Volatility_std'] = df['Volatility_std'].fillna(method='ffill')
    
    # Select volatility metric based on configuration
    if config.VOLATILITY_METRIC == 'atr':
        df['Volatility'] = df['ATR_normalized'].fillna(config.VOLATILITY_MIN)
        print(f"Using ATR as volatility metric")
    else:  # 'std'
        df['Volatility'] = df['Volatility_std'].fillna(df['ATR_normalized']).fillna(config.VOLATILITY_MIN)
        print(f"Using Standard Deviation as volatility metric")
    
    # Initialize strategy
    strategy = MomentumTrailingStrategy(config)
    
    print(f"\n=== STRATEGY CONFIGURATION ===")
    print(f"ADX Period: {config.ADX_PERIOD}")
    print(f"ADX Threshold: {config.ADX_THRESHOLD}")
    print(f"Initial Deposit: ${config.INIT_DEPOSIT:,}")
    print(f"Fixed Trade Size: ${config.FIXED_TRADE_SIZE}")
    print(f"Max Positions: {config.MAX_POSITIONS}")
    print(f"Volatility Metric: {config.VOLATILITY_METRIC.upper()}")
    print(f"Volatility Range: {config.VOLATILITY_MIN}% - {config.VOLATILITY_MAX}%")
    print(f"Base Trailing Stop: {config.BASE_TRAILING_PCT*100:.1f}%")
    print(f"Min Trailing Stop: {config.MIN_TRAILING_PCT*100:.1f}%")
    print(f"\nStarting backtest...")
    
    # Run backtest
    for i, row in df.iterrows():
        current_date = row['Date']
        current_price = row['Close']
        high_price = row['High']
        low_price = row['Low']
        current_adx = row['ADX'] if pd.notna(row['ADX']) else 0
        current_volatility = row['Volatility'] if pd.notna(row['Volatility']) else 2.0
        
        # Check for signal
        signal = signals_dict.get(current_date, 0)
        
        # Update balance history
        strategy.update_balance_history(current_date, current_price)
        
        # Check trailing stops first using OHLC data
        strategy.check_trailing_stops(high_price, low_price, current_price, current_date)
        
        # Check if ADX is below threshold - close all positions
        if current_adx <= config.ADX_THRESHOLD:
            if strategy.long_positions or strategy.short_positions:
                print(f"  ADX below threshold ({current_adx:.1f} <= {config.ADX_THRESHOLD}) - closing all positions")
                strategy.close_all_positions('long', current_price, current_date, 'adx_exit')
                strategy.close_all_positions('short', current_price, current_date, 'adx_exit')
        
        # Only trade if ADX is above threshold
        if current_adx > config.ADX_THRESHOLD and signal != 0:
            print(f"  Signal: {signal}, ADX: {current_adx:.1f}, Volatility: {current_volatility:.2f}%")
            strategy.add_position(signal, current_price, current_date, current_volatility)
        
        # Update trailing stops
        strategy.update_trailing_stops(current_price, current_volatility, current_date)
        
        # Progress indicator
        if i % 10000 == 0:
            print(f"Processed {i:,} candles...")
    
    # Final results
    print(f"\n=== BACKTEST RESULTS ===")
    print(f"Initial Deposit: ${strategy.init_deposit:,}")
    print(f"Final Balance: ${strategy.balance:,.2f}")
    print(f"Total Return: {((strategy.balance / strategy.init_deposit) - 1) * 100:.2f}%")
    print(f"Total Trades: {len(strategy.trades_history)}")
    
    if strategy.trades_history:
        trades_df = pd.DataFrame(strategy.trades_history)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)")
        print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")
        print(f"Average Win: ${winning_trades['pnl'].mean():.2f}" if len(winning_trades) > 0 else "Average Win: $0.00")
        print(f"Average Loss: ${losing_trades['pnl'].mean():.2f}" if len(losing_trades) > 0 else "Average Loss: $0.00")
        print(f"Largest Win: ${trades_df['pnl'].max():.2f}")
        print(f"Largest Loss: ${trades_df['pnl'].min():.2f}")
    
    # Create visualization
    print("\nGenerating interactive chart...")
    create_momentum_trailing_chart(df, strategy, "SOLUSDT", config)
    
    print("\nBacktest completed!")

if __name__ == "__main__":
    main()
