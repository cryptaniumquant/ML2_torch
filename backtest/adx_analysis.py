import pandas as pd
import numpy as np
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def load_signals_data(signals_file_path):
    """Load signals data from CSV file"""
    df = pd.read_csv(signals_file_path)
    df['date'] = pd.to_datetime(df['date'])
    signals_dict = dict(zip(df['date'], df['predicted'].astype(int)))
    print(f"Loaded {len(signals_dict)} signals from {signals_file_path}")
    return signals_dict

def calculate_adx(df, period=24):
    """Calculate ADX using TA-Lib"""
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    adx = talib.ADX(high, low, close, timeperiod=period)
    plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
    minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)
    
    return adx, plus_di, minus_di

def resample_ohlc(df, timeframe):
    """Resample 1-minute data to specified timeframe"""
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy = df_copy.set_index('Date')
    
    # Resample to specified timeframe
    resampled = df_copy.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Reset index to get Date as column
    resampled = resampled.reset_index()
    
    return resampled

def get_signal_for_datetime(target_datetime, signals_dict):
    """Get signal for specific datetime"""
    return signals_dict.get(target_datetime, None)

def create_adx_visualization():
    """Create comprehensive ADX analysis with multiple timeframes"""
    
    # Load price data
    price_file = "data/ETHUSDT_1m_20210107_to_20250826.csv"
    signals_file = "data/eth_new.csv"
    
    print("Loading price data...")
    df = pd.read_csv(price_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Filter data to match signal timeframe
    signals_start_date = pd.to_datetime('2025-03-10')
    df = df[df['Date'] >= signals_start_date].reset_index(drop=True)
    
    # Take recent data for better visualization (last 30 days)
    recent_days = 120
    recent_data = df.tail(recent_days * 1440).copy()  # 1440 minutes per day
    
    print(f"Using data from {recent_data['Date'].min()} to {recent_data['Date'].max()}")
    
    # Load signals
    print("Loading signals...")
    signals_dict = load_signals_data(signals_file)
    
    # Resample to different timeframes
    print("Resampling data to different timeframes...")
    df_1h = resample_ohlc(recent_data, '1H')
    df_30m = resample_ohlc(recent_data, '30T')
    df_4h = resample_ohlc(recent_data, '4H')
    
    # Calculate ADX for each timeframe
    print("Calculating ADX indicators...")
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(df_1h, period=24)
    adx_30m, plus_di_30m, minus_di_30m = calculate_adx(df_30m, period=24)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(df_4h, period=24)
    
    # Add ADX data to dataframes
    df_1h['ADX'] = adx_1h
    df_1h['PLUS_DI'] = plus_di_1h
    df_1h['MINUS_DI'] = minus_di_1h
    
    df_30m['ADX'] = adx_30m
    df_30m['PLUS_DI'] = plus_di_30m
    df_30m['MINUS_DI'] = minus_di_30m
    
    df_4h['ADX'] = adx_4h
    df_4h['PLUS_DI'] = plus_di_4h
    df_4h['MINUS_DI'] = minus_di_4h
    
    # Get signals for the timeframe
    signal_data = []
    for _, row in recent_data.iterrows():
        signal = get_signal_for_datetime(row['Date'], signals_dict)
        if signal is not None and signal in [0, 2]:  # Only buy/sell signals
            signal_data.append({
                'date': row['Date'],
                'price': row['Close'],
                'signal': signal,
                'signal_name': 'BUY' if signal == 2 else 'SELL'
            })
    
    signals_df = pd.DataFrame(signal_data)
    
    # Create subplots
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            'Price & Trading Signals',
            '1H ADX (Period 24)',
            '30M ADX (Period 24)', 
            '4H ADX (Period 24)',
            'DI+ vs DI- (1H)'
        ],
        row_heights=[0.3, 0.2, 0.2, 0.2, 0.1]
    )
    
    # 1. Price chart with signals
    fig.add_trace(
        go.Scatter(
            x=recent_data['Date'],
            y=recent_data['Close'],
            name='BTCUSDT Price',
            line=dict(color='blue', width=1),
            yaxis='y'
        ),
        row=1, col=1
    )
    
    # Add buy signals (green triangles up)
    if not signals_df.empty:
        buy_signals = signals_df[signals_df['signal'] == 2]
        sell_signals = signals_df[signals_df['signal'] == 0]
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['date'],
                    y=buy_signals['price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='green', size=10),
                    name='BUY Signals',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['date'],
                    y=sell_signals['price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', color='red', size=10),
                    name='SELL Signals',
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # 2. 1H ADX
    fig.add_trace(
        go.Scatter(
            x=df_1h['Date'],
            y=df_1h['ADX'],
            name='ADX 1H',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128,0,128,0.1)'
        ),
        row=2, col=1
    )
    
    # Add ADX threshold lines
    fig.add_hline(y=25, line_dash="dash", line_color="orange", opacity=0.7, row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    
    # 3. 30M ADX
    fig.add_trace(
        go.Scatter(
            x=df_30m['Date'],
            y=df_30m['ADX'],
            name='ADX 30M',
            line=dict(color='darkgreen', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,100,0,0.1)'
        ),
        row=3, col=1
    )
    
    fig.add_hline(y=25, line_dash="dash", line_color="orange", opacity=0.7, row=3, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.7, row=3, col=1)
    
    # 4. 15M ADX
    fig.add_trace(
        go.Scatter(
            x=df_4h['Date'],
            y=df_4h['ADX'],
            name='ADX 4H',
            line=dict(color='darkorange', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,140,0,0.1)'
        ),
        row=4, col=1
    )
    
    fig.add_hline(y=25, line_dash="dash", line_color="orange", opacity=0.7, row=4, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.7, row=4, col=1)
    
    # 5. DI+ vs DI- (1H timeframe)
    fig.add_trace(
        go.Scatter(
            x=df_1h['Date'],
            y=df_1h['PLUS_DI'],
            name='DI+ (1H)',
            line=dict(color='green', width=2)
        ),
        row=5, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_1h['Date'],
            y=df_1h['MINUS_DI'],
            name='DI- (1H)',
            line=dict(color='red', width=2)
        ),
        row=5, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='BTCUSDT: Multi-Timeframe ADX Analysis with Trading Signals',
        height=1200,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="ADX Value", row=2, col=1)
    fig.update_yaxes(title_text="ADX Value", row=3, col=1)
    fig.update_yaxes(title_text="ADX Value", row=4, col=1)
    fig.update_yaxes(title_text="DI Value", row=5, col=1)
    fig.update_xaxes(title_text="Date", row=5, col=1)
    
    # Save and show
    filename = 'adx_multi_timeframe_analysis.html'
    fig.write_html(filename)
    print(f"\nADX analysis chart saved as {filename}")
    
    # Print some statistics
    print(f"\n=== ADX STATISTICS ===")
    print(f"1H ADX - Mean: {np.nanmean(df_1h['ADX']):.2f}, Max: {np.nanmax(df_1h['ADX']):.2f}")
    print(f"30M ADX - Mean: {np.nanmean(df_30m['ADX']):.2f}, Max: {np.nanmax(df_30m['ADX']):.2f}")
    print(f"4H ADX - Mean: {np.nanmean(df_4h['ADX']):.2f}, Max: {np.nanmax(df_4h['ADX']):.2f}")
    
    if not signals_df.empty:
        print(f"\n=== SIGNALS STATISTICS ===")
        print(f"Total signals: {len(signals_df)}")
        print(f"Buy signals: {len(signals_df[signals_df['signal'] == 2])}")
        print(f"Sell signals: {len(signals_df[signals_df['signal'] == 0])}")
    
    fig.show()
    
    return fig

if __name__ == "__main__":
    create_adx_visualization()
