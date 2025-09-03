import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from scipy.linalg import lstsq
from scipy.special import gammaln

# Global constants
WINDOW_SIZE = 120
POLY_ORDER = 3
STD_WINDOW_SIZE = 30
CAPITAL = 100_000  # Base capital for percentage calculations

# Distance thresholds for different signal types (in percentages)
DISTANCE_THRESHOLDS = {
    'type1': 2.0,  # 3% of capital ($3000 / $100,000)
    'type2': 4.0,  # 4% of capital ($4000 / $100,000)
    'type3': 6.0,  # 5% of capital ($5000 / $100,000)
    'type4': 10.0   # 6% of capital ($6000 / $100,000)
}




def detect_multi_type_signals(distance_metric_pct, first_derivative, dates, total_pnl):
    """
    Detect 4 types of signals with independent conditions:
    Type 1: Distance > 3.0%, independent derivative condition
    Type 2: Distance > 4.0%, independent derivative condition  
    Type 3: Distance > 5.0%, independent derivative condition
    Type 4: Distance > 6.0%, independent derivative condition
    """
    # Initialize results for each signal type
    signals = {
        'type1': {'indices': [], 'dates': [], 'pnl_values': [], 'distances': []},
        'type2': {'indices': [], 'dates': [], 'pnl_values': [], 'distances': []},
        'type3': {'indices': [], 'dates': [], 'pnl_values': [], 'distances': []},
        'type4': {'indices': [], 'dates': [], 'pnl_values': [], 'distances': []}
    }
    
    # Track state for each signal type independently
    signal_states = {
        'type1': {'last_signal_index': -1, 'waiting_for_derivative_change': False},
        'type2': {'last_signal_index': -1, 'waiting_for_derivative_change': False},
        'type3': {'last_signal_index': -1, 'waiting_for_derivative_change': False},
        'type4': {'last_signal_index': -1, 'waiting_for_derivative_change': False}
    }
    
    for i in range(1, len(distance_metric_pct)):
        # Skip NaN values
        if np.isnan(distance_metric_pct[i]) or np.isnan(first_derivative[i]):
            continue
        
        # Check each signal type independently
        for signal_type, threshold in DISTANCE_THRESHOLDS.items():
            state = signal_states[signal_type]
            
            # Condition 1: Distance crosses threshold for this type
            if distance_metric_pct[i] > threshold:
                
                # If this is the first signal or derivative condition is met
                if state['last_signal_index'] == -1 or not state['waiting_for_derivative_change']:
                    signals[signal_type]['indices'].append(i)
                    signals[signal_type]['dates'].append(dates.iloc[i])
                    signals[signal_type]['pnl_values'].append(total_pnl[i])
                    signals[signal_type]['distances'].append(distance_metric_pct[i])
                    
                    print(f"{signal_type.upper()} SIGNAL detected at index {i}, date {dates.iloc[i]}, PnL: ${total_pnl[i]:.2f}, Distance: {distance_metric_pct[i]:.2f}%")
                    
                    state['last_signal_index'] = i
                    state['waiting_for_derivative_change'] = True
            
            # Check for derivative sign change (positive to negative) for this type
            if state['waiting_for_derivative_change'] and i > state['last_signal_index']:
                if first_derivative[i-1] > 0 and first_derivative[i] < 0:
                    state['waiting_for_derivative_change'] = False
                    print(f"  {signal_type.upper()} derivative condition reset at index {i}, date {dates.iloc[i]}")
                    print(f"  First derivative: {first_derivative[i-1]:.4f} -> {first_derivative[i]:.4f}")
    
    # Print summary for each type
    for signal_type in signals:
        count = len(signals[signal_type]['indices'])
        threshold = DISTANCE_THRESHOLDS[signal_type]
        print(f"\n{signal_type.upper()} signals (threshold {threshold}%): {count}")
    
    return signals

def causal_savgol_filter(data, window_length, polyorder):
    """
    Causal Savitzky-Golay filter that only uses past data
    """
    filtered_data = np.full_like(data, np.nan, dtype=float)
    
    for i in range(len(data)):
        # Use only past data up to current point
        start_idx = max(0, i - window_length + 1)
        end_idx = i + 1
        
        if end_idx - start_idx < polyorder + 1:
            # Not enough points for polynomial fit
            if i > 0:
                filtered_data[i] = filtered_data[i-1]  # Use previous value
            else:
                filtered_data[i] = data[i]  # Use original value
            continue
            
        # Extract window data
        window_data = data[start_idx:end_idx]
        window_indices = np.arange(len(window_data))
        
        # Create Vandermonde matrix for polynomial fitting
        A = np.vander(window_indices, polyorder + 1, increasing=True)
        
        # Solve least squares to get polynomial coefficients
        try:
            coeffs, _, _, _ = lstsq(A, window_data, rcond=None)
            # Evaluate polynomial at the last point (current point)
            filtered_data[i] = np.polyval(coeffs[::-1], len(window_data) - 1)
        except:
            # Fallback to simple average if fitting fails
            filtered_data[i] = np.mean(window_data)
    
    return filtered_data

def create_clean_savgol_analysis():
    """Create clean SavGol analysis with properly isolated panels"""
    
    # Load PnL data
    print("Loading PnL data...")
    df = pd.read_csv('pnl_data_for_savgol_eth.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df)} data points")
    print(f"Full date range: {df['date'].min()} to {df['date'].max()}")
    
    # Limit to last 3 months for faster processing
    end_date = df['date'].max()
    start_date = end_date - pd.Timedelta(days=200)  # Last 3 months
    
    # Filter data
    df_filtered = df[df['date'] >= start_date].copy()
    print(f"Using last 3 months: {len(df_filtered)} data points")
    print(f"Filtered date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")
    
    # Extract data
    dates = df_filtered['date']
    total_pnl = df_filtered['total_pnl'].values
    
    # Apply SavGol filter using global constants
    window = WINDOW_SIZE
    polyorder = POLY_ORDER
    if len(total_pnl) < window:
        print("Not enough data for filtering")
        return None
    
    # Apply CAUSAL SavGol filter - only uses past data
    print("Applying causal SavGol filter (no look-ahead)...")
    filtered_pnl = causal_savgol_filter(total_pnl, window, polyorder)
    
    # Test causal filter with spike test
    test_signal = np.zeros(200)
    test_signal[150] = 1000  # Spike at position 150
    test_filtered_causal = causal_savgol_filter(test_signal, WINDOW_SIZE, POLY_ORDER)
    
    # Check if positions BEFORE the spike are affected
    before_spike = test_filtered_causal[140:150]  # 10 points before spike
    after_spike = test_filtered_causal[151:161]   # 10 points after spike
    
    print(f"Causal filter - values before spike: {before_spike}")
    print(f"Max value before spike: {np.max(np.abs(before_spike)):.6f}")
    
    if np.max(np.abs(before_spike)) < 0.001:
        print("SUCCESS: Causal SavGol filter has no look-ahead bias")
    else:
        print("WARNING: Still some look-ahead detected")
    
    # Calculate derivatives of FILTERED PnL only
    first_derivative = np.gradient(filtered_pnl)
    second_derivative = np.gradient(first_derivative)
    
    # Debug: print derivative ranges and verify calculation source
    print(f"Filtered PnL range: {np.min(filtered_pnl):.2f} to {np.max(filtered_pnl):.2f}")
    print(f"1st derivative range: {np.min(first_derivative):.6f} to {np.max(first_derivative):.6f}")
    print(f"2nd derivative range: {np.min(second_derivative):.8f} to {np.max(second_derivative):.8f}")
    
    # For comparison, calculate derivatives of original PnL
    orig_first_deriv = np.gradient(total_pnl)
    orig_second_deriv = np.gradient(orig_first_deriv)
    print(f"Original 1st derivative range: {np.min(orig_first_deriv):.6f} to {np.max(orig_first_deriv):.6f}")
    print(f"Original 2nd derivative range: {np.min(orig_second_deriv):.8f} to {np.max(orig_second_deriv):.8f}")
    
    # Verify we're using the correct derivatives
    print(f"\nVERIFICATION:")
    print(f"Are we plotting derivatives of FILTERED PnL? {np.array_equal(first_derivative, np.gradient(filtered_pnl))}")
    print(f"Are we plotting derivatives of ORIGINAL PnL? {np.array_equal(first_derivative, orig_first_deriv)}")
    
    # Show sample values to confirm
    print(f"\nSample comparison (first 5 values):")
    print(f"Filtered PnL: {filtered_pnl[:5]}")
    print(f"Original PnL: {total_pnl[:5]}")
    print(f"1st deriv (filtered): {first_derivative[:5]}")
    print(f"1st deriv (original): {orig_first_deriv[:5]}")
    
    # Convert PnL values to percentages
    total_pnl_pct = (total_pnl / CAPITAL) * 100
    filtered_pnl_pct = (filtered_pnl / CAPITAL) * 100
    
    # Calculate absolute difference (how much real PnL is above/below smoothed) in percentages
    pnl_absolute_diff_pct = total_pnl_pct - filtered_pnl_pct
    
    # Calculate Euclidean distance only when real PnL is above smoothed PnL (in percentages)
    positive_diff_mask = pnl_absolute_diff_pct > 0
    pnl_euclidean_distance_pct = np.where(positive_diff_mask, pnl_absolute_diff_pct, 0)
    
    # Signal detection algorithm using distance metric with multiple types
    signal_points = detect_multi_type_signals(pnl_euclidean_distance_pct, first_derivative, dates, total_pnl)
    
    # Calculate statistics for the distance metric (only positive cases) in percentages
    positive_distances_pct = pnl_absolute_diff_pct[positive_diff_mask]
    if len(positive_distances_pct) > 0:
        avg_distance_pct = np.mean(positive_distances_pct)
        max_distance_pct = np.max(positive_distances_pct)
        std_distance_pct = np.std(positive_distances_pct)
        positive_count = len(positive_distances_pct)
        positive_percentage = (positive_count / len(pnl_absolute_diff_pct)) * 100
    else:
        avg_distance_pct = max_distance_pct = std_distance_pct = 0
        positive_count = 0
        positive_percentage = 0
    
    # Statistics for absolute difference in percentages
    avg_abs_diff_pct = np.mean(np.abs(pnl_absolute_diff_pct))
    max_abs_diff_pct = np.max(np.abs(pnl_absolute_diff_pct))
    std_abs_diff_pct = np.std(pnl_absolute_diff_pct)
    
    print(f"\nPnL Euclidean Distance Statistics (Real PnL > Smoothed only):")
    print(f"Average distance: {avg_distance_pct:.2f}%")
    print(f"Maximum distance: {max_distance_pct:.2f}%")
    print(f"Standard deviation of distance: {std_distance_pct:.2f}%")
    print(f"Positive cases: {positive_count}/{len(pnl_absolute_diff_pct)} ({positive_percentage:.1f}%)")
    print(f"\nPnL Absolute Difference Statistics:")
    print(f"Average absolute difference: {avg_abs_diff_pct:.2f}%")
    print(f"Maximum absolute difference: {max_abs_diff_pct:.2f}%")
    print(f"Standard deviation of difference: {std_abs_diff_pct:.2f}%")
    
    # Create figure with 3 subplots and secondary y-axis for panel 3
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=[
            f'Original PnL vs SavGol ({WINDOW_SIZE}, {POLY_ORDER}) Filter', 
            'First Derivative of SavGol Filter (Velocity)',
            'PnL Distance: When Real > Smoothed & All Differences'
        ],
        row_heights=[0.4, 0.3, 0.3],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}], 
               [{"secondary_y": True}]]
    )
    
    # Panel 1: PnL comparison (in percentages)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=total_pnl_pct,
            name='Original PnL (%)',
            line=dict(color='blue', width=1.5),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add signal points with different colors for each type
    signal_colors = {
        'type1': 'blue',    # 3% threshold
        'type2': 'red',     # 4% threshold  
        'type3': 'orange',  # 5% threshold
        'type4': 'purple'   # 6% threshold
    }
    
    signal_symbols = {
        'type1': 'circle',
        'type2': 'square', 
        'type3': 'diamond',
        'type4': 'triangle-up'
    }
    
    for signal_type, signal_data in signal_points.items():
        if signal_data['indices']:
            threshold = DISTANCE_THRESHOLDS[signal_type]
            # Convert PnL values to percentages for display
            pnl_values_pct = [(pnl / CAPITAL) * 100 for pnl in signal_data['pnl_values']]
            
            fig.add_trace(
                go.Scatter(
                    x=signal_data['dates'],
                    y=pnl_values_pct,
                    mode='markers',
                    name=f'{signal_type.upper()} ({threshold}%)',
                    marker=dict(
                        color=signal_colors[signal_type], 
                        size=10, 
                        symbol=signal_symbols[signal_type],
                        line=dict(width=1, color='black')
                    ),
                    hovertemplate=f'<b>{signal_type.upper()} Signal</b><br>Date: %{{x}}<br>PnL: %{{y:.2f}}%<br>Distance: %{{customdata:.2f}}%<extra></extra>',
                    customdata=signal_data['distances'],
                    showlegend=True
                ),
                row=1, col=1
            )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=filtered_pnl_pct,
            name=f'SavGol ({WINDOW_SIZE}, {POLY_ORDER}) (%)',
            line=dict(color='green', width=2.5)
        ),
        row=1, col=1
    )
    
    
    # Panel 2: First derivative ONLY (from filtered PnL)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=first_derivative,
            name='1st Derivative of SavGol',
            line=dict(color='red', width=2),
            hovertemplate='<b>1st Derivative (SavGol)</b><br>Date: %{x}<br>Velocity: %{y:.4f}<extra></extra>',
            showlegend=True
        ),
        row=2, col=1
    )
    
    # Panel 3: Euclidean distance only when real PnL > smoothed PnL (in percentages)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pnl_euclidean_distance_pct,
            name='Distance (Real > Smoothed) (%)',
            line=dict(color='orange', width=2),
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.3)',
            hovertemplate='<b>Distance</b><br>Date: %{x}<br>Distance: %{y:.2f}%<extra></extra>',
            showlegend=True
        ),
        row=3, col=1
    )
    
    # Add absolute difference line (how much real PnL is above smoothed PnL) in percentages
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pnl_absolute_diff_pct,
            name='All Differences (Real - Smoothed) (%)',
            line=dict(color='red', width=1.5, dash='dash'),
            hovertemplate='<b>Difference</b><br>Date: %{x}<br>Difference: %{y:.2f}%<extra></extra>',
            showlegend=True
        ),
        row=3, col=1, secondary_y=False
    )
    
    # Add zero line for reference in distance panel
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    
    # Update layout
    # Update y-axis titles
    fig.update_yaxes(title_text="PnL (%)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity", row=2, col=1)
    fig.update_yaxes(title_text="Distance (Real > Smoothed) (%)", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="All Differences (%)", row=3, col=1, secondary_y=True)
    
    # Update x-axis titles
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # Save the chart
    filename = 'savgol_distance_analysis.html'
    fig.write_html(filename)
    print(f"\nClean analysis saved as {filename}")
    
    # Print statistics
    noise_reduction = (1 - np.std(filtered_pnl) / np.std(total_pnl)) * 100
    final_diff = filtered_pnl[-1] - total_pnl[-1]
    max_first_deriv = np.max(np.abs(first_derivative))
    max_second_deriv = np.max(np.abs(second_derivative))
    
    print(f"\nSavGol ({WINDOW_SIZE}, {POLY_ORDER}) Analysis with Distance-Based Signals:")
    print("=" * 70)
    print(f"Noise Reduction:              {noise_reduction:.2f}%")
    print(f"Final Value Diff:             ${final_diff:.2f}")
    print(f"Max |1st Derivative|:         {max_first_deriv:.4f}")
    print(f"Max |2nd Derivative|:         {max_second_deriv:.6f}")
    print(f"Distance Thresholds:          Type1={DISTANCE_THRESHOLDS['type1']}%, Type2={DISTANCE_THRESHOLDS['type2']}%, Type3={DISTANCE_THRESHOLDS['type3']}%, Type4={DISTANCE_THRESHOLDS['type4']}%")
    print(f"Distance (Real > Smoothed):")
    print(f"  Average Distance:           {avg_distance_pct:.2f}%")
    print(f"  Max Distance:               {max_distance_pct:.2f}%")
    print(f"  Std Dev:                    {std_distance_pct:.2f}%")
    print(f"  Positive Cases:             {positive_count}/{len(pnl_absolute_diff_pct)} ({positive_percentage:.1f}%)")
    print(f"All Differences:")
    print(f"  Average Absolute Diff:      {avg_abs_diff_pct:.2f}%")
    print(f"  Max Absolute Diff:          {max_abs_diff_pct:.2f}%")
    print(f"  Absolute Diff Std Dev:      {std_abs_diff_pct:.2f}%")
    
    return fig

def main():
    """Main function"""
    print("Creating clean SavGol analysis...")
    
    fig = create_clean_savgol_analysis()
    
    if fig:
        print("\nClean analysis complete!")
        print("File created: savgol_distance_analysis.html")
    else:
        print("Error: Could not create analysis")

if __name__ == "__main__":
    main()
