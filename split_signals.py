import pandas as pd
import os
from datetime import datetime

def format_date(date_str):
    """Convert datetime string to the format shown in the image."""
    dt = pd.to_datetime(date_str)
    return dt.strftime('%d.%m.%Y %H:%M')

def process_signals(input_file, buy_output_file, sell_output_file):
    """
    Process the main.csv file and split into buy and sell signal files.
    
    Args:
        input_file: Path to the input CSV file
        buy_output_file: Path to the output file for buy signals (predicted=2)
        sell_output_file: Path to the output file for sell signals (predicted=0)
    """
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Format the date column
    df['formatted_date'] = df['date'].apply(format_date)
    
    # Select columns needed for output
    columns = ['formatted_date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Create buy signals dataframe (predicted=2)
    buy_df = df[df['predicted'] == 2].copy()
    if not buy_df.empty:
        # Add 'long' and 'TRUE' columns
        buy_df['long'] = 1.0
        buy_df['TRUE'] = 1.0
        
        # Select and reorder columns for output
        buy_output = buy_df[columns + ['long', 'TRUE']]
        
        # Rename the date column
        buy_output = buy_output.rename(columns={'formatted_date': 'Date'})
        
        # Save to CSV
        buy_output.to_csv(buy_output_file, sep=';', index=False)
        print(f"Buy signals saved to {buy_output_file}")
    else:
        print("No buy signals found")
    
    # Create sell signals dataframe (predicted=0)
    sell_df = df[df['predicted'] == 0].copy()
    if not sell_df.empty:
        # Add 'short' and 'TRUE' columns
        sell_df['short'] = 1.0
        sell_df['TRUE'] = 1.0
        
        # Select and reorder columns for output
        sell_output = sell_df[columns + ['short', 'TRUE']]
        
        # Rename the date column
        sell_output = sell_output.rename(columns={'formatted_date': 'Date'})
        
        # Save to CSV
        sell_output.to_csv(sell_output_file, sep=';', index=False)
        print(f"Sell signals saved to {sell_output_file}")
    else:
        print("No sell signals found")

if __name__ == "__main__":
    # Define file paths
    input_file = "results/dot_signals_agg_15min.csv"
    buy_output_file = "results/dot_buy_signals_15min.csv"
    sell_output_file = "results/dot_sell_signals_15min.csv"
    
    # Make sure the results directory exists
    os.makedirs(os.path.dirname(buy_output_file), exist_ok=True)
    
    # Process the signals
    process_signals(input_file, buy_output_file, sell_output_file)
    
    print("Processing complete!")
