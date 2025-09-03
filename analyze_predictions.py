import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime, timedelta
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def analyze_predictions(file_path, days=60):
    """
    Analyze prediction results and create a Plotly graph for the last specified days.
    
    Args:
        file_path (str): Path to the CSV file with predictions
        days (int): Number of days to display in the graph
    """
    # Load the data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date to ensure chronological order
    df = df.sort_values('date')
    
    # Calculate the cutoff date for the last month
    latest_date = df['date'].max()
    cutoff_date = latest_date - timedelta(days=days)
    
    # Filter data for the last month
    df_last_month = df[df['date'] >= cutoff_date]
    
    print(f"Analyzing data from {df_last_month['date'].min()} to {df_last_month['date'].max()}")
    print(f"Total data points: {len(df_last_month)}")
    
    # Add a column for confidence in the predicted class
    df_last_month['confidence'] = df_last_month.apply(
        lambda row: row[f'prob_class_{int(row["predicted"])}'], axis=1
    )
    
    # Add a column to indicate if prediction was correct
    df_last_month['is_correct'] = df_last_month['predicted'] == df_last_month['actual']
    
    # Calculate correlation between confidence and correctness
    correlation = analyze_confidence_correlation(df_last_month)
    
    # Create buy and sell signals dataframes
    sell_signals = df_last_month[df_last_month['predicted'] == 0]
    buy_signals = df_last_month[df_last_month['predicted'] == 2]
    
    print(f"Buy signals: {len(buy_signals)}")
    print(f"Sell signals: {len(sell_signals)}")
    
    # Calculate cumulative position
    # Map predictions to position changes: 0 (sell) -> -1, 2 (buy) -> +1, 1 (hold) -> 0
    df_last_month['position_change'] = df_last_month['predicted'].map({0: -1, 1: 0, 2: 1})
    
    # Calculate cumulative position
    df_last_month['cumulative_position'] = df_last_month['position_change'].cumsum()
    
    # Create the plot
    fig = go.Figure()
    
    # Add closing price line
    fig.add_trace(go.Scatter(
        x=df_last_month['date'],
        y=df_last_month['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=1),
        yaxis='y1'
    ))
    
    # Add buy signals
    fig.add_trace(go.Scatter(
        x=buy_signals['date'],
        y=buy_signals['Close'],
        mode='markers',
        name='Buy Signal (2)',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        yaxis='y1'
    ))
    
    # Add sell signals
    fig.add_trace(go.Scatter(
        x=sell_signals['date'],
        y=sell_signals['Close'],
        mode='markers',
        name='Sell Signal (0)',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        yaxis='y1'
    ))
    
    # Add cumulative position line on secondary y-axis
    fig.add_trace(go.Scatter(
        x=df_last_month['date'],
        y=df_last_month['cumulative_position'],
        mode='lines',
        name='Cumulative Position',
        line=dict(color='purple', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title=f'Price and Trading Signals - Last {days} Days',
        xaxis=dict(
            title='Date',
            domain=[0.05, 0.95]
        ),
        yaxis=dict(
            title='Price',
            side='left',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Cumulative Position',
            side='right',
            overlaying='y',
            titlefont=dict(color='purple'),
            tickfont=dict(color='purple'),
            showgrid=False
        ),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Show the plot
    fig.show()
    
    # Save the plot to HTML file
    output_file = os.path.join(os.path.dirname(file_path), 'prediction_analysis.html')
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")
    
    # Print some statistics about the cumulative position
    print(f"Final cumulative position: {df_last_month['cumulative_position'].iloc[-1]}")
    print(f"Max cumulative position: {df_last_month['cumulative_position'].max()}")
    print(f"Min cumulative position: {df_last_month['cumulative_position'].min()}")
    
    return fig

def analyze_confidence_correlation(df):
    """
    Analyze the correlation between prediction confidence and correctness.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions data
        
    Returns:
        dict: Dictionary with correlation statistics
    """
    print("\nAnalyzing correlation between confidence and correct predictions...")
    
    # Convert boolean to int for correlation calculation
    df['is_correct_int'] = df['is_correct'].astype(int)
    
    # Calculate correlation coefficient and p-value for overall confidence
    corr_coef, p_value = pearsonr(df['confidence'], df['is_correct_int'])
    print(f"Overall Pearson correlation coefficient: {corr_coef:.4f} (p-value: {p_value:.4f})")
    
    # Calculate average confidence for correct and incorrect predictions
    avg_conf_correct = df[df['is_correct']]['confidence'].mean()
    avg_conf_incorrect = df[~df['is_correct']]['confidence'].mean()
    print(f"Average confidence for correct predictions: {avg_conf_correct:.4f}")
    print(f"Average confidence for incorrect predictions: {avg_conf_incorrect:.4f}")
    
    # Calculate accuracy at different confidence thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    print("\nAccuracy at different confidence thresholds:")
    for threshold in thresholds:
        high_conf = df[df['confidence'] >= threshold]
        if len(high_conf) > 0:
            accuracy = high_conf['is_correct'].mean() * 100
            coverage = len(high_conf) / len(df) * 100
            print(f"  Threshold {threshold:.1f}: Accuracy {accuracy:.2f}%, Coverage {coverage:.2f}% ({len(high_conf)} samples)")
    
    # Analyze correlation for each class separately
    print("\nAnalyzing correlation between confidence and accuracy for each class:")
    
    # Create confidence bins for more detailed analysis
    bin_edges = [0, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
    bin_labels = ['0-55%', '55-65%', '65-75%', '75-85%', '85-95%', '95-100%']
    
    # Create figures for per-class confidence analysis
    fig_accuracy, axes_accuracy = plt.subplots(1, 3, figsize=(18, 6))
    fig_counts, axes_counts = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create a figure for confidence distribution by class
    plt.figure(figsize=(12, 8))
    
    # Dictionary to store results for each class
    class_results = {}
    
    for class_idx in [0, 1, 2]:
        # Create a copy of the dataframe for this class analysis
        class_data = df.copy()
        
        # Add a column indicating if this specific class was predicted
        class_data[f'predicted_class_{class_idx}'] = (class_data['predicted'] == class_idx).astype(int)
        
        # Add a column indicating if this specific class was the actual class
        class_data[f'actual_class_{class_idx}'] = (class_data['actual'] == class_idx).astype(int)
        
        # Add a column indicating if the prediction for this class was correct
        class_data[f'correct_class_{class_idx}'] = ((class_data['predicted'] == class_idx) & 
                                                  (class_data['actual'] == class_idx)).astype(int)
        
        # Calculate correlation between confidence for this class and correctness
        conf_col = f'prob_class_{class_idx}'
        corr_class, p_val_class = pearsonr(class_data[conf_col], class_data[f'correct_class_{class_idx}'])
        
        print(f"\nClass {class_idx}:")
        print(f"  Correlation between {conf_col} and correct predictions: {corr_class:.4f} (p-value: {p_val_class:.4f})")
        
        # Bin the confidence values for this class
        class_data[f'conf_bin_{class_idx}'] = pd.cut(class_data[conf_col], bins=bin_edges, labels=bin_labels)
        
        # Calculate accuracy within each confidence bin for this class
        bin_stats = []
        for bin_label in bin_labels:
            bin_data = class_data[class_data[f'conf_bin_{class_idx}'] == bin_label]
            if len(bin_data) > 0:
                # Count predictions where this class was predicted
                predictions_in_bin = len(bin_data[bin_data['predicted'] == class_idx])
                
                # Count correct predictions for this class
                correct_in_bin = len(bin_data[bin_data[f'correct_class_{class_idx}'] == 1])
                
                # Calculate accuracy if there were any predictions for this class in this bin
                if predictions_in_bin > 0:
                    accuracy_in_bin = correct_in_bin / predictions_in_bin * 100
                else:
                    accuracy_in_bin = 0
                
                bin_stats.append({
                    'bin': bin_label,
                    'total_samples': len(bin_data),
                    'predictions_for_class': predictions_in_bin,
                    'correct_predictions': correct_in_bin,
                    'accuracy': accuracy_in_bin
                })
        
        # Convert bin stats to DataFrame for easier analysis
        bin_df = pd.DataFrame(bin_stats)
        if not bin_df.empty:
            print(f"  Accuracy by confidence bin for Class {class_idx}:")
            for _, row in bin_df.iterrows():
                print(f"    Bin {row['bin']}: {row['accuracy']:.2f}% accuracy, {row['predictions_for_class']} predictions")
            
            # Plot accuracy vs confidence bin for this class
            if len(bin_df) > 1:
                # Plot accuracy graph
                axes_accuracy[class_idx].bar(bin_df['bin'], bin_df['accuracy'])
                axes_accuracy[class_idx].set_title(f'Class {class_idx} Accuracy by Confidence')
                axes_accuracy[class_idx].set_xlabel('Confidence Bin')
                axes_accuracy[class_idx].set_ylabel('Accuracy (%)')
                axes_accuracy[class_idx].set_ylim(0, 105)  # Set y-axis limit to 0-105%
                for i, v in enumerate(bin_df['accuracy']):
                    if not np.isnan(v):
                        axes_accuracy[class_idx].text(i, v + 2, f"{v:.1f}%", ha='center')
                
                # Plot prediction counts graph
                axes_counts[class_idx].bar(bin_df['bin'], bin_df['predictions_for_class'])
                axes_counts[class_idx].set_title(f'Class {class_idx} Prediction Counts by Confidence')
                axes_counts[class_idx].set_xlabel('Confidence Bin')
                axes_counts[class_idx].set_ylabel('Number of Predictions')
                for i, v in enumerate(bin_df['predictions_for_class']):
                    if v > 0:  # Only show text for non-zero values
                        axes_counts[class_idx].text(i, v + 1, str(v), ha='center')
        
        # Store results for this class
        class_results[class_idx] = {
            'correlation': corr_class,
            'p_value': p_val_class,
            'bin_stats': bin_stats if bin_stats else None
        }
    
    # Adjust layout and save the figures
    plt.tight_layout()
    
    # Save accuracy by confidence plot
    class_accuracy_plot_path = os.path.join(os.path.dirname(df.name if hasattr(df, 'name') else '.'), 'class_accuracy_by_confidence.png')
    fig_accuracy.tight_layout()
    fig_accuracy.savefig(class_accuracy_plot_path)
    print(f"\nClass accuracy by confidence plot saved to {class_accuracy_plot_path}")
    
    # Save prediction counts by confidence plot
    class_counts_plot_path = os.path.join(os.path.dirname(df.name if hasattr(df, 'name') else '.'), 'class_prediction_counts_by_confidence.png')
    fig_counts.tight_layout()
    fig_counts.savefig(class_counts_plot_path)
    print(f"Class prediction counts by confidence plot saved to {class_counts_plot_path}")
    
    # Create a figure for confidence distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='confidence', hue='is_correct', bins=20, kde=True)
    plt.title('Confidence Distribution for Correct vs Incorrect Predictions')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    
    # Save the figure
    confidence_plot_path = os.path.join(os.path.dirname(df.name if hasattr(df, 'name') else '.'), 'confidence_distribution.png')
    plt.savefig(confidence_plot_path)
    print(f"Confidence distribution plot saved to {confidence_plot_path}")
    
    # Create a scatter plot of confidence vs correctness
    plt.figure(figsize=(10, 6))
    plt.scatter(df['confidence'], df['is_correct_int'], alpha=0.1)
    plt.title('Confidence vs Correctness')
    plt.xlabel('Confidence')
    plt.ylabel('Correct (1) / Incorrect (0)')
    
    # Save the scatter plot
    scatter_plot_path = os.path.join(os.path.dirname(df.name if hasattr(df, 'name') else '.'), 'confidence_vs_correctness.png')
    plt.savefig(scatter_plot_path)
    print(f"Scatter plot saved to {scatter_plot_path}")
    
    # Create confusion matrices for each class
    print("\nConfusion matrix statistics by class:")
    for class_idx in [0, 1, 2]:
        class_df = df[df['actual'] == class_idx]
        if len(class_df) > 0:
            class_accuracy = (class_df['predicted'] == class_df['actual']).mean() * 100
            class_avg_conf = class_df['confidence'].mean()
            print(f"  Class {class_idx}: Accuracy {class_accuracy:.2f}%, Avg Confidence {class_avg_conf:.4f}, Samples: {len(class_df)}")
    
    # Return correlation statistics
    return {
        'overall_correlation': {
            'correlation_coefficient': corr_coef,
            'p_value': p_value,
            'avg_confidence_correct': avg_conf_correct,
            'avg_confidence_incorrect': avg_conf_incorrect
        },
        'class_correlations': class_results
    }


def filter_high_confidence_predictions(file_path, confidence_threshold=0.8, output_file=None):
    """
    Filter predictions with confidence higher than the specified threshold and save to a new file.
    
    Args:
        file_path (str): Path to the CSV file with predictions
        confidence_threshold (float): Minimum confidence threshold (default: 0.95)
        output_file (str): Path to save filtered predictions (default: None, auto-generated)
        
    Returns:
        pd.DataFrame: DataFrame with filtered high confidence predictions
    """
    print(f"\nFiltering predictions with confidence > {confidence_threshold}...")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Add a column for confidence in the predicted class
    df['confidence'] = df.apply(
        lambda row: row[f'prob_class_{int(row["predicted"])}'], axis=1
    )
    
    # Add a column to indicate if prediction was correct
    df['is_correct'] = df['predicted'] == df['actual']
    
    # Filter predictions with high confidence
    high_conf_df = df[df['confidence'] > confidence_threshold]
    
    print(f"Found {len(high_conf_df)} predictions with confidence > {confidence_threshold}")
    print(f"Accuracy of high confidence predictions: {high_conf_df['is_correct'].mean() * 100:.2f}%")
    
    # Count predictions by class
    class_counts = high_conf_df['predicted'].value_counts().to_dict()
    for class_idx in sorted(class_counts.keys()):
        print(f"  Class {class_idx}: {class_counts.get(class_idx, 0)} predictions")
    
    # Save to file if requested
    if output_file is None:
        # Create default output filename based on threshold
        threshold_str = str(confidence_threshold).replace('.', '_')
        output_file = os.path.join(os.path.dirname(file_path), f"high_conf_{threshold_str}.csv")
    
    high_conf_df.to_csv(output_file, index=False)
    print(f"High confidence predictions saved to {output_file}")
    
    return high_conf_df


if __name__ == "__main__":
    # Path to the predictions file
    predictions_file = "results/eth_new.csv"
    
    # Analyze predictions for the last 90 days
    analyze_predictions(predictions_file, days=150)
    
    # Filter and save high confidence predictions
    #filter_high_confidence_predictions(predictions_file, confidence_threshold=0.6)
    
    # If you want to only analyze the correlation without creating the full analysis plot
    # Uncomment the following lines:
    # df = pd.read_csv(predictions_file)
    # df['date'] = pd.to_datetime(df['date'])
    # df['confidence'] = df.apply(lambda row: row[f'prob_class_{int(row["predicted"])}'], axis=1)
    # df['is_correct'] = df['predicted'] == df['actual']
    # analyze_confidence_correlation(df)
