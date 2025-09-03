import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


def calculate_conformal_thresholds(df, test_size=0.2, alpha=0.1, class_specific=True):
    """
    Calculate conformal thresholds for prediction filtering.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions data
        test_size (float): Fraction of data to use for calibration
        alpha (float): Significance level (1-alpha = confidence level)
        class_specific (bool): Whether to calculate class-specific thresholds
        
    Returns:
        dict: Dictionary with conformal thresholds
    """
    # Ensure we have confidence and correctness columns
    if 'confidence' not in df.columns:
        df['confidence'] = df.apply(
            lambda row: row[f'prob_class_{int(row["predicted"])}'], axis=1
        )
    
    if 'is_correct' not in df.columns:
        df['is_correct'] = df['predicted'] == df['actual']
    
    # Split data into training and calibration sets
    train_df, cal_df = train_test_split(df, test_size=test_size, random_state=42)
    
    results = {}
    
    if class_specific:
        # Calculate class-specific thresholds
        for class_idx in sorted(df['predicted'].unique()):
            class_cal_df = cal_df[cal_df['predicted'] == class_idx]
            
            if len(class_cal_df) > 10:  # Ensure enough samples for reliable threshold
                # Calculate nonconformity scores (1 - confidence for correct predictions)
                correct_scores = 1 - class_cal_df[class_cal_df['is_correct']]['confidence']
                incorrect_scores = class_cal_df[~class_cal_df['is_correct']]['confidence']
                
                # Combine scores (higher score = more nonconforming)
                scores = pd.concat([correct_scores, incorrect_scores])
                
                if len(scores) > 0:
                    # Calculate threshold as (1-alpha) quantile
                    threshold = 1 - np.quantile(scores, alpha)
                    results[f'class_{class_idx}'] = threshold
                    
                    print(f"Class {class_idx} conformal threshold: {threshold:.4f}")
                    print(f"  - Samples in calibration set: {len(class_cal_df)}")
                    print(f"  - Expected accuracy at threshold: {1-alpha:.2f}")
                    
                    # Calculate actual accuracy at this threshold
                    filtered_df = train_df[(train_df['predicted'] == class_idx) & 
                                          (train_df['confidence'] >= threshold)]
                    if len(filtered_df) > 0:
                        actual_acc = filtered_df['is_correct'].mean()
                        print(f"  - Actual accuracy at threshold: {actual_acc:.4f}")
                        print(f"  - Coverage (% of predictions retained): {len(filtered_df)/len(train_df[train_df['predicted'] == class_idx]):.2%}")
    
    # Calculate overall threshold
    scores = 1 - cal_df[cal_df['is_correct']]['confidence']
    scores = pd.concat([scores, cal_df[~cal_df['is_correct']]['confidence']])
    
    if len(scores) > 0:
        overall_threshold = 1 - np.quantile(scores, alpha)
        results['overall'] = overall_threshold
        
        print(f"Overall conformal threshold: {overall_threshold:.4f}")
        
        # Calculate actual accuracy at this threshold
        filtered_df = train_df[train_df['confidence'] >= overall_threshold]
        if len(filtered_df) > 0:
            actual_acc = filtered_df['is_correct'].mean()
            print(f"  - Actual accuracy at threshold: {actual_acc:.4f}")
            print(f"  - Coverage (% of predictions retained): {len(filtered_df)/len(train_df):.2%}")
    
    return results


def apply_conformal_filtering(df, thresholds, output_file=None):
    """
    Apply conformal thresholds to filter predictions.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        thresholds (dict): Dictionary with conformal thresholds
        output_file (str): Path to save filtered predictions
        
    Returns:
        pd.DataFrame: DataFrame with filtered predictions
    """
    # Ensure we have confidence column
    if 'confidence' not in df.columns:
        df['confidence'] = df.apply(
            lambda row: row[f'prob_class_{int(row["predicted"])}'], axis=1
        )
    
    # Create a copy of the dataframe
    filtered_df = df.copy()
    
    # Add a column to indicate if prediction passes the conformal threshold
    filtered_df['passes_threshold'] = False
    
    # Apply class-specific thresholds if available
    for class_idx in df['predicted'].unique():
        class_key = f'class_{class_idx}'
        if class_key in thresholds:
            class_mask = (filtered_df['predicted'] == class_idx) & \
                         (filtered_df['confidence'] >= thresholds[class_key])
            filtered_df.loc[class_mask, 'passes_threshold'] = True
    
    # Apply overall threshold for any remaining predictions
    if 'overall' in thresholds:
        overall_mask = (~filtered_df['passes_threshold']) & \
                       (filtered_df['confidence'] >= thresholds['overall'])
        filtered_df.loc[overall_mask, 'passes_threshold'] = True
    
    # Filter predictions that pass the threshold
    result_df = filtered_df[filtered_df['passes_threshold']]
    
    # Print detailed statistics
    print("\n" + "="*50)
    print("CONFORMAL PREDICTION RESULTS SUMMARY")
    print("="*50)
    
    # Overall statistics
    total_predictions = len(df)
    filtered_predictions = len(df) - len(result_df)
    retained_predictions = len(result_df)
    retention_rate = retained_predictions / total_predictions
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total predictions: {total_predictions}")
    print(f"Filtered out: {filtered_predictions} ({(1 - retention_rate):.2%})")
    print(f"Retained: {retained_predictions} ({retention_rate:.2%})")
    
    # Accuracy comparison
    if 'is_correct' in result_df.columns:
        before_accuracy = df['is_correct'].mean()
        after_accuracy = result_df['is_correct'].mean()
        accuracy_improvement = after_accuracy - before_accuracy
        relative_improvement = (accuracy_improvement / before_accuracy) * 100
        
        print(f"\nACCURACY COMPARISON:")
        print(f"Before conformal filtering: {before_accuracy:.4f} ({total_predictions} predictions)")
        print(f"After conformal filtering:  {after_accuracy:.4f} ({retained_predictions} predictions)")
        print(f"Absolute improvement:      {accuracy_improvement:.4f} ({'+' if accuracy_improvement > 0 else ''}{accuracy_improvement:.4f})")
        print(f"Relative improvement:      {'+' if relative_improvement > 0 else ''}{relative_improvement:.2f}%")
    
    # Class-specific statistics
    print("\nCLASS-SPECIFIC STATISTICS:")
    for class_idx in sorted(df['predicted'].unique()):
        class_df = df[df['predicted'] == class_idx]
        class_result_df = result_df[result_df['predicted'] == class_idx]
        
        class_total = len(class_df)
        class_retained = len(class_result_df)
        class_retention_rate = class_retained / class_total if class_total > 0 else 0
        
        print(f"\nClass {class_idx}:")
        print(f"  Total predictions: {class_total}")
        print(f"  Retained: {class_retained} ({class_retention_rate:.2%})")
        
        if 'is_correct' in df.columns and class_total > 0:
            class_before_acc = class_df['is_correct'].mean()
            class_after_acc = class_result_df['is_correct'].mean() if class_retained > 0 else 0
            class_acc_improvement = class_after_acc - class_before_acc
            
            print(f"  Accuracy before: {class_before_acc:.4f}")
            print(f"  Accuracy after:  {class_after_acc:.4f} ({'+' if class_acc_improvement > 0 else ''}{class_acc_improvement:.4f})")
            
            # Show threshold used
            threshold_key = f'class_{class_idx}'
            if threshold_key in thresholds:
                print(f"  Threshold used:  {thresholds[threshold_key]:.4f}")
            elif 'overall' in thresholds:
                print(f"  Threshold used:  {thresholds['overall']:.4f} (overall)")
    
    print("\n" + "="*50)
    
    # Save to file if requested
    if output_file is not None:
        result_df.to_csv(output_file, index=False)
        print(f"Filtered predictions saved to {output_file}")
    
    return result_df


def visualize_conformal_thresholds(df, thresholds):
    """
    Visualize the effect of conformal thresholds on accuracy and coverage.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        thresholds (dict): Dictionary with conformal thresholds
    """
    # Ensure we have confidence and correctness columns
    if 'confidence' not in df.columns:
        df['confidence'] = df.apply(
            lambda row: row[f'prob_class_{int(row["predicted"])}'], axis=1
        )
    
    if 'is_correct' not in df.columns:
        df['is_correct'] = df['predicted'] == df['actual']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy vs Confidence Threshold
    thresholds_range = np.linspace(0.5, 1.0, 50)
    accuracies = []
    coverages = []
    
    for threshold in thresholds_range:
        filtered = df[df['confidence'] >= threshold]
        if len(filtered) > 0:
            accuracies.append(filtered['is_correct'].mean())
            coverages.append(len(filtered) / len(df))
        else:
            accuracies.append(np.nan)
            coverages.append(0)
    
    axes[0].plot(thresholds_range, accuracies, 'b-', linewidth=2)
    axes[0].set_xlabel('Confidence Threshold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Confidence Threshold')
    axes[0].grid(True)
    
    # Add vertical lines for the calculated thresholds
    for key, threshold in thresholds.items():
        if key == 'overall':
            axes[0].axvline(x=threshold, color='r', linestyle='--', 
                          label=f'Overall Threshold: {threshold:.3f}')
        else:
            axes[0].axvline(x=threshold, color='g', linestyle=':', 
                          label=f'{key} Threshold: {threshold:.3f}')
    
    axes[0].legend()
    
    # Plot 2: Coverage vs Confidence Threshold
    axes[1].plot(thresholds_range, coverages, 'g-', linewidth=2)
    axes[1].set_xlabel('Confidence Threshold')
    axes[1].set_ylabel('Coverage (% of predictions retained)')
    axes[1].set_title('Coverage vs Confidence Threshold')
    axes[1].grid(True)
    
    # Add vertical lines for the calculated thresholds
    for key, threshold in thresholds.items():
        if key == 'overall':
            axes[1].axvline(x=threshold, color='r', linestyle='--')
        else:
            axes[1].axvline(x=threshold, color='g', linestyle=':')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(df.name if hasattr(df, 'name') else '.'), 
                              'conformal_thresholds.png')
    plt.savefig(output_path)
    print(f"Conformal thresholds visualization saved to {output_path}")


def visualize_conformal_intervals(df, alpha_range=[0.01, 0.05, 0.1, 0.2, 0.3]):
    """
    Visualize the width of conformal intervals at different confidence levels.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        alpha_range (list): List of alpha values to calculate intervals for
    """
    print("\nVisualizing conformal interval widths...")
    
    # Ensure we have confidence and correctness columns
    if 'confidence' not in df.columns:
        df['confidence'] = df.apply(
            lambda row: row[f'prob_class_{int(row["predicted"])}'], axis=1
        )
    
    if 'is_correct' not in df.columns:
        df['is_correct'] = df['predicted'] == df['actual']
    
    # Sort by date if available
    if 'date' in df.columns:
        df = df.sort_values('date')
    
    # Split data for calibration
    train_size = int(0.7 * len(df))
    cal_size = int(0.15 * len(df))
    test_size = len(df) - train_size - cal_size
    
    train_df = df.iloc[:train_size]
    cal_df = df.iloc[train_size:train_size+cal_size]
    test_df = df.iloc[train_size+cal_size:]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Interval width over time
    interval_widths = {}
    interval_lower = {}
    interval_upper = {}
    
    # Calculate nonconformity scores on calibration set
    cal_scores = 1 - cal_df[cal_df['is_correct']]['confidence']
    cal_scores = pd.concat([cal_scores, cal_df[~cal_df['is_correct']]['confidence']])
    
    # Calculate thresholds for different alpha values
    for alpha in alpha_range:
        threshold = 1 - np.quantile(cal_scores, alpha)
        interval_widths[alpha] = []
        interval_lower[alpha] = []
        interval_upper[alpha] = []
        
        for idx, row in test_df.iterrows():
            # For simplicity, we'll use a symmetric interval around the predicted class probability
            # In a real implementation, you might want to calculate class-specific intervals
            pred_class = int(row['predicted'])
            pred_prob = row[f'prob_class_{pred_class}']
            
            # Calculate interval width based on threshold
            width = 2 * (1 - threshold)
            interval_widths[alpha].append(width)
            
            # Calculate lower and upper bounds
            lower = max(0, pred_prob - width/2)
            upper = min(1, pred_prob + width/2)
            interval_lower[alpha].append(lower)
            interval_upper[alpha].append(upper)
    
    # Plot interval widths over time
    x_values = range(len(test_df))
    for alpha in alpha_range:
        axes[0, 0].plot(x_values, interval_widths[alpha], label=f'α={alpha} (conf={(1-alpha)*100:.0f}%)')
    
    axes[0, 0].set_title('Conformal Interval Width Over Time')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Interval Width')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Interval width distribution
    for alpha in alpha_range:
        axes[0, 1].hist(interval_widths[alpha], bins=30, alpha=0.6, label=f'α={alpha} (conf={(1-alpha)*100:.0f}%)')
    
    axes[0, 1].set_title('Distribution of Conformal Interval Widths')
    axes[0, 1].set_xlabel('Interval Width')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Interval width vs confidence
    for alpha in alpha_range:
        axes[1, 0].scatter(test_df['confidence'], interval_widths[alpha], 
                         alpha=0.3, label=f'α={alpha} (conf={(1-alpha)*100:.0f}%)')
    
    axes[1, 0].set_title('Interval Width vs Prediction Confidence')
    axes[1, 0].set_xlabel('Prediction Confidence')
    axes[1, 0].set_ylabel('Interval Width')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Example of intervals for a specific alpha
    selected_alpha = 0.05  # 90% confidence level
    sample_size = min(100, len(test_df))  # Limit to 100 samples for clarity
    sample_indices = np.linspace(0, len(test_df)-1, sample_size, dtype=int)
    
    x_sample = range(sample_size)
    y_sample = [test_df.iloc[i]['confidence'] for i in sample_indices]
    lower_bound = [interval_lower[selected_alpha][i] for i in sample_indices]
    upper_bound = [interval_upper[selected_alpha][i] for i in sample_indices]
    
    axes[1, 1].plot(x_sample, y_sample, 'bo-', label='Predicted Confidence')
    axes[1, 1].fill_between(x_sample, lower_bound, upper_bound, color='blue', alpha=0.2, label=f'Conformal Interval (α={selected_alpha})')
    
    # Mark correct and incorrect predictions
    correct_indices = [i for i, idx in enumerate(sample_indices) if test_df.iloc[idx]['is_correct']]
    incorrect_indices = [i for i, idx in enumerate(sample_indices) if not test_df.iloc[idx]['is_correct']]
    
    if correct_indices:
        axes[1, 1].plot([x_sample[i] for i in correct_indices], 
                      [y_sample[i] for i in correct_indices], 
                      'go', label='Correct Prediction')
    
    if incorrect_indices:
        axes[1, 1].plot([x_sample[i] for i in incorrect_indices], 
                      [y_sample[i] for i in incorrect_indices], 
                      'rx', label='Incorrect Prediction')
    
    axes[1, 1].set_title(f'Conformal Intervals for {sample_size} Sample Predictions (90% Confidence)')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(df.name if hasattr(df, 'name') else '.'), 
                              'conformal_intervals.png')
    plt.savefig(output_path)
    print(f"Conformal intervals visualization saved to {output_path}")


if __name__ == "__main__":
    # Path to the predictions file
    predictions_file = "results/sol_new.csv"
    
    # Load the data
    print(f"Loading data from {predictions_file}...")
    df = pd.read_csv(predictions_file)
    
    # Convert date column to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Add confidence and correctness columns
    df['confidence'] = df.apply(
        lambda row: row[f'prob_class_{int(row["predicted"])}'], axis=1
    )
    df['is_correct'] = df['predicted'] == df['actual']
    
    # Calculate conformal thresholds
    print("\nCalculating conformal thresholds...")
    thresholds = calculate_conformal_thresholds(df, alpha=0.95, class_specific=True)
    
    # Visualize the effect of thresholds
    visualize_conformal_thresholds(df, thresholds)
    
    # Visualize conformal intervals
    visualize_conformal_intervals(df, alpha_range=[0.01, 0.05, 0.1, 0.2, 0.3])
    
    # Apply conformal filtering
    print("\nApplying conformal filtering...")
    output_file = os.path.join(os.path.dirname(predictions_file), "conformal_filtered.csv")
    filtered_df = apply_conformal_filtering(df, thresholds, output_file)
