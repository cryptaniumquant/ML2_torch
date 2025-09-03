import numpy as np
import torch
import time
import pickle
import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import os
import pathlib
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Configure CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Found GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found. Running on CPU.")

from architectures_pytorch.helpers.constants import hyperparameters, coin_list, direction, threshold, selected_model
from architectures_pytorch.vision_transformer import get_vit_model

# Import CCT model if available
try:
    from architectures_pytorch.compact_conv_transformer import get_cct_model
    CCT_AVAILABLE = True
except ImportError:
    CCT_AVAILABLE = False

# Import ViT-Small model if available
try:
    from architectures_pytorch.vit_small_dataset import get_vit_small_model
    VIT_SMALL_AVAILABLE = True
except ImportError:
    VIT_SMALL_AVAILABLE = False

# Narrow hyperparameters dictionary to the currently selected model
hyperparameters = hyperparameters[selected_model]

from train_torch import make_datasets, evaluate_model

def load_and_prepare_data(val_size=0.01, test_size=0.99, random_state=42, filter_step=None, use_all_as_test=False):
    print("Loading data...")
    # Load feature matrices
    matrices = np.load('data/eth.npy')
    print(f"Loaded matrices with shape: {matrices.shape}, dtype: {matrices.dtype}")
    
    # Load labels for 3-class classification
    labels_data = np.load('data/eth.npz')
    print(f"Available labels: {list(labels_data.keys())}")
    
    # Load 3-class labels
    labels = labels_data['class_labels']
    dates = labels_data['dates']
    
    # Print class distribution
    unique_values, counts = np.unique(labels, return_counts=True)
    print("Class distribution:")
    for value, count in zip(unique_values, counts):
        print(f"  Class {int(value)}: {count} samples ({count/len(labels)*100:.2f}%)")
    print(f"Selected {direction} labels with shape: {labels.shape}")
    
    # Apply filter to select every Nth element if filter_step is provided
    if filter_step is not None and filter_step > 1:
        print(f"Applying filter to select every {filter_step}th element")
        matrices = matrices[::filter_step]
        labels = labels[::filter_step]
        dates = dates[::filter_step]
        print(f"After filtering: matrices shape: {matrices.shape}, labels shape: {labels.shape}")
    
    # Handle the case where we want to use all data as test data
    if use_all_as_test:
        print("Using all data as test data")
        x_train = np.empty((0, *matrices.shape[1:]), dtype=matrices.dtype)
        y_train = np.empty(0, dtype=labels.dtype)
        dates_train = np.empty(0, dtype=dates.dtype)
        
        x_val = np.empty((0, *matrices.shape[1:]), dtype=matrices.dtype)
        y_val = np.empty(0, dtype=labels.dtype)
        dates_val = np.empty(0, dtype=dates.dtype)
        
        x_test = matrices
        y_test = labels
        dates_test = dates
    else:
        # Split data into train, validation, and test sets
        # First split into train and temp
        x_train, x_temp, y_train, y_temp, dates_train, dates_temp = train_test_split(
            matrices, labels, dates, 
            test_size=(val_size + test_size), 
            random_state=random_state, 
            shuffle=False
        )
        
        # Then split temp into validation and test
        relative_test_size = test_size / (val_size + test_size)
        x_val, x_test, y_val, y_test, dates_val, dates_test = train_test_split(
            x_temp, y_temp, dates_temp,
            test_size=relative_test_size,
            random_state=random_state,
            shuffle=False
        )
    
    print(f"Train set: {x_train.shape}, Validation set: {x_val.shape}, Test set: {x_test.shape}")
    
    # Print class distribution for each dataset
    def print_class_distribution(y, dataset_name):
        unique_classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        print(f"\n{dataset_name} class distribution:")
        for cls, count in zip(unique_classes, counts):
            percentage = (count / total) * 100
            print(f"  Class {cls}: {count} samples ({percentage:.2f}%)")
    
    print_class_distribution(y_train, "Training")
    print_class_distribution(y_val, "Validation")
    print_class_distribution(y_test, "Test")
    
    return x_train, y_train, x_val, y_val, x_test, y_test, dates_train, dates_val, dates_test

def load_model(model_path, model_type=None):
    """
    Load a saved PyTorch model based on selected_model from constants.py
    If model_type is provided, it overrides selected_model (useful for ad-hoc loading).
    """
    model_type = model_type if model_type else selected_model
    """
    Load a saved PyTorch model
    
    Args:
        model_path: Path to the saved model weights
        model_type: Type of model to load (default: "vision_transformer")
        
    Returns:
        Loaded PyTorch model
    """
    print(f"Loading model from {model_path} ({model_type}) ...")

    # Instantiate appropriate architecture
    if model_type == "cct" and CCT_AVAILABLE:
        model = get_cct_model(load_weights=False, weights_path=None)
    elif model_type == "vit_small" and VIT_SMALL_AVAILABLE:
        model = get_vit_small_model(load_weights=False, weights_path=None)
    else:
        # Fallback to ViT
        model = get_vit_model(load_weights=False, weights_path=None)

    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model

def predict_dataset(model, test_loader):
    """
    Make predictions on a dataset
    
    Args:
        model: PyTorch model
        test_loader: DataLoader containing test data
        
    Returns:
        Tuple of (predictions, actual labels, probabilities)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device).float()  # ensure float32 to match model weights
            labels = labels.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Convert to float32 before converting to numpy to avoid BFloat16 error
            all_preds.extend(predicted.cpu().float().numpy())
            all_labels.extend(labels.cpu().float().numpy())
            all_probs.extend(probabilities.cpu().float().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def save_predictions(predictions, actual, dates, probabilities, model_name, direction_type):
    """
    Save predictions to a file
    
    Args:
        predictions: Model predictions
        actual: Actual labels
        dates: Dates corresponding to predictions
        probabilities: Prediction probabilities
        model_name: Name of the model
        direction_type: Direction type (long/short)
    """
    # Create results directory if it doesn't exist
    pathlib.Path('results').mkdir(exist_ok=True)
    
    # Create a DataFrame for easier analysis
    probabilities_df = pd.DataFrame({
        'date': dates,
        'actual': actual,
        'predicted': predictions,
        'prob_class_0': probabilities[:, 0],
        'prob_class_1': probabilities[:, 1],
        'prob_class_2': probabilities[:, 2]
    })
    
    # Load OHLCV data
    try:
        print("Loading OHLCV data...")
        ohlcv_df = pd.read_csv('data/ETHUSDT_1m_20210107_to_20250826.csv')
        
        # Convert date columns to datetime for merging
        ohlcv_df['Date'] = pd.to_datetime(ohlcv_df['Date'])
        probabilities_df['date'] = pd.to_datetime(probabilities_df['date'])
        
        # Merge prediction results with OHLCV data based on date
        print("Merging prediction results with OHLCV data...")
        merged_df = pd.merge(probabilities_df, ohlcv_df, left_on='date', right_on='Date', how='left')
        
        # Drop duplicate date column
        if 'Date' in merged_df.columns:
            merged_df = merged_df.drop('Date', axis=1)
        
        # Use the merged dataframe for saving
        probabilities_df = merged_df
        print(f"Successfully added OHLCV data to predictions. Final columns: {probabilities_df.columns.tolist()}")
    except Exception as e:
        print(f"Warning: Could not add OHLCV data to predictions: {str(e)}")
    
    # Generate timestamp for filenames
    timestamp = int(time.time())
    
    # Save as CSV
    csv_path = f'results/{model_name}_{direction_type}_predictions_{timestamp}.csv'
    probabilities_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")
    
    # Also save as pickle for compatibility with existing code
    results = {
        'predictions': predictions,
        'actual': actual,
        'dates': dates,
        'probabilities': probabilities
    }
    
    pickle_path = f'results/{model_name}_{direction_type}_predictions_{timestamp}.pickle'
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Predictions also saved to {pickle_path}")
    
    return probabilities_df

def analyze_predictions(predictions, actual):
    """
    Analyze prediction results
    
    Args:
        predictions: Model predictions
        actual: Actual labels
    """
    accuracy = np.mean(predictions == actual)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Calculate per-class accuracy
    for class_idx in range(3):  # Assuming 3 classes (0, 1, 2)
        class_mask = np.array(actual) == class_idx
        if np.sum(class_mask) > 0:  # Avoid division by zero
            class_acc = np.mean(np.array(predictions)[class_mask] == class_idx)
            print(f"  Class {class_idx} accuracy: {class_acc:.4f} (samples: {np.sum(class_mask)})")
    
    print("\nClassification Report:")
    target_names = [f"Class {i}" for i in range(3)]
    print(classification_report(actual, predictions, target_names=target_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(actual, predictions)
    print(cm)
    
    # Calculate F1 score
    f1 = f1_score(actual, predictions, average='weighted')
    print(f"\nF1 Score (weighted): {f1:.4f}")
    
    return accuracy, f1

def predict_last_20_percent(model_path, filter_step=None, direction_type=None, output_dir='results'):
    """
    Predict the last 20% of the dataset (validation + test sets)
    
    Args:
        model_path: Path to the saved model weights
        filter_step: Filter step for data selection
        direction_type: Direction type (long/short)
        output_dir: Directory to save results
        
    Returns:
        DataFrame with predictions
    """
    # Use provided direction or default from constants
    direction_type = direction_type if direction_type else direction
    
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Load the model
    model = load_model(model_path)
    
    # Load and prepare data
    print("Loading and preparing data...")
    x_train, y_train, x_val, y_val, x_test, y_test, dates_train, dates_val, dates_test = load_and_prepare_data(
        filter_step=filter_step
    )
    
    # Combine validation and test sets (last 20% of data)
    x_last_20 = np.concatenate([x_val, x_test])
    y_last_20 = np.concatenate([y_val, y_test])
    dates_last_20 = np.concatenate([dates_val, dates_test])
    
    print(f"Combined validation and test sets: {x_last_20.shape} samples")
    
    # Create dataset for the last 20%
    last_20_dataset = make_datasets(x_last_20, y_last_20)
    
    # Make predictions
    print("Making predictions...")
    predictions, actual, probabilities = predict_dataset(model, last_20_dataset)
    
    # Save predictions
    results_df = save_predictions(
        predictions, 
        actual, 
        dates_last_20, 
        probabilities, 
        selected_model, 
        direction_type
    )
    
    # Analyze predictions
    analyze_predictions(predictions, actual)
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    
    return results_df

def predict_test_only(model_path, filter_step=None, direction_type=None, output_dir='results', test_size=0.99, random_state=42, use_all_as_test=False):
    """
    Predict only the test dataset, formatted the same way as in train_torch.py
    
    Args:
        model_path: Path to the saved model weights
        filter_step: Filter step for data selection
        direction_type: Direction type (long/short)
        output_dir: Directory to save results
        test_size: Size of the test set (default: 0.1)
        random_state: Random state for reproducibility (default: 42)
        
    Returns:
        DataFrame with predictions
    """
    # Use provided direction or default from constants
    direction_type = direction_type if direction_type else direction
    
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Load the model
    model = load_model(model_path)
    
    # Load and prepare data
    print("Loading and preparing data...")
    x_train, y_train, x_val, y_val, x_test, y_test, dates_train, dates_val, dates_test = load_and_prepare_data(
        val_size=0.01,
        test_size=test_size,
        random_state=random_state,
        filter_step=filter_step,
        use_all_as_test=use_all_as_test
    )
    
    print(f"Test set: {x_test.shape} samples")
    
    # Create dataset for the test set only
    test_dataset = make_datasets(x_test, y_test)
    
    # Make predictions
    print("Making predictions on test set only...")
    predictions, actual, probabilities = predict_dataset(model, test_dataset)
    
    # Save predictions
    results_df = save_predictions(
        predictions, 
        actual, 
        dates_test, 
        probabilities, 
        selected_model, 
        direction_type
    )
    
    # Analyze predictions
    analyze_predictions(predictions, actual)
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    
    return results_df

def main():
    # Configuration values hardcoded in the script
    model_path = 'models/eth_new/vision_transformer_3_epoch_150_1756375947.pth'  # Path to the saved model weights
    filter_step = 1  # Filter step for data selection
    output_dir = 'results'  # Directory to save results
    test_size = 0.1  # Size of the test set (not used when use_all_as_test=True)
    random_state = 42  # Random state for reproducibility
    use_all_as_test = False  # Use all data as test data
    
    # Call the prediction function for test set only
    predict_test_only(
        model_path=model_path,
        filter_step=filter_step,
        output_dir=output_dir,
        test_size=test_size,
        random_state=random_state,
        use_all_as_test=use_all_as_test
    )

if __name__ == '__main__':
    main()
