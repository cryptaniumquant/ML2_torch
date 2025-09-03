from tkinter import TRUE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
import time
import pickle
import sys
import os
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import os
import wandb
from tqdm import tqdm
import pathlib
from architectures_pytorch.helpers.wandb_handler import initialize_wandb

# Configure logging to file
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_filename = f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
log_path = os.path.join(log_dir, log_filename)

# Create a class to redirect stdout to both console and file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to our Logger class
sys.stdout = Logger(log_path)

# Configure CUDA
device = torch.device('cuda')
if torch.cuda.is_available():
    print(f"Found GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found. Running on CPU.")

print(f"Log file created at: {log_path}")

from architectures_pytorch.helpers.constants import (
    hyperparameters,
    coin_list,
    direction,
    threshold,
    selected_model,
    enable_mae_pretrain,
    mae_weights_path,
)
from architectures_pytorch.helpers.wandb_handler import initialize_wandb
from architectures_pytorch.vision_transformer import get_vit_model
from architectures_pytorch.helpers.warmup_cosine import WarmUpCosineScheduler
from architectures_pytorch.helpers.one_cycle import OneCycleLRScheduler

# Import CCT model if available
try:
    from architectures_pytorch.compact_conv_transformer import get_cct_model
    CCT_AVAILABLE = True
except ImportError:
    CCT_AVAILABLE = False

# Import ViT Small-Dataset model if available
try:
    from architectures_pytorch.vit_small_dataset import get_vit_small_model
    VIT_SMALL_AVAILABLE = True
except ImportError:
    VIT_SMALL_AVAILABLE = False

hyperparameters = hyperparameters[selected_model]
t = time.time()
epoch_counter = 1
print(coin_list)
print(direction)
print(threshold)
print(hyperparameters)

def make_datasets(images, labels, shuffle=False):
    # Reshape images to [B, C, H, W] format for PyTorch
    if len(images.shape) == 3:  # If shape is [B, H, W]
        images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2])
    elif len(images.shape) == 4 and images.shape[-1] == 1:  # If shape is [B, H, W, C]
        images = images.transpose(0, 3, 1, 2)
    # Convert to bfloat16 format
    images_tensor = torch.tensor(images, dtype=torch.bfloat16)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(images_tensor, labels_tensor)
    return DataLoader(
        dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )

def get_finalized_datasets(x_train, y_train, x_val, y_val, x_test, y_test):
    train_dataset = make_datasets(x_train, y_train, shuffle=True)
    val_dataset = make_datasets(x_val, y_val, shuffle=True)
    test_dataset = make_datasets(x_test, y_test, shuffle=False)
    return train_dataset, val_dataset, test_dataset

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = total_loss / len(test_loader)
    
    # Calculate per-class accuracy for multi-class classification
    class_accuracies = {}
    for class_idx in range(hyperparameters["num_classes"]):
        class_mask = np.array(all_labels) == class_idx
        if np.sum(class_mask) > 0:  # Avoid division by zero
            class_acc = np.mean(np.array(all_preds)[class_mask] == class_idx)
            class_accuracies[class_idx] = class_acc
            print(f"  Class {class_idx} accuracy: {class_acc:.4f} (samples: {np.sum(class_mask)})")
    
    print(f"Per-class accuracies: {class_accuracies}")
    
    return avg_loss, accuracy, all_preds, all_labels

def run_experiment(model, train_dataset, val_dataset, test_dataset, save_epochs=True):
    # Инициализируем wandb для логирования метрик
    try:
        initialize_wandb()
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
        print("Training will continue without wandb logging")
    
    # Convert model to bfloat16
    model = model.to(torch.bfloat16)
    model = model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=hyperparameters["weight_decay"])
 
    #optimizer = optim.Adadelta(model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=hyperparameters["weight_decay"])

    
    if hyperparameters["learning_rate_type"] == "WarmUpCosine":
        total_steps = len(train_dataset) * hyperparameters["num_epochs"]
        scheduler = WarmUpCosineScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=10000,
            warmup_learning_rate=0.0
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.75,
            patience=3,
            verbose=True
        )
    
    best_val_loss = float('inf')
    best_model_state = None
    
    try:
        for epoch in range(hyperparameters["num_epochs"]):
            model.train()
            train_loss = 0
            train_steps = 0
            
            # Training loop
            progress_bar = tqdm(train_dataset, desc=f'Epoch {epoch + 1}/{hyperparameters["num_epochs"]}')
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if scheduler is not None and hyperparameters["learning_rate_type"] == "WarmUpCosine":
                    scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                progress_bar.set_postfix({'loss': train_loss / train_steps})
            
            # Validation
            val_loss, val_accuracy, val_all_preds, val_all_labels = evaluate_model(model, val_dataset)
            test_loss, test_accuracy, test_all_preds, test_all_labels = evaluate_model(model, test_dataset)
            
            # Step scheduler for ReduceLROnPlateau
            if scheduler is not None and hyperparameters["learning_rate_type"] != "WarmUpCosine":
                scheduler.step(val_loss)
            
            # Log metrics
            try:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss / train_steps,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            except Exception as e:
                print(f"Warning: Could not log to wandb: {e}")
                # Продолжаем обучение даже если логирование не работает
            
            print(f'Epoch {epoch + 1}: Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
            
            print("\nConfusion Matrix Val:")
            conf_m_val = confusion_matrix(val_all_labels, val_all_preds)
            print(conf_m_val)
            print("\nTP Success Ratio: ", conf_m_val[:, 1][1] / (conf_m_val[:, 1][0] + conf_m_val[:, 1][1]) * 100)
            print("\nConfusion Matrix Test:")
            conf_m_test = confusion_matrix(test_all_labels, test_all_preds)
            print(conf_m_test)
            print("\nTP Success Ratio: ", conf_m_test[:, 1][1] / (conf_m_test[:, 1][0] + conf_m_test[:, 1][1]) * 100)
            
            # Save model for each epoch if enabled
            if save_epochs:
                # Create models directory if it doesn't exist
                pathlib.Path('models').mkdir(exist_ok=True)
                # Save model for current epoch
                epoch_model_path = f'models/{selected_model}_{direction}_epoch_{epoch+1}_{int(time.time())}.pth'
                torch.save(model.state_dict(), epoch_model_path)
                print(f"Model saved for epoch {epoch+1} at {epoch_model_path}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                
                # Save best model so far if save_epochs is enabled
                if save_epochs:
                    best_model_path = f'models/{selected_model}_{direction}_best_{int(time.time())}.pth'
                    torch.save(best_model_state, best_model_path)
                    print(f"New best model saved at {best_model_path}")
                
    except Exception as e:
        print(f"Error during training: {e}")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return model
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def load_and_prepare_data(val_size=0.2, test_size=0.1, random_state=42, filter_step=None):
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
    
    # Convert labels to multi-class classification (0, 1, 2)
    multiclass_labels = labels.astype(np.int64)
    print(f"Multi-class labels distribution: {np.bincount(multiclass_labels)}")
    
    # Split data into train+validation and test sets
    # The test set will be chronologically at the end and not shuffled
    x_trainval, x_test, y_trainval, y_test, dates_trainval, dates_test = train_test_split(
        matrices, multiclass_labels, dates, 
        test_size=test_size, 
        random_state=random_state, 
        shuffle=False  # Keep chronological order
    )
    
    # Now split train+validation into train and validation sets with shuffling
    # This will mix the train and validation data together
    relative_val_size = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val, dates_train, dates_val = train_test_split(
        x_trainval, y_trainval, dates_trainval,
        test_size=relative_val_size,
        random_state=random_state,
        shuffle=False  # Mix train and validation data
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
    
    # Print earliest and latest dates for each dataset
    print(f"\nChronological date ranges:")
    print(f"Training set: earliest date = {min(dates_train)}, latest date = {max(dates_train)}")
    print(f"Validation set: earliest date = {min(dates_val)}, latest date = {max(dates_val)}")
    print(f"Test set: earliest date = {min(dates_test)}, latest date = {max(dates_test)}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test, dates_train, dates_val, dates_test

if __name__ == '__main__':
    # Print start time
    print(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize wandb
    initialize_wandb()
    
    # Load and prepare data - use filter_step=10 to select every 10th element
    x_train, y_train, x_val, y_val, x_test, y_test, dates_train, dates_val, dates_test = load_and_prepare_data(filter_step=3)

    # Optional MAE pretraining on training images (labels not used)
    if enable_mae_pretrain and selected_model in ["vision_transformer", "vit", "vit_small"]:
        print("Starting MAE pretraining...")
        from architectures_pytorch.mae_pretrain import pretrain_mae
        pretrain_mae(x_train, epochs=hyperparameters.get("mae_epochs_pretrain", 50))
        load_pretrained = True
    else:
        # If pretraining disabled but weights file already exists, reuse it
        if pathlib.Path(mae_weights_path).is_file():
            print(f"Found existing MAE weights at {mae_weights_path}. Will load them into ViT encoder.")
            load_pretrained = True
        else:
            load_pretrained = False
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = get_finalized_datasets(
        x_train, y_train, x_val, y_val, x_test, y_test
    )
    
    # Get model based on selected_model in constants.py
    if selected_model == "cct" and CCT_AVAILABLE:
        print(f"Using model: {selected_model}")
        model = get_cct_model(load_weights=False, weights_path=None)
    elif selected_model == "vit_small" and VIT_SMALL_AVAILABLE:
        print(f"Using model: {selected_model}")
        model = get_vit_small_model(load_weights=False, weights_path=None)
    else:
        if selected_model not in ["vision_transformer", "vit"]:
            print(f"Warning: Unknown model '{selected_model}', falling back to Vision Transformer")
        print(f"Using model: Vision Transformer")
        model = get_vit_model(load_weights=load_pretrained, weights_path=mae_weights_path if load_pretrained else None)
    
    # Train model
    trained_model = run_experiment(model, train_dataset, val_dataset, test_dataset)
    
    # Save final model
    pathlib.Path('models').mkdir(exist_ok=True)
    final_model_path = f'models/{selected_model}_{direction}_final_{int(time.time())}.pth'
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

    # Final evaluation
    test_loss, test_accuracy, test_all_preds, test_all_labels = evaluate_model(trained_model, test_dataset)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    
    # Add target names for better readability in the classification report
    target_names = [f"Class {i}" for i in range(hyperparameters["num_classes"])]
    print(classification_report(test_all_labels, test_all_preds, target_names=target_names))
    
    # Print confusion matrix for multi-class classification
    cm = confusion_matrix(test_all_labels, test_all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save predictions and actual labels for further analysis
    results = {
        'predictions': test_all_preds,
        'actual': test_all_labels,
        'dates': dates_test
    }
    
    with open(f'results/{selected_model}_{direction}_{int(time.time())}.pickle', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Total execution time: {time.time() - t:.2f} seconds")
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log saved to: {log_path}")
    
    # Close the log file and restore stdout
    if isinstance(sys.stdout, Logger):
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal