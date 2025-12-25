"""
Testing and inference module.
Handles model evaluation and prediction.
"""

import torch
import numpy as np
from pathlib import Path
import torchaudio
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def test_model(pl_module, test_loader, device='cpu'):
    """
    Test the model on the test set.
    
    Args:
        pl_module: Trained PyTorch Lightning module
        test_loader: Test data loader
        device: Device to run inference on
        
    Returns:
        Dictionary with test metrics
    """
    pl_module.eval()
    pl_module.to(device)
    
    all_predictions = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = pl_module(inputs)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)
    
    # Calculate accuracy
    accuracy = (all_predictions == all_targets).mean()
    
    # Calculate top-5 accuracy
    top5_preds = np.argsort(all_outputs, axis=1)[:, -5:]
    top5_acc = np.mean([target in top5_preds[i] 
                        for i, target in enumerate(all_targets)])
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_acc,
        'predictions': all_predictions,
        'targets': all_targets,
        'outputs': all_outputs
    }


def predict_single_file(pl_module, audio_path, feature_extractor, 
                       class_names, device='cpu', sample_rate=16000):
    """
    Make prediction on a single audio file.
    
    Args:
        pl_module: Trained PyTorch Lightning module
        audio_path: Path to audio file
        feature_extractor: Feature extraction function
        class_names: List of class names
        device: Device to run inference on
        sample_rate: Expected sample rate
        
    Returns:
        Predicted class name and confidence
    """
    pl_module.eval()
    pl_module.to(device)
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Pad or trim to 1 second
    target_length = sample_rate
    if waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    
    # Extract features
    if feature_extractor is not None:
        features = feature_extractor(waveform)
    else:
        features = waveform
    
    # Add batch dimension
    features = features.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = pl_module(features)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_value = confidence.item()
    
    print(f"Predicted: {predicted_class} (confidence: {confidence_value:.4f})")
    
    return predicted_class, confidence_value


def predict_batch(pl_module, file_paths, feature_extractor, 
                 class_names, device='cpu', sample_rate=16000):
    """
    Make predictions on a batch of audio files.
    
    Args:
        pl_module: Trained PyTorch Lightning module
        file_paths: List of paths to audio files
        feature_extractor: Feature extraction function
        class_names: List of class names
        device: Device to run inference on
        sample_rate: Expected sample rate
        
    Returns:
        List of (predicted_class, confidence) tuples
    """
    results = []
    
    for file_path in file_paths:
        try:
            predicted_class, confidence = predict_single_file(
                pl_module, file_path, feature_extractor, 
                class_names, device, sample_rate
            )
            results.append((predicted_class, confidence))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            results.append((None, 0.0))
    
    return results


def generate_classification_report(predictions, targets, class_names, 
                                   save_path=None):
    """
    Generate and print classification report.
    
    Args:
        predictions: Array of predictions
        targets: Array of ground truth labels
        class_names: List of class names
        save_path: Optional path to save the report
        
    Returns:
        Classification report as string
    """
    report = classification_report(targets, predictions, 
                                  target_names=class_names,
                                  digits=4)
    
    print("\nClassification Report:")
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {save_path}")
    
    return report


def plot_confusion_matrix(predictions, targets, class_names, 
                         save_path=None, figsize=(15, 12)):
    """
    Plot confusion matrix.
    
    Args:
        predictions: Array of predictions
        targets: Array of ground truth labels
        class_names: List of class names
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def evaluate_and_report(pl_module, test_loader, class_names, 
                       output_dir='results', device='cpu'):
    """
    Complete evaluation with reports and visualizations.
    
    Args:
        pl_module: Trained PyTorch Lightning module
        test_loader: Test data loader
        class_names: List of class names
        output_dir: Directory to save results
        device: Device to run inference on
        
    Returns:
        Test results dictionary
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Run test
    results = test_model(pl_module, test_loader, device)
    
    # Generate classification report
    report_path = os.path.join(output_dir, 'classification_report.txt')
    generate_classification_report(
        results['predictions'], 
        results['targets'], 
        class_names,
        save_path=report_path
    )
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        results['predictions'],
        results['targets'],
        class_names,
        save_path=cm_path
    )
    
    return results

