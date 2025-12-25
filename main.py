"""
Main entry point for speech commands classification.
Run with: python main.py (for training)
Run with: python main.py -t (for testing/inference)
"""

import argparse
import torch
import os
from pathlib import Path

from preprocessing import get_dataloaders
from feature import get_feature_extractor
from network import get_model
from train import train_model
from callbacks import get_callbacks
from test import evaluate_and_report, predict_single_file


def main():
    """Main function to run training or testing."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Speech Commands Classification')
    parser.add_argument('-t', '--test', action='store_true', 
                       help='Run in test/inference mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint for testing')
    parser.add_argument('--audio_file', type=str, default=None,
                       help='Path to single audio file for inference')
    args = parser.parse_args()
    
    # Configuration
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    MAX_EPOCHS = 50
    LEARNING_RATE = 0.001
    FEATURE_TYPE = 'mfcc'  # Options: 'mfcc', 'melspectrogram', 'log_mel', 'raw'
    MODEL_NAME = 'simple_cnn'  # Options: 'simple_cnn', 'deep_cnn', 'resnet', 'attention_cnn', 'efficient_cnn'
    CHECKPOINT_DIR = 'checkpoints'
    RESULTS_DIR = 'results'
    
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
        gpus = 1
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        # Note: MPS has compatibility issues with PyTorch Lightning, using CPU
        device = 'cpu'
        gpus = None
        if torch.backends.mps.is_available():
            print("Apple Silicon GPU detected but using CPU (MPS not fully compatible with Lightning)")
        else:
            print("Using CPU")
    
    # Create feature extractor
    print(f"\nFeature extraction: {FEATURE_TYPE}")
    feature_extractor = get_feature_extractor(feature_type=FEATURE_TYPE)
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        feature_extractor=feature_extractor,
        classes=None,  # Use all classes
        use_background_noise=True
    )
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names[:10]}..." if len(class_names) > 10 else f"Classes: {class_names}")
    
    # Get input shape from a sample batch
    sample_batch = next(iter(train_loader))
    sample_input = sample_batch[0]
    input_channels = sample_input.shape[1]
    print(f"Input shape: {sample_input.shape}")
    
    if args.test:
        # TESTING MODE
        print("\n" + "="*50)
        print("TESTING MODE")
        print("="*50)
        
        # Load checkpoint
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            # Find latest checkpoint
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'last.ckpt')
            if not os.path.exists(checkpoint_path):
                print(f"Error: No checkpoint found at {checkpoint_path}")
                print("Please specify checkpoint with --checkpoint or train a model first")
                return
        
        print(f"\nLoading checkpoint: {checkpoint_path}")
        
        # Create model
        model = get_model(
            model_name=MODEL_NAME,
            num_classes=num_classes,
            input_channels=input_channels
        )
        
        # Load trained module
        from train import SpeechCommandsModule
        pl_module = SpeechCommandsModule.load_from_checkpoint(
            checkpoint_path,
            model=model
        )
        pl_module.to(device)
        pl_module.eval()
        
        if args.audio_file:
            # Single file inference
            print(f"\nPredicting for: {args.audio_file}")
            predicted_class, confidence = predict_single_file(
                pl_module=pl_module,
                audio_path=args.audio_file,
                feature_extractor=feature_extractor,
                class_names=class_names,
                device=device
            )
            print(f"\nResult: {predicted_class} (confidence: {confidence:.4f})")
        else:
            # Full test set evaluation
            print("\nEvaluating on test set...")
            results = evaluate_and_report(
                pl_module=pl_module,
                test_loader=test_loader,
                class_names=class_names,
                output_dir=RESULTS_DIR,
                device=device
            )
            
            print(f"\nFinal Test Accuracy: {results['accuracy']:.4f}")
            print(f"Final Test Top-5 Accuracy: {results['top5_accuracy']:.4f}")
            print(f"\nResults saved to {RESULTS_DIR}/")
    
    else:
        # TRAINING MODE
        print("\n" + "="*50)
        print("TRAINING MODE")
        print("="*50)
        
        # Create model
        print(f"\nCreating model: {MODEL_NAME}")
        model = get_model(
            model_name=MODEL_NAME,
            num_classes=num_classes,
            input_channels=input_channels,
            dropout=0.5
        )
        
        # Print model summary
        print(f"\nModel architecture:")
        print(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Get callbacks
        callbacks = get_callbacks(
            checkpoint_dir=CHECKPOINT_DIR,
            monitor='val_acc',
            patience=15,
            mode='max'
        )
        
        # Train model
        print("\nStarting training...")
        print(f"Max epochs: {MAX_EPOCHS}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Batch size: {BATCH_SIZE}")
        
        pl_module, trainer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            max_epochs=MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
            optimizer='adam',
            scheduler='reduce_on_plateau',
            callbacks=callbacks,
            gpus=gpus,
            precision=32
        )
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)
        
        # Test on test set
        print("\nEvaluating on test set...")
        results = evaluate_and_report(
            pl_module=pl_module,
            test_loader=test_loader,
            class_names=class_names,
            output_dir=RESULTS_DIR,
            device=device
        )
        
        print(f"\nFinal Test Accuracy: {results['accuracy']:.4f}")
        print(f"Final Test Top-5 Accuracy: {results['top5_accuracy']:.4f}")
        print(f"\nModel saved to {CHECKPOINT_DIR}/")
        print(f"Results saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()

