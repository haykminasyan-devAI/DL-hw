# Speech Commands Classification - PyTorch Lightning Implementation

This is a complete PyTorch Lightning implementation for training a keyword spotting classifier on the Google Speech Commands v2 dataset.

## Dataset

The [Google Speech Commands v2 dataset](https://arxiv.org/abs/1804.03209) contains 105,829 one-second audio files of spoken words from 35 different classes, including:
- **Core commands**: yes, no, up, down, left, right, on, off, stop, go
- **Digits**: zero, one, two, three, four, five, six, seven, eight, nine
- **Auxiliary words**: bed, bird, cat, dog, happy, house, marvin, sheila, tree, wow, backward, forward, follow, learn, visual

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The dataset should already be in the current directory. If not, download it:
```bash
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz
```

## Usage

### Training

To train a model, simply run:
```bash
python main.py
```

This will:
- Load the dataset with train/validation/test splits
- Create a SimpleCNN model
- Train for 50 epochs with early stopping
- Save checkpoints to `checkpoints/`
- Evaluate on the test set
- Save results to `results/`

### Testing/Inference

To test a trained model on the test set:
```bash
python main.py -t
```

To test on a single audio file:
```bash
python main.py -t --audio_file path/to/audio.wav
```

To use a specific checkpoint:
```bash
python main.py -t --checkpoint checkpoints/best-model.ckpt
```

## Code Structure

### Core Modules

1. **preprocessing.py** - Data loading and augmentation
   - `SpeechCommandsDataset`: PyTorch Dataset for loading audio files
   - `get_dataloaders()`: Creates train/val/test dataloaders
   - Background noise augmentation support

2. **feature.py** - Audio feature extraction
   - `MFCCFeatureExtractor`: MFCC features (default)
   - `MelSpectrogramFeatureExtractor`: Mel spectrogram features
   - `LogMelSpectrogramExtractor`: Log mel spectrogram
   - `RawWaveformExtractor`: No feature extraction (raw audio)

3. **network.py** - Neural network architectures
   - `SimpleCNN`: Basic 3-layer CNN (good baseline)
   - `DeepCNN`: Deeper 5-layer CNN (better accuracy)
   - `ResNet`: ResNet-style with residual connections
   - `AttentionCNN`: CNN with attention mechanism
   - `EfficientCNN`: Lightweight model with separable convolutions

4. **custom_layers.py** - Custom neural network layers
   - `AttentionLayer`: Self-attention for feature aggregation
   - `ResidualBlock`: Residual connections
   - `SeparableConv2d`: Depthwise separable convolutions
   - `SqueezeExcitation`: Channel attention
   - Temporal pooling layers

5. **train.py** - Training logic with PyTorch Lightning
   - `SpeechCommandsModule`: Lightning module with training/validation/test steps
   - `train_model()`: Main training function
   - Support for multiple optimizers and schedulers

6. **callbacks.py** - Training callbacks
   - Model checkpointing (save best model)
   - Early stopping
   - Learning rate monitoring
   - Metrics logging and printing

7. **test.py** - Testing and inference
   - `test_model()`: Evaluate on test set
   - `predict_single_file()`: Single file prediction
   - `evaluate_and_report()`: Generate classification reports and confusion matrix

8. **main.py** - Entry point (DO NOT MODIFY)
   - Handles command-line arguments
   - Orchestrates training or testing

## Configuration

You can modify these parameters in `main.py`:

```python
BATCH_SIZE = 128          # Batch size for training
NUM_WORKERS = 4           # Data loading workers
MAX_EPOCHS = 50           # Maximum training epochs
LEARNING_RATE = 0.001     # Initial learning rate
FEATURE_TYPE = 'mfcc'     # Feature extraction type
MODEL_NAME = 'simple_cnn' # Model architecture
```

### Available Feature Types
- `'mfcc'`: Mel-Frequency Cepstral Coefficients (recommended)
- `'melspectrogram'`: Mel spectrogram
- `'log_mel'`: Log mel spectrogram
- `'raw'`: Raw waveform

### Available Models
- `'simple_cnn'`: Simple baseline CNN (fast, good for testing)
- `'deep_cnn'`: Deeper CNN (better accuracy)
- `'resnet'`: ResNet-style architecture (best accuracy)
- `'attention_cnn'`: CNN with attention
- `'efficient_cnn'`: Lightweight model (fast inference)

## Expected Results

Based on the paper's baseline results, you should expect:
- **Simple CNN**: ~85-88% accuracy
- **Deep CNN**: ~88-90% accuracy
- **ResNet/Attention**: ~90-92% accuracy

Training time varies:
- CPU: ~2-4 hours per epoch
- GPU: ~5-10 minutes per epoch

## Files Generated

After training:
- `checkpoints/best-model-*.ckpt`: Best model checkpoint
- `checkpoints/last.ckpt`: Last model checkpoint
- `results/classification_report.txt`: Detailed classification metrics
- `results/confusion_matrix.png`: Confusion matrix visualization

## Tips for Better Performance

1. **Use a GPU**: Training is much faster on GPU
2. **Try different models**: Start with `simple_cnn`, then try `deep_cnn` or `resnet`
3. **Experiment with features**: MFCC works well, but try `log_mel` too
4. **Adjust hyperparameters**: Learning rate, batch size, dropout
5. **Data augmentation**: Background noise is enabled by default (helps a lot!)

## Troubleshooting

**Out of memory error:**
- Reduce `BATCH_SIZE` to 64 or 32
- Use a smaller model like `simple_cnn`

**Slow training:**
- Reduce `NUM_WORKERS` if CPU bottleneck
- Use GPU if available
- Try `efficient_cnn` model

**Poor accuracy:**
- Train for more epochs (increase `MAX_EPOCHS`)
- Try different model architecture
- Check that background noise augmentation is enabled
- Ensure dataset is properly loaded (check train/val/test splits)

## References

- Paper: [Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209)
- Dataset: [Google Speech Commands v2](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
- PyTorch: https://pytorch.org/
- PyTorch Lightning: https://lightning.ai/

## License

This implementation follows the Creative Commons BY 4.0 license of the dataset.

