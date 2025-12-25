"""
Preprocessing module for Google Speech Commands dataset.
Handles data loading, splitting, and augmentation.
"""

import os
import re
import hashlib
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1


def which_set(filename, validation_percentage=10, testing_percentage=10):
    """
    Determines which data partition the file should belong to.
    Uses hash-based stable assignment.
    
    Args:
        filename: File path of the data sample
        validation_percentage: Percentage for validation set
        testing_percentage: Percentage for testing set
        
    Returns:
        String: 'training', 'validation', or 'testing'
    """
    base_name = os.path.basename(filename)
    # Ignore anything after '_nohash_' for grouping files from same speaker
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        return 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        return 'testing'
    else:
        return 'training'


class SpeechCommandsDataset(Dataset):
    """
    Dataset class for Google Speech Commands.
    Handles loading, preprocessing, and augmentation of audio files.
    """
    
    def __init__(self, data_dir, split='training', feature_extractor=None, 
                 classes=None, use_background_noise=False, sample_rate=16000):
        """
        Args:
            data_dir: Path to the speech commands dataset
            split: 'training', 'validation', or 'testing'
            feature_extractor: Feature extraction function/object
            classes: List of classes to use (None = all classes)
            use_background_noise: Whether to add background noise augmentation
            sample_rate: Expected sample rate (default 16000 Hz)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.feature_extractor = feature_extractor
        self.use_background_noise = use_background_noise
        self.sample_rate = sample_rate
        
        # Get all classes (folder names except _background_noise_, checkpoints, results)
        exclude_dirs = {'_background_noise_', 'checkpoints', 'results'}
        all_classes = sorted([d.name for d in self.data_dir.iterdir() 
                            if d.is_dir() and not d.name.startswith('_') 
                            and not d.name.startswith('.') 
                            and d.name not in exclude_dirs])
        
        if classes is None:
            self.classes = all_classes
        else:
            self.classes = classes
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load file lists
        self.file_list = self._load_file_list()
        print(f"{split.capitalize()} set: {len(self.file_list)} files across {len(self.classes)} classes")
        
        # Load background noise files if needed
        if use_background_noise and split == 'training':
            self.background_noise = self._load_background_noise()
        else:
            self.background_noise = None
    
    def _load_file_list(self):
        """Load list of files for this split based on provided text files."""
        validation_list = set()
        testing_list = set()
        
        val_file = self.data_dir / 'validation_list.txt'
        test_file = self.data_dir / 'testing_list.txt'
        
        if val_file.exists():
            with open(val_file, 'r') as f:
                validation_list = set(line.strip() for line in f)
        
        if test_file.exists():
            with open(test_file, 'r') as f:
                testing_list = set(line.strip() for line in f)
        
        # Collect all wav files
        file_list = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            for wav_file in class_dir.glob('*.wav'):
                rel_path = f"{class_name}/{wav_file.name}"
                
                # Determine which set this file belongs to
                if rel_path in validation_list:
                    file_split = 'validation'
                elif rel_path in testing_list:
                    file_split = 'testing'
                else:
                    file_split = 'training'
                
                if file_split == self.split:
                    file_list.append({
                        'path': wav_file,
                        'label': class_name,
                        'label_idx': self.class_to_idx[class_name]
                    })
        
        return file_list
    
    def _load_background_noise(self):
        """Load background noise files for augmentation."""
        noise_dir = self.data_dir / '_background_noise_'
        if not noise_dir.exists():
            print("Warning: Background noise directory not found")
            return None
        
        noise_files = []
        for wav_file in noise_dir.glob('*.wav'):
            try:
                # Convert Path to string for torchaudio compatibility
                waveform, sr = torchaudio.load(str(wav_file))
                if sr != self.sample_rate:
                    waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
                noise_files.append(waveform)
            except Exception as e:
                print(f"Warning: Could not load {wav_file}: {e}")
                continue
        
        if noise_files:
            print(f"Loaded {len(noise_files)} background noise files")
        return noise_files if noise_files else None
    
    def _add_background_noise(self, waveform, noise_level=0.1):
        """Add background noise to the waveform for augmentation."""
        if self.background_noise is None or len(self.background_noise) == 0:
            return waveform
        
        # Randomly select a noise file
        noise = self.background_noise[np.random.randint(len(self.background_noise))]
        
        # Extract a random segment from the noise
        if noise.shape[1] > waveform.shape[1]:
            start_idx = np.random.randint(0, noise.shape[1] - waveform.shape[1])
            noise_segment = noise[:, start_idx:start_idx + waveform.shape[1]]
        else:
            noise_segment = noise
            if noise_segment.shape[1] < waveform.shape[1]:
                # Repeat noise if too short
                repeats = (waveform.shape[1] // noise_segment.shape[1]) + 1
                noise_segment = noise_segment.repeat(1, repeats)[:, :waveform.shape[1]]
        
        # Match channels
        if noise_segment.shape[0] != waveform.shape[0]:
            noise_segment = noise_segment.mean(dim=0, keepdim=True)
        
        # Add noise with random level
        noise_factor = np.random.uniform(0, noise_level)
        return waveform + noise_factor * noise_segment
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """Load and process a single audio file."""
        item = self.file_list[idx]
        
        # Load audio file (convert Path to string for torchaudio compatibility)
        waveform, sr = torchaudio.load(str(item['path']))
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad or trim to exactly 1 second
        target_length = self.sample_rate
        if waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        
        # Add background noise for training (80% of the time)
        if self.use_background_noise and self.split == 'training':
            if np.random.random() < 0.8:
                waveform = self._add_background_noise(waveform)
        
        # Extract features if feature extractor is provided
        if self.feature_extractor is not None:
            features = self.feature_extractor(waveform)
        else:
            features = waveform
        
        return features, item['label_idx']


def get_dataloaders(data_dir, batch_size=32, num_workers=4, feature_extractor=None, 
                   classes=None, use_background_noise=True):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        feature_extractor: Feature extraction function
        classes: List of classes (None = all)
        use_background_noise: Whether to use background noise augmentation
        
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    
    train_dataset = SpeechCommandsDataset(
        data_dir, 
        split='training', 
        feature_extractor=feature_extractor,
        classes=classes,
        use_background_noise=use_background_noise
    )
    
    val_dataset = SpeechCommandsDataset(
        data_dir, 
        split='validation', 
        feature_extractor=feature_extractor,
        classes=classes,
        use_background_noise=False
    )
    
    test_dataset = SpeechCommandsDataset(
        data_dir, 
        split='testing', 
        feature_extractor=feature_extractor,
        classes=classes,
        use_background_noise=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes

