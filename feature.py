"""
Feature extraction module for audio processing.
Implements various feature extractors for speech commands.
"""

import torch
import torchaudio
import torch.nn as nn


class MFCCFeatureExtractor:
    """Extract MFCC (Mel-Frequency Cepstral Coefficients) features from waveform."""
    
    def __init__(self, sample_rate=16000, n_mfcc=40, melkwargs=None):
        """
        Args:
            sample_rate: Sample rate of audio
            n_mfcc: Number of MFCC coefficients
            melkwargs: Arguments for MelSpectrogram computation
        """
        if melkwargs is None:
            melkwargs = {
                'n_fft': 512,
                'hop_length': 160,  # 10ms hop
                'n_mels': 40,
                'f_min': 20,
                'f_max': 4000
            }
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs=melkwargs
        )
    
    def __call__(self, waveform):
        """
        Extract MFCC features from waveform.
        
        Args:
            waveform: Input waveform tensor (channels, samples)
            
        Returns:
            MFCC features (channels, n_mfcc, time)
        """
        mfcc = self.mfcc_transform(waveform)
        return mfcc


class MelSpectrogramFeatureExtractor:
    """Extract Mel Spectrogram features from waveform."""
    
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, 
                 n_mels=40, f_min=20, f_max=4000):
        """
        Args:
            sample_rate: Sample rate of audio
            n_fft: FFT size
            hop_length: Hop length for STFT
            n_mels: Number of mel filterbanks
            f_min: Minimum frequency
            f_max: Maximum frequency
        """
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def __call__(self, waveform):
        """
        Extract Mel Spectrogram features from waveform.
        
        Args:
            waveform: Input waveform tensor (channels, samples)
            
        Returns:
            Mel spectrogram in dB (channels, n_mels, time)
        """
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        return mel_spec_db


class RawWaveformExtractor:
    """Return raw waveform without feature extraction."""
    
    def __call__(self, waveform):
        """
        Return raw waveform.
        
        Args:
            waveform: Input waveform tensor
            
        Returns:
            Same waveform
        """
        return waveform


class LogMelSpectrogramExtractor:
    """Extract log mel-spectrogram features (similar to TensorFlow's approach)."""
    
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, 
                 n_mels=40, f_min=20, f_max=4000):
        """
        Args:
            sample_rate: Sample rate of audio
            n_fft: FFT size
            hop_length: Hop length for STFT
            n_mels: Number of mel filterbanks
            f_min: Minimum frequency
            f_max: Maximum frequency
        """
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
    
    def __call__(self, waveform):
        """
        Extract log mel-spectrogram features.
        
        Args:
            waveform: Input waveform tensor (channels, samples)
            
        Returns:
            Log mel spectrogram (channels, n_mels, time)
        """
        mel_spec = self.mel_transform(waveform)
        # Add small epsilon to avoid log(0)
        log_mel_spec = torch.log(mel_spec + 1e-9)
        return log_mel_spec


def get_feature_extractor(feature_type='mfcc', **kwargs):
    """
    Factory function to get feature extractor.
    
    Args:
        feature_type: Type of features ('mfcc', 'melspectrogram', 'log_mel', 'raw')
        **kwargs: Additional arguments for the feature extractor
        
    Returns:
        Feature extractor object
    """
    if feature_type == 'mfcc':
        return MFCCFeatureExtractor(**kwargs)
    elif feature_type == 'melspectrogram':
        return MelSpectrogramFeatureExtractor(**kwargs)
    elif feature_type == 'log_mel':
        return LogMelSpectrogramExtractor(**kwargs)
    elif feature_type == 'raw':
        return RawWaveformExtractor()
    else:
        raise ValueError(f"Unknown feature type: {feature_type}. "
                        f"Choose from: 'mfcc', 'melspectrogram', 'log_mel', 'raw'")

